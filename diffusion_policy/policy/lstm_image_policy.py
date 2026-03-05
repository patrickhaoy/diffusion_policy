from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

class LSTMImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int = 1,
            lstm_hidden_dim: int = 512,
            lstm_num_layers: int = 2,
            lstm_dropout: float = 0.0,
            mlp_hidden_dim: int = 256,
            mlp_hidden_depth: int = 2,
            aux_loss_weight: float = 0.0,
            **kwargs):
        assert n_action_steps == 1, "LSTMImagePolicy only supports n_action_steps=1"

        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.aux_loss_weight = aux_loss_weight
        self.kwargs = kwargs

        self.lstm = nn.LSTM(
            input_size=obs_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_ln = nn.LayerNorm(lstm_hidden_dim)

        # Post-LSTM MLP
        mlp_layers = []
        last_dim = lstm_hidden_dim
        for _ in range(mlp_hidden_depth):
            mlp_layers += [nn.Linear(last_dim, mlp_hidden_dim), nn.ReLU()]
            last_dim = mlp_hidden_dim
        self.post_lstm_mlp = nn.Sequential(*mlp_layers)

        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)
        self.log_std_limits = (-5.0, 2.0)

        # Stateful hidden state carried across predict_action calls
        self._hidden = None

        # Aux heads on single-frame encoder features (pre-LSTM)
        self.aux_heads = nn.ModuleDict()
        auxiliary_shape_meta = shape_meta.get('auxiliary_obs', None)
        if auxiliary_shape_meta is not None and aux_loss_weight > 0:
            for key, attr in auxiliary_shape_meta.items():
                dim = attr['shape'][0]
                self.aux_heads[key] = nn.Sequential(
                    nn.Linear(obs_feature_dim, lstm_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(lstm_hidden_dim, dim)
                )

    def reset(self):
        self._hidden = None

    def _post_lstm(self, lstm_out: torch.Tensor) -> torch.Tensor:
        return self.post_lstm_mlp(self.lstm_ln(lstm_out))

    def get_action_dist(self, h: torch.Tensor) -> Normal:
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=self.log_std_limits[0], max=self.log_std_limits[1])
        return Normal(mean, torch.exp(log_std))

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1, *x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, To, -1)

        lstm_out, hidden = self.lstm(nobs_features, self._hidden)
        self._hidden = tuple(h.detach() for h in hidden)
        h = self._post_lstm(lstm_out[:, -1])

        dist = self.get_action_dist(h)
        action_pred = dist.mean
        action = self.normalizer['action'].unnormalize(action_pred)
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]

        # obs may have fewer timesteps than actions when dataset_obs_steps < horizon
        # Use obs temporal dim for encoding, action temporal dim for loss
        if isinstance(nobs, dict):
            T_obs = next(iter(nobs.values())).shape[1]
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            T_obs = nobs.shape[1]
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs).reshape(B, T_obs, -1)

        # LSTM over full obs sequence (fresh hidden state each batch)
        lstm_out, _ = self.lstm(nobs_features)  # (B, T_obs, lstm_hidden)
        h = self._post_lstm(lstm_out)  # (B, T_obs, mlp_out)

        # Supervise action prediction at every obs timestep
        dist = self.get_action_dist(h)
        bc_loss = -dist.log_prob(nactions[:, :T_obs]).sum(dim=-1).mean()

        # Aux loss on last frame's encoder features
        aux_loss = torch.tensor(0.0, device=h.device)
        if self.aux_heads and 'auxiliary_obs' in batch:
            last_feat = nobs_features[:, -1]
            for key, head in self.aux_heads.items():
                pred = head(last_feat)
                aux_target = self.normalizer[key].normalize(
                    batch['auxiliary_obs'][key][:, T_obs - 1])
                aux_loss = aux_loss + F.mse_loss(pred, aux_target)

        loss = bc_loss + self.aux_loss_weight * aux_loss
        return {'loss': loss, 'bc_loss': bc_loss, 'aux_loss': aux_loss}
