from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal


class MLPLowdimPolicy(BaseImagePolicy):
    """
    MLP policy operating on low-dimensional state observations only (no images).
    Inherits BaseImagePolicy for compatibility with the existing training
    workspace and Sim2RealImageMultiDataset (which extends BaseImageDataset).
    """

    def __init__(self,
            shape_meta: dict[str, Any],
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            aux_loss_weight: float = 0.0,
            loss_type: str = "nll",
            **kwargs):
        assert n_action_steps == 1, "MLPLowdimPolicy only supports n_action_steps=1"
        assert loss_type in ("nll", "kl"), f"loss_type must be 'nll' or 'kl', got '{loss_type}'"

        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        self.lowdim_keys = sorted([
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim'
        ])
        self.obs_key_dims = {
            k: shape_meta['obs'][k]['shape'][0] for k in self.lowdim_keys
        }
        obs_feature_dim = sum(self.obs_key_dims.values())

        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.aux_loss_weight = aux_loss_weight
        self.loss_type = loss_type
        self.kwargs = kwargs

        input_dim = obs_feature_dim * n_obs_steps

        layers = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)
        self.log_std_limits = (-5.0, 2.0)

        self.aux_heads = nn.ModuleDict()
        auxiliary_shape_meta = shape_meta.get('auxiliary_obs', None)
        if auxiliary_shape_meta is not None and aux_loss_weight > 0:
            for key, attr in auxiliary_shape_meta.items():
                dim = attr['shape'][0]
                self.aux_heads[key] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                )

    def _encode_obs(self, nobs: dict, To: int) -> torch.Tensor:
        """Concatenate normalised low-dim observations into a flat vector."""
        B = next(iter(nobs.values())).shape[0]
        parts = []
        for key in self.lowdim_keys:
            parts.append(nobs[key][:, :To].reshape(B, To, -1))
        per_step = torch.cat(parts, dim=-1)  # (B, To, obs_feature_dim)
        return per_step.reshape(B, -1)  # (B, To * obs_feature_dim)

    def get_trunk_features(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs_features)

    def get_action_dist(self, h: torch.Tensor) -> Normal:
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(
            min=self.log_std_limits[0], max=self.log_std_limits[1])
        return Normal(mean, torch.exp(log_std))

    def forward(self, obs_features: torch.Tensor) -> Normal:
        return self.get_action_dist(self.get_trunk_features(obs_features))

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        To = self.n_obs_steps
        mlp_input = self._encode_obs(nobs, To)

        dist = self.forward(mlp_input)
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
        To = self.n_obs_steps
        Ta = self.n_action_steps
        Da = self.action_dim

        mlp_input = self._encode_obs(nobs, To)
        h = self.get_trunk_features(mlp_input)

        assert Ta == 1, "MLPLowdimPolicy only supports n_action_steps=1"
        student_dist = self.get_action_dist(h)

        if self.loss_type == "kl":
            assert 'expert_dist' in batch, \
                "loss_type='kl' requires expert distribution data in dataset"
            raw_mean = batch['expert_dist']['expert_action_mean'][:, To-1]
            raw_std = batch['expert_dist']['expert_action_std'][:, To-1]

            norm_expert_mean = self.normalizer['action'].normalize(raw_mean)
            action_scale = self.normalizer['action'].params_dict['scale']
            norm_expert_std = raw_std * action_scale

            expert_dist = Normal(norm_expert_mean, norm_expert_std)
            bc_loss = torch.distributions.kl_divergence(
                expert_dist, student_dist).sum(dim=-1).mean()
        else:
            target = nactions[:, To-1:To+Ta-1].squeeze()
            bc_loss = -student_dist.log_prob(target).sum(dim=-1).mean()

        aux_loss = torch.tensor(0.0, device=h.device)
        if self.aux_heads and 'auxiliary_obs' in batch:
            for key, head in self.aux_heads.items():
                pred = head(mlp_input)
                aux_target = self.normalizer[key].normalize(
                    batch['auxiliary_obs'][key][:, To-1])
                aux_loss = aux_loss + F.mse_loss(pred, aux_target)

        loss = bc_loss + self.aux_loss_weight * aux_loss
        log_std = torch.log(student_dist.scale)
        return {
            'loss': loss,
            'bc_loss': bc_loss,
            'aux_loss': aux_loss,
            'log_std_mean': log_std.mean(),
            'log_std_min': log_std.min(),
            'log_std_max': log_std.max()
        }
