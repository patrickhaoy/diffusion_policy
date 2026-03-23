from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class Sim2RealStateDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys=[],
        action_key='actions',
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()

        assert len(obs_keys) > 0, "obs_keys must be a non-empty list"

        # Load data from zarr and create replay buffer manually
        self.replay_buffer = self._create_replay_buffer_from_zarr(zarr_path, obs_keys, action_key)

        # Combine observations into a single vector
        self.obs_keys = sorted(obs_keys)
        self.action_key = action_key

        # Create combined observation data
        self._create_combined_obs()

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _create_replay_buffer_from_zarr(self, zarr_path, obs_keys, action_key):
        """Create replay buffer from flat zarr structure"""
        # Load zarr data
        root = zarr.open(zarr_path)

        # Get actions data
        actions_data = root[action_key][:]

        # Get observations data
        obs_data = {}
        for key in obs_keys:
            if key in root['obs']:
                obs_data[key] = root['obs'][key][:]

        # Get episode information from dones
        dones_data = root['dones'][:]

        # Create episode boundaries
        episode_ends = []
        current_episode_length = 0

        for i, done in enumerate(dones_data):
            current_episode_length += 1
            if done:
                episode_ends.append(i + 1)
                current_episode_length = 0

        # If the last episode doesn't end with done=True, add it
        if current_episode_length > 0:
            episode_ends.append(len(dones_data))

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Add data to replay buffer
        for key, data in obs_data.items():
            replay_buffer.root['data'][key] = data

        replay_buffer.root['data'][action_key] = actions_data

        # Add episode metadata
        replay_buffer.root['meta']['episode_ends'] = np.array(episode_ends, dtype=np.int64)

        return replay_buffer

    def _create_combined_obs(self):
        """Combine all observation keys into a single observation vector"""
        # Get the first episode to determine shapes
        episode_ends = self.replay_buffer.episode_ends
        if len(episode_ends) == 0:
            return

        # Calculate total observation dimension
        total_obs_dim = 0
        obs_shapes = {}

        for key in self.obs_keys:
            if key in self.replay_buffer:
                data = self.replay_buffer[key]
                if len(data.shape) > 1:
                    obs_shapes[key] = int(data.shape[1])
                    total_obs_dim += int(data.shape[1])
                else:
                    obs_shapes[key] = 1
                    total_obs_dim += 1

        # Create combined observation array
        total_steps = int(episode_ends[-1])
        combined_obs = np.zeros((total_steps, total_obs_dim), dtype=np.float32)

        # Fill combined observation
        current_idx = 0
        for key in self.obs_keys:
            if key in self.replay_buffer:
                data = np.array(self.replay_buffer[key])
                if len(data.shape) > 1:
                    combined_obs[:, current_idx:current_idx + data.shape[1]] = data
                    current_idx += data.shape[1]
                else:
                    combined_obs[:, current_idx] = data
                    current_idx += 1

        # Replace the replay buffer with combined data
        self.replay_buffer.root['data']['obs'] = combined_obs
        self.obs_dim = total_obs_dim

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample['obs']  # T, D_o
        action = sample[self.action_key]  # T, D_a

        data = {
            'obs': obs,
            'action': action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
