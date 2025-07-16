from typing import Dict
import torch
import numpy as np
import copy
import zarr
import os
from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class Sim2RealImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = False,
    ):
        super().__init__()
        assert os.path.isdir(dataset_path)

        # Load data and create replay buffer
        self.replay_buffer = self._create_replay_buffer_from_zarr(dataset_path, use_cache=use_cache)

        # Parse keys
        self.rgb_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'rgb']
        self.lowdim_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim']

        # Create key_first_k for performance optimization
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Split train/val
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        # Store parameters
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_ratio = val_ratio
        self.seed = seed
        self.shape_meta = shape_meta
        self.val_mask = val_mask
        self.train_mask = train_mask

    def _create_replay_buffer_from_zarr(self, dataset_path, use_cache=False):
        """Create replay buffer from zarr data"""
        z = zarr.open(dataset_path, mode='r')
        obs_group = z['data']['obs']
        action_arr = z['data']['actions']
        episode_ends = z['meta']['episode_ends']

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Add observations
        for key in obs_group.keys():
            if use_cache:
                replay_buffer.root['data'][key] = obs_group[key][:]
            else:
                replay_buffer.root['data'][key] = obs_group[key]

        # Add actions
        if use_cache:
            replay_buffer.root['data']['action'] = action_arr[:]
        else:
            replay_buffer.root['data']['action'] = action_arr

        # Add episode metadata
        if use_cache:
            replay_buffer.root['meta']['episode_ends'] = episode_ends[:]
        else:
            replay_buffer.root['meta']['episode_ends'] = episode_ends

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (np.moveaxis(data[key][T_slice], -1, 1)
                             .astype(np.float32) / 255.)
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]

        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data
