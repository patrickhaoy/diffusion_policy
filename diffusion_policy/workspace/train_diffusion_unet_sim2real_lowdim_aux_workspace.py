if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetSim2RealLowdimAuxWorkspace(BaseWorkspace):
    """
    Trains DiffusionUnetLowdimPolicy on Sim2RealImageMultiDataset.
    Bridges the format gap: the dataset returns per-key obs dicts,
    but the diffusion policy expects flat obs tensors.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

        obs_shape_meta = cfg.shape_meta['obs']
        self.obs_keys = sorted([
            k for k, v in obs_shape_meta.items()
            if v.get('type', 'low_dim') == 'low_dim'
        ])
        self.obs_key_dims = {
            k: obs_shape_meta[k]['shape'][0] for k in self.obs_keys
        }

    def _flatten_obs(self, obs_dict):
        """Convert per-key obs dict to flat tensor. (B, T, obs_dim)"""
        parts = []
        for key in self.obs_keys:
            parts.append(obs_dict[key])
        return torch.cat(parts, dim=-1)

    def _flatten_batch(self, batch):
        """Convert dataset batch format to flat format for diffusion policy."""
        flat_obs = self._flatten_obs(batch['obs'])
        return {
            'obs': flat_obs,
            'action': batch['action'],
        }

    def _build_flat_normalizer(self, dataset_normalizer):
        """
        Build a flat obs+action LinearNormalizer from the per-key normalizer
        returned by Sim2RealImageMultiDataset.
        """
        scales = []
        offsets = []
        input_mins = []
        input_maxs = []
        input_means = []
        input_stds = []

        for key in self.obs_keys:
            field_norm = dataset_normalizer[key]
            params = field_norm.params_dict
            dim = self.obs_key_dims[key]
            s = params['scale']
            o = params['offset']
            if s.numel() == 1 and dim > 1:
                s = s.expand(dim)
                o = o.expand(dim)
            scales.append(s.flatten())
            offsets.append(o.flatten())
            stats = params['input_stats']
            for name, lst in [('min', input_mins), ('max', input_maxs),
                              ('mean', input_means), ('std', input_stds)]:
                v = stats[name]
                if v.numel() == 1 and dim > 1:
                    v = v.expand(dim)
                lst.append(v.flatten())

        flat_scale = torch.cat(scales)
        flat_offset = torch.cat(offsets)
        flat_stats = {
            'min': torch.cat(input_mins),
            'max': torch.cat(input_maxs),
            'mean': torch.cat(input_means),
            'std': torch.cat(input_stds),
        }
        obs_normalizer = SingleFieldLinearNormalizer.create_manual(
            flat_scale, flat_offset, flat_stats)

        combined = LinearNormalizer()
        combined['obs'] = obs_normalizer
        combined['action'] = dataset_normalizer['action']
        return combined

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        dataset_normalizer = dataset.get_normalizer()
        flat_normalizer = self._build_flat_normalizer(dataset_normalizer)

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(flat_normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(flat_normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {"output_dir": self.output_dir},
            allow_val_change=True
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        flat_batch = self._flatten_batch(batch)
                        if train_sampling_batch is None:
                            train_sampling_batch = flat_batch

                        raw_loss = self.model.compute_loss(flat_batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                            # Step-based validation
                            val_every_n_steps = cfg.training.get('val_every_n_steps', None)
                            if val_every_n_steps is not None and self.global_step % val_every_n_steps == 0:
                                self._run_step_validation(
                                    cfg, device, val_dataloader, train_sampling_batch,
                                    wandb_run, json_logger)

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                flat_batch = self._flatten_batch(batch)
                                loss = self.model.compute_loss(flat_batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:, start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch, obs_dict, gt_action, result, pred_action, mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _run_step_validation(self, cfg, device, val_dataloader,
                             train_sampling_batch, wandb_run, json_logger):
        """Run validation at a step boundary (not epoch boundary)."""
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        val_step_log = {}

        with torch.no_grad():
            val_losses = []
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                flat_batch = self._flatten_batch(val_batch)
                loss = policy.compute_loss(flat_batch)
                val_losses.append(loss.item())
                if (cfg.training.max_val_steps is not None) \
                    and val_batch_idx >= (cfg.training.max_val_steps - 1):
                    break
            if len(val_losses) > 0:
                val_step_log['val_loss'] = np.mean(val_losses)

            # val action MSE
            val_mse_list = []
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                flat_batch = self._flatten_batch(val_batch)
                obs_dict = {'obs': flat_batch['obs']}
                gt_action = flat_batch['action']
                result = policy.predict_action(obs_dict)
                if cfg.pred_action_steps_only:
                    pred_action = result['action']
                    start = cfg.n_obs_steps - 1
                    end = start + cfg.n_action_steps
                    gt_action = gt_action[:, start:end]
                else:
                    pred_action = result['action_pred']
                val_mse_list.append(
                    torch.nn.functional.mse_loss(pred_action, gt_action).item())
                if (cfg.training.max_val_steps is not None) \
                    and val_batch_idx >= (cfg.training.max_val_steps - 1):
                    break
            if len(val_mse_list) > 0:
                val_step_log['val_action_mse_error'] = np.mean(val_mse_list)

            # train action MSE
            if train_sampling_batch is not None:
                obs_dict = {'obs': train_sampling_batch['obs']}
                gt_action = train_sampling_batch['action']
                result = policy.predict_action(obs_dict)
                if cfg.pred_action_steps_only:
                    pred_action = result['action']
                    start = cfg.n_obs_steps - 1
                    end = start + cfg.n_action_steps
                    gt_action = gt_action[:, start:end]
                else:
                    pred_action = result['action_pred']
                val_step_log['train_action_mse_error'] = \
                    torch.nn.functional.mse_loss(pred_action, gt_action).item()

        if val_step_log:
            wandb_run.log(val_step_log, step=self.global_step)
            json_logger.log(val_step_log)
        policy.train()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetSim2RealLowdimAuxWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
