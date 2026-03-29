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
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.mlp_lowdim_policy import MLPLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainMLPLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'last_checkpoint_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: MLPLowdimPolicy = (
            hydra.utils.instantiate(cfg.policy))

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0
        self.last_checkpoint_step = 0

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
        )
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        checkpoint_loaded = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                checkpoint_loaded = True

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        dataloader_kwargs = dict(cfg.dataloader)
        if hasattr(dataset, 'weighted_sampler') and dataset.weighted_sampler is not None:
            dataloader_kwargs['sampler'] = dataset.weighted_sampler
            dataloader_kwargs.pop('shuffle', None)

        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            print("Checkpoint loaded with normalizer - preserving existing normalizer statistics")
            normalizer = self.model.normalizer
        else:
            print("Computing normalizer from dataset")
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        if self.accelerator.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = self.accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.accelerator.device
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

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss_output = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                        if isinstance(loss_output, dict):
                            raw_loss = loss_output['loss']
                        else:
                            raw_loss = loss_output
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        self.accelerator.backward(loss)
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if cfg.training.get('gradient_clip_norm', None) is not None:
                                self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.gradient_clip_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        if isinstance(loss_output, dict):
                            step_log['train_bc_loss'] = loss_output['bc_loss'].item()
                            step_log['train_aux_loss'] = loss_output['aux_loss'].item()
                            if 'log_std_mean' in loss_output:
                                step_log['train_log_std_mean'] = loss_output['log_std_mean'].item()
                                step_log['train_log_std_min'] = loss_output['log_std_min'].item()
                                step_log['train_log_std_max'] = loss_output['log_std_max'].item()

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            if self.accelerator.is_main_process:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                            next_checkpoint_step = self.last_checkpoint_step + cfg.training.checkpoint_every
                            if self.global_step >= next_checkpoint_step and self.accelerator.is_main_process:
                                print(f"Saving checkpoint at step {self.global_step} (target was {next_checkpoint_step})")
                                model_ddp = self.model
                                self.model = self.accelerator.unwrap_model(self.model)
                                if cfg.checkpoint.save_last_ckpt:
                                    self.save_checkpoint()
                                if cfg.checkpoint.save_last_snapshot:
                                    self.save_snapshot()

                                step_ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'step_{self.global_step:07d}.ckpt')
                                os.makedirs(os.path.dirname(step_ckpt_path), exist_ok=True)
                                self.save_checkpoint(path=step_ckpt_path)
                                self.model = model_ddp
                                self.model.train()

                                self.last_checkpoint_step = self.global_step

                            # Step-based validation and sampling
                            val_every_n_steps = cfg.training.get('val_every_n_steps', None)
                            if val_every_n_steps is not None and self.global_step % val_every_n_steps == 0:
                                policy = self.accelerator.unwrap_model(self.model)
                                policy.eval()
                                val_step_log = {}
                                with torch.no_grad():
                                    val_losses = list()
                                    val_bc_losses = list()
                                    val_aux_losses = list()
                                    val_log_std_means = list()
                                    val_log_std_mins = list()
                                    val_log_std_maxs = list()
                                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                                        val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                                        loss_output = policy.compute_loss(val_batch)
                                        if isinstance(loss_output, dict):
                                            val_losses.append(loss_output['loss'].item())
                                            if 'bc_loss' in loss_output:
                                                val_bc_losses.append(loss_output['bc_loss'].item())
                                            if 'aux_loss' in loss_output:
                                                val_aux_losses.append(loss_output['aux_loss'].item())
                                            if 'log_std_mean' in loss_output:
                                                val_log_std_means.append(loss_output['log_std_mean'].item())
                                                val_log_std_mins.append(loss_output['log_std_min'].item())
                                                val_log_std_maxs.append(loss_output['log_std_max'].item())
                                        else:
                                            val_losses.append(loss_output.item())
                                        if (cfg.training.max_val_steps is not None) \
                                            and val_batch_idx >= (cfg.training.max_val_steps-1):
                                            break
                                    if len(val_losses) > 0:
                                        val_step_log['val_loss'] = np.mean(val_losses)
                                    if len(val_bc_losses) > 0:
                                        val_step_log['val_bc_loss'] = np.mean(val_bc_losses)
                                    if len(val_aux_losses) > 0:
                                        val_step_log['val_aux_loss'] = np.mean(val_aux_losses)
                                    if len(val_log_std_means) > 0:
                                        val_step_log['val_log_std_mean'] = np.mean(val_log_std_means)
                                        val_step_log['val_log_std_min'] = np.mean(val_log_std_mins)
                                        val_step_log['val_log_std_max'] = np.mean(val_log_std_maxs)

                                    # Compute val_action_mse_error from val batches
                                    val_mse_list = list()
                                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                                        val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                                        obs_dict = val_batch['obs']
                                        gt_action = val_batch['action'][:, policy.n_obs_steps-1]
                                        result = policy.predict_action(obs_dict)
                                        pred_action = result['action']
                                        val_mse_list.append(torch.nn.functional.mse_loss(pred_action, gt_action).item())
                                        if (cfg.training.max_val_steps is not None) \
                                            and val_batch_idx >= (cfg.training.max_val_steps-1):
                                            break
                                    if len(val_mse_list) > 0:
                                        val_step_log['val_action_mse_error'] = np.mean(val_mse_list)

                                    # Also compute train_action_mse_error
                                    if train_sampling_batch is not None:
                                        s_batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                                        obs_dict = s_batch['obs']
                                        gt_action = s_batch['action'][:, policy.n_obs_steps-1]
                                        result = policy.predict_action(obs_dict)
                                        pred_action = result['action']
                                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                        val_step_log['train_action_mse_error'] = mse.item()

                                if self.accelerator.is_main_process and val_step_log:
                                    wandb_run.log(val_step_log, step=self.global_step)
                                    json_logger.log(val_step_log)
                                policy.train()

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                policy = self.accelerator.unwrap_model(self.model)
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_bc_losses = list()
                        val_aux_losses = list()
                        val_log_std_means = list()
                        val_log_std_mins = list()
                        val_log_std_maxs = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_output = self.accelerator.unwrap_model(self.model).compute_loss(batch)
                                if isinstance(loss_output, dict):
                                    val_losses.append(loss_output['loss'].item())
                                    if 'bc_loss' in loss_output:
                                        val_bc_losses.append(loss_output['bc_loss'].item())
                                    if 'aux_loss' in loss_output:
                                        val_aux_losses.append(loss_output['aux_loss'].item())
                                    if 'log_std_mean' in loss_output:
                                        val_log_std_means.append(loss_output['log_std_mean'].item())
                                        val_log_std_mins.append(loss_output['log_std_min'].item())
                                        val_log_std_maxs.append(loss_output['log_std_max'].item())
                                else:
                                    val_losses.append(loss_output.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = np.mean(val_losses)
                            step_log['val_loss'] = val_loss
                        if len(val_bc_losses) > 0:
                            step_log['val_bc_loss'] = np.mean(val_bc_losses)
                        if len(val_aux_losses) > 0:
                            step_log['val_aux_loss'] = np.mean(val_aux_losses)
                        if len(val_log_std_means) > 0:
                            step_log['val_log_std_mean'] = np.mean(val_log_std_means)
                            step_log['val_log_std_min'] = np.mean(val_log_std_mins)
                            step_log['val_log_std_max'] = np.mean(val_log_std_maxs)

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action'][:, policy.n_obs_steps-1]
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                policy.train()

                if self.accelerator.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        self.accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainMLPLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
