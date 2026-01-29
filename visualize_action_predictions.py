"""
Visualize action predictions of trained model vs ground truth actions from dataset.

Usage:
python visualize_action_predictions.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --dataset /path/to/dataset.zarr \
    --episode 0 \
    --output ./visualizations

This script loads a trained diffusion policy model and compares its action predictions
with ground truth actions from a zarr dataset, providing comprehensive visualizations
and analysis.
"""

import os
import sys
import time
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import dill
import hydra
import pathlib
from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_ours
from diffusion_policy.common.pytorch_util import dict_apply

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[BaseImagePolicy, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        policy: Loaded policy model
        cfg: Model configuration
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # Create workspace and load model
    cls = hydra.utils.get_class(cfg._target_)
    cfg['policy']['obs_encoder']['extra_randomizations'] = []
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy: BaseImagePolicy = getattr(workspace, 'model')
    if cfg.training.use_ema:
        policy = getattr(workspace, 'ema_model')
    
    policy.eval().to(device)
    
    # Set inference parameters
    if hasattr(policy, 'num_inference_steps'):
        setattr(policy, 'num_inference_steps', 16)  # DDIM inference iterations
    
    # Get n_action_steps - different policies store this differently
    n_action_steps = getattr(policy, 'n_action_steps', cfg.get('n_action_steps', 1))
    n_obs_steps = getattr(policy, 'n_obs_steps', cfg.get('n_obs_steps', 2))
    
    # Calculate horizon if available
    horizon = getattr(policy, 'horizon', None)
    if horizon is None and hasattr(policy, 'n_action_steps') and hasattr(policy, 'n_obs_steps'):
        policy_n_action_steps = getattr(policy, 'n_action_steps', 1)
        policy_n_obs_steps = getattr(policy, 'n_obs_steps', 2)
        horizon = policy_n_action_steps + policy_n_obs_steps - 1
    
    print(f"Model loaded successfully")
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"n_action_steps: {n_action_steps}")
    if horizon is not None:
        print(f"horizon: {horizon}")
    else:
        print("horizon: Not available (MLP policy)")
    
    return policy, cfg


def load_zarr_data(zarr_path: str, episode_idx: int = 0) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load observations and actions from zarr dataset for specific episode.
    
    Args:
        zarr_path: Path to zarr dataset
        episode_idx: Episode index to load
        
    Returns:
        observations: Dictionary of observations
        actions: Ground truth actions
        episode_info: Episode metadata
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Dataset not found: {zarr_path}")
    
    print(f"Loading zarr dataset from {zarr_path}")
    
    # Open zarr dataset
    root = zarr.open(zarr_path, mode='r')
    
    # Validate structure
    if 'data' not in root:
        raise ValueError("Dataset does not contain 'data' group")
    if 'meta' not in root or 'episode_ends' not in root['meta']:
        raise ValueError("Dataset does not contain episode metadata")
    
    data_group = root['data']
    
    # Get episode boundaries
    episode_ends = np.array(root['meta']['episode_ends'][:])
    if episode_idx >= len(episode_ends):
        raise ValueError(f"Episode index {episode_idx} out of range. Dataset has {len(episode_ends)} episodes.")
    
    # Calculate episode boundaries
    if episode_idx == 0:
        start_idx = 0
    else:
        start_idx = int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    episode_length = end_idx - start_idx
    
    print(f"Loading episode {episode_idx}: timesteps {start_idx} to {end_idx} (length: {episode_length})")
    
    # Load actions
    if 'actions' in data_group:
        actions = np.array(data_group['actions'][start_idx:end_idx])
    elif 'action' in data_group:
        actions = np.array(data_group['action'][start_idx:end_idx])
    else:
        raise ValueError("Dataset does not contain 'actions' or 'action' array")
    
    # Load observations
    observations = {}
    obs_group = data_group['obs']
    
    # Get observation keys - handle zarr group properly
    obs_keys = []
    try:
        # Try to get keys from zarr group
        if hasattr(obs_group, 'keys'):
            obs_keys = list(obs_group.keys())  # type: ignore
        else:
            # Fallback: iterate through the group
            obs_keys = [str(k) for k in obs_group]
    except Exception:
        # Last resort: use common observation keys
        obs_keys = ['front_rgb', 'side_rgb', 'wrist_rgb', 'arm_joint_pos', 'end_effector_pose']
        obs_keys = [k for k in obs_keys if k in obs_group]
    
    for key in obs_keys:
        obs_data = obs_group[key]
        if len(obs_data.shape) == 4:  # Image data (T, H, W, C)
            observations[key] = np.array(obs_data[start_idx:end_idx])
        else:  # Low-dim data
            observations[key] = np.array(obs_data[start_idx:end_idx])
    
    episode_info = {
        'episode_idx': episode_idx,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'length': episode_length,
        'action_shape': actions.shape,
        'obs_keys': list(observations.keys())
    }
    
    print(f"Loaded {len(actions)} actions with shape {actions.shape}")
    print(f"Observation keys: {episode_info['obs_keys']}")
    
    return observations, actions, episode_info


def process_observations(observations: Dict[str, np.ndarray], 
                        shape_meta: dict, 
                        n_obs_steps: int,
                        device: str = 'cuda') -> List[Dict[str, torch.Tensor]]:
    """
    Process observations for model inference.
    
    Args:
        observations: Raw observations from dataset
        shape_meta: Shape metadata from model config
        n_obs_steps: Number of observation steps for model
        device: Device to put tensors on
        
    Returns:
        processed_obs_list: List of processed observation dictionaries for each timestep
    """
    processed_obs_list = []
    episode_length = len(list(observations.values())[0])
    
    for t in range(episode_length - n_obs_steps + 1):
        # Extract observation window
        obs_window = {}
        for key, obs_data in observations.items():
            obs_window[key] = obs_data[t:t + n_obs_steps]
        
        # Process using real inference utility
        obs_dict_np = get_real_obs_ours(
            env_obs=obs_window, 
            shape_meta=shape_meta
        )
        
        # Convert to tensors
        obs_dict = {}
        for key, value in obs_dict_np.items():
            obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        
        processed_obs_list.append(obs_dict)
    
    return processed_obs_list


def predict_actions(policy: BaseImagePolicy, 
                   processed_obs_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
    """
    Predict actions using the loaded policy.
    
    Args:
        policy: Loaded policy model
        processed_obs_list: List of processed observations
        
    Returns:
        predicted_actions: Array of predicted actions
    """
    predicted_actions = []
    
    print("Running model inference...")
    if hasattr(policy, 'reset'):
        policy.reset()
    
    with torch.no_grad():
        for i, obs_dict in enumerate(processed_obs_list):
            try:
                result = policy.predict_action(obs_dict)
                action_tensor = result['action'][0]  # Remove batch dimension
                
                # Handle different action tensor shapes
                if len(action_tensor.shape) == 2:
                    # Multi-step prediction (e.g., diffusion policy)
                    action = action_tensor[0].detach().to('cpu').numpy()  # Take first action
                elif len(action_tensor.shape) == 1:
                    # Single-step prediction (e.g., MLP policy)
                    action = action_tensor.detach().to('cpu').numpy()
                else:
                    raise ValueError(f"Unexpected action tensor shape: {action_tensor.shape}")
                
                predicted_actions.append(action)
                
                if i % 10 == 0:
                    print(f"Processed {i+1}/{len(processed_obs_list)} timesteps")
                    
            except Exception as e:
                print(f"Error at timestep {i}: {e}")
                # Use previous action or zeros as fallback
                if len(predicted_actions) > 0:
                    predicted_actions.append(predicted_actions[-1])
                else:
                    # Try to infer action dimension from policy
                    action_dim = getattr(policy, 'action_dim', 6)
                    predicted_actions.append(np.zeros(action_dim))
    
    predicted_actions = np.array(predicted_actions)
    print(f"Generated {len(predicted_actions)} action predictions")
    
    return predicted_actions


def visualize_comparison(gt_actions: np.ndarray, 
                        pred_actions: np.ndarray,
                        episode_info: dict,
                        output_dir: str,
                        observations: Optional[Dict[str, np.ndarray]] = None):
    """
    Create comprehensive visualizations comparing predicted vs ground truth actions.
    
    Args:
        gt_actions: Ground truth actions
        pred_actions: Predicted actions
        episode_info: Episode metadata
        output_dir: Directory to save visualizations
        observations: Optional observations for context
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Align arrays (pred_actions might be shorter due to observation window)
    min_len = min(len(gt_actions), len(pred_actions))
    gt_actions = gt_actions[:min_len]
    pred_actions = pred_actions[:min_len]
    
    action_dim = gt_actions.shape[1]
    timesteps = np.arange(min_len)
    
    # Action dimension names (assuming 6DOF pose)
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    if action_dim != 6:
        action_names = [f'Action_{i}' for i in range(action_dim)]
    
    # 1. Time series comparison
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]
    
    for i in range(action_dim):
        axes[i].plot(timesteps, gt_actions[:, i], label='Ground Truth', alpha=0.8, linewidth=2)
        axes[i].plot(timesteps, pred_actions[:, i], label='Predicted', alpha=0.8, linewidth=2)
        axes[i].set_ylabel(action_names[i])
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate and show MSE
        mse = np.mean((gt_actions[:, i] - pred_actions[:, i])**2)
        axes[i].set_title(f'{action_names[i]} - MSE: {mse:.6f}')
    
    axes[-1].set_xlabel('Timestep')
    plt.suptitle(f'Action Predictions vs Ground Truth - Episode {episode_info["episode_idx"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'action_comparison_episode_{episode_info["episode_idx"]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error analysis
    errors = pred_actions - gt_actions
    
    fig, axes = plt.subplots(2, action_dim, figsize=(4*action_dim, 8))
    if action_dim == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(action_dim):
        # Error over time
        axes[0, i].plot(timesteps, errors[:, i], alpha=0.7)
        axes[0, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, i].set_title(f'{action_names[i]} Error over Time')
        axes[0, i].set_ylabel('Error')
        axes[0, i].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, i].hist(errors[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes[1, i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, i].set_title(f'{action_names[i]} Error Distribution')
        axes[1, i].set_xlabel('Error')
        axes[1, i].set_ylabel('Frequency')
        
        # Add statistics
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])
        axes[1, i].text(0.02, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                       transform=axes[1, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0, -1].set_xlabel('Timestep')
    plt.suptitle(f'Prediction Error Analysis - Episode {episode_info["episode_idx"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_analysis_episode_{episode_info["episode_idx"]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Action correlation plot
    fig, axes = plt.subplots(1, action_dim, figsize=(4*action_dim, 4))
    if action_dim == 1:
        axes = [axes]
    
    for i in range(action_dim):
        axes[i].scatter(gt_actions[:, i], pred_actions[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(gt_actions[:, i].min(), pred_actions[:, i].min())
        max_val = max(gt_actions[:, i].max(), pred_actions[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[i].set_xlabel(f'Ground Truth {action_names[i]}')
        axes[i].set_ylabel(f'Predicted {action_names[i]}')
        axes[i].set_title(f'{action_names[i]} Correlation')
        axes[i].grid(True, alpha=0.3)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(gt_actions[:, i], pred_actions[:, i])[0, 1]
        axes[i].text(0.02, 0.95, f'Corr: {corr:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Action Correlation Analysis - Episode {episode_info["episode_idx"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_analysis_episode_{episode_info["episode_idx"]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary statistics
    metrics = {}
    for i in range(action_dim):
        mse = np.mean((gt_actions[:, i] - pred_actions[:, i])**2)
        mae = np.mean(np.abs(gt_actions[:, i] - pred_actions[:, i]))
        corr = np.corrcoef(gt_actions[:, i], pred_actions[:, i])[0, 1]
        
        metrics[action_names[i]] = {
            'MSE': mse,
            'MAE': mae,
            'Correlation': corr,
            'RMSE': np.sqrt(mse)
        }
    
    # Save metrics to file
    with open(os.path.join(output_dir, f'metrics_episode_{episode_info["episode_idx"]}.txt'), 'w') as f:
        f.write(f"Action Prediction Metrics - Episode {episode_info['episode_idx']}\n")
        f.write("="*60 + "\n")
        f.write(f"Episode length: {episode_info['length']}\n")
        f.write(f"Predicted timesteps: {min_len}\n")
        f.write(f"Action dimensions: {action_dim}\n\n")
        
        for action_name, action_metrics in metrics.items():
            f.write(f"{action_name}:\n")
            for metric_name, value in action_metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")
    
    # 5. Sample observation context (if available)
    if observations is not None and any('rgb' in key for key in observations.keys()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show sample frames at different timesteps
        sample_timesteps = [0, min_len//3, min_len*2//3] if min_len > 3 else [0]
        rgb_keys = [key for key in observations.keys() if 'rgb' in key][:3]  # Max 3 cameras
        
        for i, timestep in enumerate(sample_timesteps):
            for j, rgb_key in enumerate(rgb_keys):
                if timestep < len(observations[rgb_key]):
                    img = observations[rgb_key][timestep]
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    axes[0, j].imshow(img)
                    axes[0, j].set_title(f'{rgb_key} - t={timestep}')
                    axes[0, j].axis('off')
                    
                    # Show corresponding action
                    if timestep < min_len:
                        axes[1, j].bar(range(action_dim), gt_actions[timestep], alpha=0.7, label='GT')
                        axes[1, j].bar(range(action_dim), pred_actions[timestep], alpha=0.7, label='Pred')
                        axes[1, j].set_title(f'Actions at t={timestep}')
                        axes[1, j].set_xticks(range(action_dim))
                        axes[1, j].set_xticklabels(action_names, rotation=45)
                        axes[1, j].legend()
        
        plt.suptitle(f'Observation Context and Actions - Episode {episode_info["episode_idx"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'context_episode_{episode_info["episode_idx"]}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    print("\nSummary Metrics:")
    for action_name, action_metrics in metrics.items():
        print(f"{action_name}: MSE={action_metrics['MSE']:.6f}, "
              f"MAE={action_metrics['MAE']:.6f}, "
              f"Corr={action_metrics['Correlation']:.3f}")


@click.command()
@click.option('--checkpoint', '-c', required=True, 
              help='Path to model checkpoint')
@click.option('--dataset', '-d', required=True,
              help='Path to zarr dataset')
@click.option('--episode', '-e', default=0, type=int,
              help='Episode index to analyze (default: 0)')
@click.option('--output', '-o', default='./visualizations',
              help='Output directory for visualizations')
@click.option('--device', default='cuda',
              help='Device to run inference on (cuda/cpu)')
@click.option('--episodes', '-eps', default=None, type=str,
              help='Comma-separated list of episodes to analyze (e.g., "0,1,2")')
def main(checkpoint, dataset, episode, output, device, episodes):
    """
    Visualize action predictions of trained model vs ground truth actions.
    """
    print("="*60)
    print("Action Prediction Visualization Tool")
    print("="*60)
    
    # Determine episodes to process
    if episodes is not None:
        episode_list = [int(x.strip()) for x in episodes.split(',')]
    else:
        episode_list = [episode]
    
    print(f"Processing episodes: {episode_list}")
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    try:
        # Load model
        policy, cfg = load_model(checkpoint, device)
        
        # Process each episode
        for ep_idx in episode_list:
            print(f"\n{'='*40}")
            print(f"Processing Episode {ep_idx}")
            print(f"{'='*40}")
            
            try:
                # Load data
                observations, gt_actions, episode_info = load_zarr_data(dataset, ep_idx)
                
                # Process observations
                processed_obs_list = process_observations(
                    observations, cfg['shape_meta'], cfg.get('n_obs_steps', 2), device)
                
                # Predict actions
                pred_actions = predict_actions(policy, processed_obs_list)
                
                # Create episode-specific output directory
                episode_output_dir = os.path.join(output, f'episode_{ep_idx}')
                
                # Visualize comparison
                visualize_comparison(
                    gt_actions, pred_actions, episode_info, 
                    episode_output_dir, observations)
                
                print(f"Episode {ep_idx} completed successfully!")
                
            except Exception as e:
                print(f"Error processing episode {ep_idx}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("Analysis completed!")
        print(f"Results saved to: {output}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
