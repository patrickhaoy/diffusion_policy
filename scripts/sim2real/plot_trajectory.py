#!/usr/bin/env python3
"""
Plot arm joint positions and actions from a trajectory pickle file.
Shows 6 individual plots, one for each joint.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trajectory_data(pickle_path):
    """Load trajectory data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_joint_trajectories(data, save_path=None):
    """Plot joint positions and actions in separate subplots."""
    
    # Extract joint data
    if 'observations' in data and 'actions' in data:
        # Format: observations and actions as separate keys
        obs_data = data['observations']
        action_data = data['actions']
        
        # Extract arm joint positions from observations
        if isinstance(obs_data, list) and len(obs_data) > 0:
            # Check if observations are nested under 'policy' key
            if 'policy' in obs_data[0]:
                policy_obs = [obs['policy'] for obs in obs_data]
                if 'arm_joint_pos' in policy_obs[0]:
                    joint_pos = np.array([obs['arm_joint_pos'] for obs in policy_obs])
                else:
                    print("Available policy observation keys:", policy_obs[0].keys() if isinstance(policy_obs[0], dict) else "Not a dict")
                    return
            elif 'arm_joint_pos' in obs_data[0]:
                joint_pos = np.array([obs['arm_joint_pos'] for obs in obs_data])
            else:
                print("Available observation keys:", obs_data[0].keys() if isinstance(obs_data[0], dict) else "Not a dict")
                return
        else:
            print("Unexpected observations format")
            return
            
        # Extract actions
        if isinstance(action_data, list):
            actions = np.array(action_data)
        else:
            actions = action_data
            
    elif 'obs' in data and 'action' in data:
        # Alternative format: obs and action as keys
        obs_data = data['obs']
        action_data = data['action']
        
        if 'arm_joint_pos' in obs_data:
            joint_pos = obs_data['arm_joint_pos']
        else:
            print("Available obs keys:", obs_data.keys() if isinstance(obs_data, dict) else "Not a dict")
            return
            
        actions = action_data
        
    else:
        print("Available data keys:", data.keys())
        print("Expected format: either {'observations': [...], 'actions': [...]} or {'obs': {...}, 'action': [...]}")
        return
    
    print(f"Joint positions shape: {joint_pos.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Determine number of joints
    num_joints = min(joint_pos.shape[1], 6)  # Limit to 6 joints for arm
    if actions.shape[1] > num_joints:
        # Actions might include gripper, so take first 6 for joints
        joint_actions = actions[:, :num_joints]
    else:
        joint_actions = actions
        
    print(f"Plotting {num_joints} joints")
    print(f"Trajectory length: {len(joint_pos)} timesteps")
    
    # Create figure with 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    # Colors for different data types
    colors = plt.colormaps['tab10'](np.linspace(0, 1, 10))
    
    for j in range(num_joints):
        ax = axes_flat[j]
        
        # Plot joint positions (observations)
        timesteps = np.arange(len(joint_pos))
        ax.plot(timesteps, joint_pos[:, j], color='blue', linewidth=2, label='Joint Position (obs)', alpha=0.8)
        
        # Plot joint actions (commands)
        action_timesteps = np.arange(len(joint_actions))
        ax.plot(action_timesteps, joint_actions[:, j], color='red', linewidth=2, linestyle='--', label='Joint Action (cmd)', alpha=0.8)
        
        # Calculate error between action and next observation
        if len(joint_actions) == len(joint_pos):
            # Same length - compare directly
            error = np.abs(joint_pos[:, j] - joint_actions[:, j])
            rmse_error = np.sqrt(np.mean(error ** 2))
        elif len(joint_actions) == len(joint_pos) - 1:
            # Action at t leads to observation at t+1
            error = np.abs(joint_pos[1:, j] - joint_actions[:, j])
            rmse_error = np.sqrt(np.mean(error ** 2))
        else:
            rmse_error = 0.0
            
        ax.set_title(f'Joint {j} (RMSE: {rmse_error:.4f})', fontsize=12)
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Set consistent y-axis limits
        all_values = np.concatenate([joint_pos[:, j], joint_actions[:, j]])
        y_min = all_values.min() - 0.1
        y_max = all_values.max() + 0.1
        ax.set_ylim(y_min, y_max)
    
    plt.suptitle(f'Joint Trajectories: Positions vs Actions', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot arm joint trajectories from pickle file")
    parser.add_argument("pickle_path", type=str, help="Path to trajectory pickle file")
    parser.add_argument("--save", type=str, help="Path to save plot image", default=None)
    args = parser.parse_args()
    
    pickle_path = Path(args.pickle_path)
    if not pickle_path.exists():
        print(f"Error: File not found: {pickle_path}")
        return
    
    print(f"Loading trajectory data from: {pickle_path}")
    
    try:
        data = load_trajectory_data(pickle_path)
        print("Data loaded successfully!")
        print(f"Top-level keys: {list(data.keys())}")
        
        plot_joint_trajectories(data, args.save)
        
    except Exception as e:
        print(f"Error loading or plotting data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 