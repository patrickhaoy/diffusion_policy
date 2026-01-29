"""
Script to replay saved trajectories on real robot.

Usage:
python replay_real_robot.py -i <trajectory_file_or_dir> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start replay (hand control over to policy).
Press "Q" to exit program.

================ Replay in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop replay and gain control back.
"""

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import json
import pathlib
import math
import os
import pickle
import glob
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import scipy.spatial.transform as st
import zarr  # Add zarr import for loading zarr datasets
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution_ours, 
    get_real_obs_dict, 
    get_real_obs_ours
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.cv2_util import get_image_transform

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_trajectory(filepath):
    """Load trajectory data from file."""
    with open(filepath, 'rb') as f:
        trajectory_data = pickle.load(f)
    return trajectory_data

def load_zarr_dataset(zarr_path, episode_idx=0):
    """Load trajectory data from zarr dataset.
    
    Args:
        zarr_path: Path to the zarr dataset
        episode_idx: Episode index to load (default: 0)
        
    Returns:
        dict: Dictionary containing trajectory data with format similar to pickle
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr dataset path does not exist: {zarr_path}")
    
    # Open zarr dataset
    root = zarr.open(zarr_path, mode='r')
    
    # Validate dataset structure
    if 'data' not in root:
        raise ValueError("Dataset does not contain 'data' group")
    
    data_group = root['data']
    
    # Get episode boundaries if available
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends']
        episode_ends_array = np.array(episode_ends[:])
        if episode_idx >= len(episode_ends_array):
            raise ValueError(f"Episode index {episode_idx} out of range. Dataset has {len(episode_ends_array)} episodes.")
        
        # Calculate episode boundaries  
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = int(episode_ends_array[episode_idx - 1])
        end_idx = int(episode_ends_array[episode_idx])
        
        episode_length = end_idx - start_idx
        print(f"Loading zarr episode {episode_idx}: timesteps {start_idx} to {end_idx} (length: {episode_length})")
    else:
        # Fallback: assume single episode
        start_idx = 0
        if 'action' in data_group:
            end_idx = data_group['action'].shape[0]
        elif 'actions' in data_group:
            end_idx = data_group['actions'].shape[0]
        else:
            raise ValueError("Dataset does not contain 'action' or 'actions' array")
        episode_length = end_idx
        print(f"No episode metadata found. Loading full zarr dataset: {episode_length} timesteps")
    
    # Load actions - try both 'action' and 'actions' keys
    actions = None
    if 'action' in data_group:
        actions = np.array(data_group['action'][start_idx:end_idx])
    elif 'actions' in data_group:
        actions = np.array(data_group['actions'][start_idx:end_idx])
    else:
        raise ValueError("Dataset does not contain 'action' or 'actions' array")
    
    # Get initial joint positions from arm_joint_pos[0] if available
    initial_joints = None
    if 'arm_joint_pos' in data_group:
        initial_joints = np.array(data_group['arm_joint_pos'][start_idx])
        print(f"Found initial joint positions from zarr: {initial_joints}")
    
    # Create trajectory data in format expected by existing code
    trajectory_data = {
        'actions': actions,
        'episode_length': episode_length,
        'trajectory_id': f'zarr_episode_{episode_idx}',
        'success': True,  # Assume success for zarr data
        'initial_state': None  # Will be filled below if initial_joints available
    }
    
    # Add initial state if joint positions are available
    if initial_joints is not None:
        trajectory_data['initial_state'] = {
            'articulation': {
                'ur5': {  # Use ur5 as robot name for consistency
                    'joint_position': torch.tensor(initial_joints)
                }
            }
        }
    
    return trajectory_data

def is_zarr_dataset(path):
    """Check if path points to a zarr dataset."""
    if os.path.isdir(path):
        # Check if it's a zarr directory
        zarr_file = os.path.join(path, '.zarray') or os.path.join(path, '.zgroup')
        if os.path.exists(zarr_file):
            return True
        # Also check for replay_buffer.zarr specifically
        replay_buffer_zarr = os.path.join(path, 'replay_buffer.zarr')
        if os.path.exists(replay_buffer_zarr):
            return True
    elif path.endswith('.zarr'):
        return os.path.exists(path)
    return False

def load_metadata(trajectory_dir):
    """Load metadata from trajectory directory."""
    metadata_path = os.path.join(trajectory_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    return None

def get_trajectory_files(trajectory_path):
    """Get list of trajectory files to replay."""
    if os.path.isfile(trajectory_path):
        return [trajectory_path]
    elif os.path.isdir(trajectory_path):
        # Check if it's a zarr dataset directory
        if is_zarr_dataset(trajectory_path):
            return [trajectory_path]  # Return the zarr directory itself
        
        # Check for replay_buffer.zarr inside directory
        replay_buffer_zarr = os.path.join(trajectory_path, 'replay_buffer.zarr')
        if os.path.exists(replay_buffer_zarr):
            return [replay_buffer_zarr]
        
        # Get all trajectory pickle files in directory
        pattern = os.path.join(trajectory_path, "trajectory_*.pkl")
        files = sorted(glob.glob(pattern))
        return files
    else:
        raise ValueError(f"Path {trajectory_path} is neither a file nor a directory")

def get_initial_robot_pose(trajectory):
    """Extract initial robot pose from trajectory initial state.
    
    Args:
        trajectory: Trajectory data containing initial_state
        
    Returns:
        dict with joint positions if available, None otherwise
    """
    initial_state = trajectory.get('initial_state')
    if initial_state is None:
        return None
    
    # Look for robot articulation in initial state
    articulations = initial_state.get('articulation', {})
    
    # Try to find the robot (common names)
    robot_names = ['robot', 'arm', 'manipulator', 'franka', 'ur5', 'ur10']
    robot_state = None
    robot_name = 'unknown'  # Initialize robot_name
    
    for name in robot_names:
        if name in articulations:
            robot_state = articulations[name]
            robot_name = name
            break
    
    # If no common name found, take the first articulation
    if robot_state is None and articulations:
        robot_name = list(articulations.keys())[0]
        robot_state = articulations[robot_name]
        print(f"Using first articulation '{robot_name}' as robot")
    
    if robot_state is None:
        return None
    
    # Extract joint positions
    joint_positions = robot_state.get('joint_position')
    if joint_positions is not None:
        if isinstance(joint_positions, torch.Tensor):
            # Take first environment if batched
            if joint_positions.dim() > 1:
                joint_positions = joint_positions[0]
            joint_positions = joint_positions.cpu().numpy()
        
        return {
            'joint_positions': joint_positions,
            'root_pose': robot_state.get('root_pose'),
            'robot_name': robot_name
        }
    
    return None

def get_camera_obs_key(vis_camera_idx):
    """Get the correct camera observation key based on index."""
    camera_keys = ['front_rgb', 'wrist_rgb', 'side_rgb']
    if vis_camera_idx < len(camera_keys):
        return camera_keys[vis_camera_idx]
    else:
        # Fallback to first camera if index is out of range
        print(f"Warning: Camera index {vis_camera_idx} out of range, using front camera")
        return camera_keys[0]

def convert_sim_to_real_action(sim_action, action_mapping='direct'):
    """Convert simulation action to real robot action.
    
    Args:
        sim_action: Action from simulation (typically 6D pose)
        action_mapping: How to map actions ('direct', 'scaled', etc.)
        
    Returns:
        Real robot action
    """
    if isinstance(sim_action, torch.Tensor):
        sim_action = sim_action.cpu().numpy()
    
    # For now, use direct mapping
    # In practice, you might need coordinate transformations or scaling
    real_action = sim_action.copy()
    
    # Potential transformations:
    # - Coordinate frame conversions (sim vs real)
    # - Action scaling (sim units vs real units)
    # - Safety clamping
    
    return real_action

def plot_joint_comparison(achieved_joints, target_joints, save_path=None):
    """Plot achieved vs target joint positions in 6 separate subplots."""
    
    if not achieved_joints or not target_joints:
        print("No joint data collected for plotting")
        return
    
    achieved_joints = np.array(achieved_joints)
    target_joints = np.array(target_joints)
    
    print(f"Plotting joint data:")
    print(f"  Achieved joints shape: {achieved_joints.shape}")
    print(f"  Target joints shape: {target_joints.shape}")
    
    # Ensure we have at least 6 joints to plot
    num_joints = min(achieved_joints.shape[1], target_joints.shape[1], 6)
    
    # Create figure with 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for j in range(num_joints):
        ax = axes_flat[j]
        
        # Plot trajectories
        timesteps = np.arange(len(achieved_joints))
        ax.plot(timesteps, achieved_joints[:, j], color='blue', linewidth=2, label='Achieved', alpha=0.8)
        ax.plot(timesteps, target_joints[:, j], color='red', linewidth=2, linestyle='--', label='Target', alpha=0.8)
        
        # Calculate error metrics
        error = np.abs(achieved_joints[:, j] - target_joints[:, j])
        rmse_error = np.sqrt(np.mean(error ** 2))
        max_error = np.max(error)
        
        ax.set_title(f'Joint {j} (RMSE: {rmse_error:.4f}, Max: {max_error:.4f})', fontsize=12)
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Set consistent y-axis limits
        all_values = np.concatenate([achieved_joints[:, j], target_joints[:, j]])
        y_min = all_values.min() - 0.1
        y_max = all_values.max() + 0.1
        ax.set_ylim(y_min, y_max)
    
    plt.suptitle(f'Real Robot: Achieved vs Target Joint Positions', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Joint comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nJoint Tracking Performance Summary:")
    print("=" * 50)
    for j in range(num_joints):
        error = np.abs(achieved_joints[:, j] - target_joints[:, j])
        rmse = np.sqrt(np.mean(error ** 2))
        max_err = np.max(error)
        mean_err = np.mean(error)
        print(f"Joint {j}: RMSE={rmse:.4f}, Max={max_err:.4f}, Mean={mean_err:.4f}")
    
    total_rmse = np.sqrt(np.mean((achieved_joints[:, :num_joints] - target_joints[:, :num_joints]) ** 2))
    print(f"Overall RMSE: {total_rmse:.4f}")

@click.command()
@click.option('--input', '-i', required=True, help='Path to trajectory file, directory, or zarr dataset')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="Robot's IP address e.g. 192.168.0.204")
@click.option('--trajectory_idx', '-ti', default=0, type=int, help='Which trajectory to replay (for pickle files in directory)')
@click.option('--episode_idx', '-ei', default=0, type=int, help='Which episode to replay (for zarr datasets)')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize (0=front, 1=wrist, 2=side).")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration from trajectory.")
@click.option('--steps_per_inference', '-si', default=1, type=int, help="Steps per action execution (use 1 for direct replay).")
@click.option('--max_duration', '-md', default=60, help='Max duration for each replay in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between command and execution in Sec.")
@click.option('--action_mapping', default='direct', help="How to map simulation actions to real robot ('direct', 'scaled').")
@click.option('--preview_mode', is_flag=True, default=False, help="Preview trajectory without executing on robot.")
@click.option('--plot_joints', is_flag=True, default=True, help="Plot achieved vs target joint positions after replay.")
def main(input, output, robot_ip, trajectory_idx, episode_idx, vis_camera_idx, init_joints, 
         steps_per_inference, max_duration, frequency, command_latency,
         action_mapping, preview_mode, plot_joints):
    
    # Load trajectory files
    trajectory_files = get_trajectory_files(input)
    if not trajectory_files:
        print(f"No trajectory files found in {input}")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Handle zarr datasets differently
    if len(trajectory_files) == 1 and is_zarr_dataset(trajectory_files[0]):
        # Single zarr dataset
        trajectory_file = trajectory_files[0]
        print(f"Loading zarr dataset: {os.path.basename(trajectory_file)}")
        trajectory_file += f"/replay_buffer.zarr"  # Ensure we use the correct zarr file
        trajectory = load_zarr_dataset(trajectory_file, episode_idx)
        print(f"Using episode index: {episode_idx}")
    else:
        # Pickle files
        # Select trajectory to replay
        if trajectory_idx >= len(trajectory_files):
            print(f"Trajectory index {trajectory_idx} out of range. Available: 0-{len(trajectory_files)-1}")
            return
        
        trajectory_file = trajectory_files[trajectory_idx]
        print(f"Loading trajectory: {os.path.basename(trajectory_file)}")
        
        # Load trajectory data
        trajectory = load_trajectory(trajectory_file)
    
    print(f"Trajectory info:")
    print(f"  ID: {trajectory.get('trajectory_id', 'Unknown')}")
    print(f"  Length: {trajectory.get('episode_length', len(trajectory.get('actions', [])))}")
    print(f"  Success: {trajectory.get('success', 'Unknown')}")
    print(f"  Has initial state: {trajectory.get('initial_state') is not None}")
    
    # Load metadata if available
    if os.path.isdir(input):
        metadata = load_metadata(input)
        if metadata:
            print(f"Dataset metadata:")
            print(f"  Task: {metadata.get('task', 'Unknown')}")
            print(f"  Checkpoint: {metadata.get('checkpoint', 'Unknown')}")
            print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
    
    # Extract actions and validate
    actions = trajectory.get('actions', [])

    
    print(f"Loaded {len(actions)} actions for replay")
    
    # Get initial robot pose if available
    initial_robot_pose = None
    if init_joints:
        initial_robot_pose = get_initial_robot_pose(trajectory)
        if initial_robot_pose:
            print(f"Initial robot pose available: {initial_robot_pose['robot_name']}")
            print(f"Joint positions: {initial_robot_pose['joint_positions']}")
        else:
            print("Warning: No initial robot pose found in trajectory")
    
    # Setup device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Load RealSense camera configurations
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "415_wrist.json"))
    ]
    
    # Setup timing
    dt = 1/frequency
    
    # For real robot, we need to determine observation resolution from the first action
    # This is a simplification - in practice you'd get this from the policy config
    first_action = actions[0]
    if isinstance(first_action, torch.Tensor):
        action_dim = first_action.shape[-1]
    else:
        action_dim = len(first_action)
    
    print(f"Action dimension: {action_dim}")
    
    # Dummy observation resolution for real robot setup
    # In practice, this should match your training setup
    obs_res = (224, 224)  # Common resolution
    n_obs_steps = 1  # Common value, should match training
    
    if preview_mode:
        print("\n=== PREVIEW MODE ===")
        print("Showing trajectory actions without robot execution")
        for i, action in enumerate(actions[:10]):  # Show first 10 actions
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            print(f"Action {i}: {action}")
        print(f"... and {len(actions)-10} more actions")
        return
    
    print(f"\n=== STARTING REAL ROBOT REPLAY ===")
    print(f"Robot IP: {robot_ip}")
    print(f"Frequency: {frequency} Hz")
    print(f"Action mapping: {action_mapping}")
    
    # Prepare custom initial joints for RealEnv if available
    custom_joints = None
    if init_joints and initial_robot_pose:
        custom_joints = initial_robot_pose['joint_positions'][:6]
        print(f"Will initialize robot with custom joints: {custom_joints}")
    
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            custom_init_joints=custom_joints,  # Pass custom joints to RealEnv
            enable_multi_cam_vis=True,
            record_raw_video=True,
            rolling_action_buffer=True,
            camera_serial_numbers=['215122255213', '832112070487',
                                  '746112060198'],
            camera_configs=configs,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            
            cv2.setNumThreads(1)

            # Camera setup (should match training)
            env.realsense.set_exposure(exposure=500, gain=0)
            env.realsense.set_white_balance(white_balance=2000)

            print("Waiting for realsense")
            time.sleep(1.0)

            # Robot initialization is now handled by RealEnv constructor
            # Get current robot state for reference
            robot_state = env.get_robot_state()
            initial_tcp_pose = robot_state['TargetTCPPose']
            print(f"Current robot TCP pose: {initial_tcp_pose}")
            
            if init_joints and initial_robot_pose:
                print(f"Robot initialized to trajectory starting pose")
            
            time.sleep(5.0)

            print('Ready for trajectory replay!')
            print('Press "C" in the camera window to start replay')
            
            # Get the correct camera key for visualization
            camera_key = get_camera_obs_key(vis_camera_idx)
            print(f"Using camera: {camera_key}")
            
            achieved_joints = []
            target_joints = []

            while True:
                # ========== Human control phase ==============
                obs = env.get_obs()
                
                # Show camera view
                vis_img = obs[camera_key][-1]
                text = 'Press "C" to start replay, "Q" to quit'
                cv2.putText(
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2,
                    color=(0, 255, 0)
                )
                cv2.imshow('Trajectory Replay', vis_img[...,::-1])
                
                key_stroke = cv2.waitKey(1) & 0xFF
                if key_stroke == ord('q'):
                    print("Exiting...")
                    break
                elif key_stroke == ord('c'):
                    print("Starting trajectory replay...")
                    
                    # ========== Trajectory replay phase ==============
                    try:
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start)
                        
                        # Wait for precise timing
                        frame_latency = 1/30
                        precise_wait(eval_t_start - frame_latency, time_func=time.time)
                        print("Replay started!")
                        
                        # Execute trajectory actions
                        for action_idx, sim_action in enumerate(actions):
                            # Calculate timing for this action
                            t_cycle_end = t_start + (action_idx + steps_per_inference) * dt
                            
                            # Get current observations
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            print(f'Action {action_idx}/{len(actions)}, Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')
                            
                            # Convert simulation action to real robot action
                            real_action = convert_sim_to_real_action(sim_action, action_mapping)
                            
                            # Prepare action for execution
                            if len(real_action.shape) == 1:
                                this_target_poses = np.expand_dims(real_action, axis=0)
                            else:
                                this_target_poses = real_action
                            
                            # Store target joint positions for plotting (first 6 joints)
                            target_joints.append(this_target_poses[0, :6])
                            
                            # joint_actions = this_target_poses[:, :6]
                            # gripper_actions = this_target_poses[:, 6:7]
                            # current_joints = env.get_robot_state()['ActualQ'][None, :]
                            # joint_actions = current_joints + (joint_actions - current_joints) * 0.6
                            # this_target_poses = np.concatenate([joint_actions, gripper_actions], axis=1)

                            # Handle timing
                            action_timestamps = np.array([obs_timestamps[-1] + dt])
                            action_exec_latency = command_latency
                            curr_time = time.time()
                            
                            # Check if we're on schedule
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # Behind schedule
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(f'Behind schedule: {action_timestamp - curr_time:.3f}s')
                                action_timestamps = np.array([action_timestamp])
                            
                            # Execute action on robot
                            try:
                                env.exec_actions(
                                    actions=this_target_poses[:1],  # Execute one action at a time
                                    timestamps=action_timestamps[:1]
                                )
                                print(f"Executed action {action_idx}")
                                
                                # Get achieved joint positions after action execution
                                robot_state = env.get_robot_state()
                                achieved_joints.append(robot_state['ActualQ'][:6])
                                
                            except Exception as e:
                                print(f"Error executing action {action_idx}: {e}")
                                break
                            
                            # Show progress
                            vis_img = obs[camera_key][-1]
                            progress = f'Replay: {action_idx+1}/{len(actions)} ({(action_idx+1)/len(actions)*100:.1f}%)'
                            time_remaining = (len(actions) - action_idx - 1) * dt
                            status_text = f'{progress}, {time_remaining:.1f}s remaining'
                            
                            cv2.putText(
                                vis_img,
                                status_text,
                                (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6,
                                thickness=2,
                                color=(0, 255, 255)
                            )
                            cv2.putText(
                                vis_img,
                                'Press "S" to stop',
                                (10, 60),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6,
                                thickness=2,
                                color=(0, 0, 255)
                            )
                            cv2.imshow('Trajectory Replay', vis_img[...,::-1])
                            
                            # Check for stop command
                            key_stroke = cv2.pollKey()
                            if key_stroke == ord('s'):
                                print("Replay stopped by user")
                                break
                            
                            # Check timeout
                            if time.monotonic() - t_start > max_duration:
                                print("Replay terminated by timeout")
                                break
                            
                            # Wait for next action timing
                            precise_wait(t_cycle_end - frame_latency)
                        
                        print("Trajectory replay completed!")
                        env.end_episode()
                        
                    except KeyboardInterrupt:
                        print("Replay interrupted!")
                        env.end_episode()
                    except Exception as e:
                        print(f"Error during replay: {e}")
                        env.end_episode()
                
                time.sleep(0.01)  # Small delay in main loop

                # Plot joint comparison after replay
                if plot_joints:
                    plot_joint_comparison(achieved_joints, target_joints, save_path=os.path.join(output, "joint_comparison.png"))
                        

            cv2.destroyAllWindows()
            print("Replay session ended")

if __name__ == '__main__':
    main() 