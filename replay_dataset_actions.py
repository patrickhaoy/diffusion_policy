"""
Replay actions from a dataset to debug action execution.
Loads trajectory from zarr dataset, initializes robot to first state,
and replays actions to verify they look reasonable.
"""
import os
import sys
import time
import click
import numpy as np
import zarr
from multiprocessing.managers import SharedMemoryManager

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.real_world.real_env import RealEnv


@click.command()
@click.option('--dataset', '-d', required=True, 
              help='Path to zarr dataset directory')
@click.option('--robot_ip', default='192.168.0.139',
              help='Robot IP address')
@click.option('--episode', '-e', default=0, type=int,
              help='Episode index to replay')
@click.option('--frequency', '-f', default=10.0, type=float,
              help='Control frequency in Hz')
@click.option('--action_scale', default=1.0, type=float,
              help='Scale factor for actions (for relative actions)')
@click.option('--relative_actions', is_flag=True, default=False,
              help='Treat actions as relative joint deltas (add to current)')
@click.option('--dry_run', is_flag=True, default=False,
              help='Print actions without executing on robot')
@click.option('--output', '-o', default='replay_output',
              help='Output directory for replay data')
def main(dataset, robot_ip, episode, frequency, action_scale, relative_actions, dry_run, output):
    # Load zarr dataset
    zarr_path = os.path.join(dataset, 'rgb0.zarr')
    if not os.path.exists(zarr_path):
        # Try direct path
        zarr_path = dataset
    
    print(f"Loading dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    print(f"Dataset has {n_episodes} episodes")
    print(f"Episode ends: {episode_ends}")
    
    if episode >= n_episodes:
        print(f"Error: Episode {episode} does not exist (max: {n_episodes - 1})")
        return
    
    # Get episode slice
    start_idx = 0 if episode == 0 else episode_ends[episode - 1]
    end_idx = episode_ends[episode]
    episode_length = end_idx - start_idx
    print(f"\nReplaying episode {episode}: steps {start_idx} to {end_idx} ({episode_length} steps)")
    
    # Load data for this episode
    actions = root['data/actions'][start_idx:end_idx]
    arm_joint_pos = root['data/obs/arm_joint_pos'][start_idx:end_idx]
    
    print(f"\nActions shape: {actions.shape}")
    print(f"Joint positions shape: {arm_joint_pos.shape}")
    
    # Get initial joint position
    init_joints = arm_joint_pos[0]
    print(f"\nInitial joint positions (rad): {init_joints}")
    print(f"Initial joint positions (deg): {np.rad2deg(init_joints)}")
    
    # Print action statistics
    print(f"\nAction statistics:")
    print(f"  Joint actions (first 6 dims):")
    print(f"    Mean: {np.mean(actions[:, :6], axis=0)}")
    print(f"    Std:  {np.std(actions[:, :6], axis=0)}")
    print(f"    Min:  {np.min(actions[:, :6], axis=0)}")
    print(f"    Max:  {np.max(actions[:, :6], axis=0)}")
    print(f"  Gripper actions (last dim):")
    print(f"    Mean: {np.mean(actions[:, 6])}")
    print(f"    Min:  {np.min(actions[:, 6])}")
    print(f"    Max:  {np.max(actions[:, 6])}")
    
    # Print first few actions
    print(f"\nFirst 5 actions:")
    for i in range(min(5, len(actions))):
        print(f"  [{i}] joints: {actions[i, :6]}, gripper: {actions[i, 6]:.3f}")
    
    # Print joint position changes in dataset
    print(f"\nJoint position changes in dataset (first 5 steps):")
    for i in range(min(5, len(arm_joint_pos) - 1)):
        delta = arm_joint_pos[i + 1] - arm_joint_pos[i]
        print(f"  [{i}] delta: {delta}")
    
    # Check if actions look like relative or absolute
    print(f"\nAnalyzing if actions are relative or absolute...")
    # Compare action magnitudes to joint position deltas
    joint_deltas = np.diff(arm_joint_pos, axis=0)
    action_joints = actions[:-1, :6]  # Exclude last action for comparison
    
    # Correlation between actions and deltas
    correlation = np.corrcoef(action_joints.flatten(), joint_deltas.flatten())[0, 1]
    print(f"  Correlation between actions[:,:6] and joint deltas: {correlation:.3f}")
    
    # Check if actions are closer to absolute positions
    abs_diff = np.mean(np.abs(action_joints - arm_joint_pos[:-1]))
    delta_diff = np.mean(np.abs(action_joints - joint_deltas))
    print(f"  Mean |action - joint_pos|: {abs_diff:.6f}")
    print(f"  Mean |action - delta|: {delta_diff:.6f}")
    
    if abs_diff < delta_diff:
        print("  -> Actions appear to be ABSOLUTE joint positions")
    else:
        print("  -> Actions appear to be RELATIVE joint deltas")
    
    if dry_run:
        print("\n[DRY RUN] Not executing on robot. Exiting.")
        return
    
    # Execute on robot
    print(f"\n{'='*60}")
    print("Starting robot execution...")
    print(f"{'='*60}")
    
    dt = 1.0 / frequency
    
    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=2,
            obs_image_resolution=(640, 480),
            max_obs_buffer_size=30,
            obs_float32=False,
            init_joints=True,
            custom_init_joints=init_joints.tolist(),
            video_capture_fps=30,
            video_capture_resolution=(640, 480),
            record_raw_video=True,
            enable_multi_cam_vis=True,
            shm_manager=shm_manager
        ) as env:
            
            print("Robot initialized. Waiting for cameras...")
            time.sleep(3.0)
            
            print(f"\nCurrent joint positions: {env.get_robot_state()['ActualQ']}")
            print(f"Target initial joints:   {init_joints}")
            
            input("\nPress Enter to start replaying actions...")
            
            # Start episode recording
            t_start = time.time()
            env.start_episode(t_start)
            
            print(f"\nReplaying {episode_length} actions at {frequency} Hz...")
            
            for i, action in enumerate(actions):
                iter_start = time.time()
                
                # Get current state
                current_joints = env.get_robot_state()['ActualQ']
                
                if relative_actions:
                    # Treat as relative: add scaled delta to current
                    joint_targets = current_joints + action[:6] * action_scale
                else:
                    # Treat as absolute: use action directly (scaled)
                    joint_targets = action[:6] * action_scale
                
                gripper_action = action[6]
                
                # Combine into 7D action
                full_action = np.concatenate([joint_targets, [gripper_action]])
                
                # Compute timestamp for this action
                action_time = t_start + (i + 1) * dt
                
                # Execute
                env.exec_actions(
                    actions=full_action[None, :],  # Add batch dim
                    timestamps=np.array([action_time])
                )
                
                if i % 10 == 0:
                    print(f"Step {i}/{episode_length}: "
                          f"target={joint_targets[:3]}, "
                          f"current={current_joints[:3]}, "
                          f"gripper={gripper_action:.2f}")
                
                # Wait for next step
                elapsed = time.time() - iter_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print("\nReplay complete!")
            
            # End episode
            env.end_episode()
            
            print(f"\nFinal joint positions: {env.get_robot_state()['ActualQ']}")
            print(f"Dataset final joints:  {arm_joint_pos[-1]}")


if __name__ == '__main__':
    main()
