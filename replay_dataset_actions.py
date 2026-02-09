"""
Replay actions from a dataset to debug action execution.
Loads trajectory from zarr dataset, initializes robot to first state,
and replays actions to verify they look reasonable.

Compares achieved joint positions with dataset and generates comparison plots.
Also compares sim vs real images side by side.
"""
import os
import sys
import time
import click
import numpy as np
import zarr
import torch
import json
from pathlib import Path
from multiprocessing.managers import SharedMemoryManager

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.real_world.real_env import RealEnv


def plot_image_comparison(sim_images, real_images, output_path, episode, camera_name='front_rgb', num_samples=5):
    """Plot side-by-side comparison of sim and real images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_steps = len(sim_images)
    # Sample evenly spaced frames
    indices = np.linspace(0, n_steps - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    for i, idx in enumerate(indices):
        # Sim image (top row)
        sim_img = sim_images[idx]
        if sim_img.dtype == np.uint8:
            sim_img = sim_img.astype(np.float32) / 255.0
        axes[0, i].imshow(sim_img)
        axes[0, i].set_title(f'Sim t={idx}')
        axes[0, i].axis('off')
        
        # Real image (bottom row)
        if idx < len(real_images):
            real_img = real_images[idx]
            if real_img.dtype == np.uint8:
                real_img = real_img.astype(np.float32) / 255.0
            axes[1, i].imshow(real_img)
            
            axes[1, i].set_title(f'Real t={idx}')
        else:
            axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Simulation', fontsize=14)
    axes[1, 0].set_ylabel('Real Robot', fontsize=14)
    
    plt.suptitle(f'{camera_name} - Sim vs Real Comparison (Episode {episode})', fontsize=16)
    plt.tight_layout()
    
    plot_path = output_path / f'image_comparison_{camera_name}_ep{episode}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved image comparison: {plot_path}")
    return plot_path


def plot_all_cameras_comparison(sim_images_dict, real_images_dict, output_path, episode, num_samples=5):
    """Plot comparison for all three cameras in one figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    camera_names = ['front_rgb', 'side_rgb', 'wrist_rgb']
    available_cams = [c for c in camera_names if c in sim_images_dict and c in real_images_dict]
    
    if not available_cams:
        print("No cameras available for comparison")
        return None
    
    n_cams = len(available_cams)
    n_steps = len(sim_images_dict[available_cams[0]])
    indices = np.linspace(0, n_steps - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(n_cams * 2, num_samples, figsize=(3 * num_samples, 4 * n_cams))
    
    for cam_idx, cam_name in enumerate(available_cams):
        sim_images = sim_images_dict[cam_name]
        real_images = real_images_dict.get(cam_name, [])
        
        for i, idx in enumerate(indices):
            # Sim image
            row_sim = cam_idx * 2
            sim_img = sim_images[idx]
            if sim_img.dtype == np.uint8:
                sim_img = sim_img.astype(np.float32) / 255.0
            axes[row_sim, i].imshow(sim_img)
            if i == 0:
                axes[row_sim, i].set_ylabel(f'{cam_name}\n(Sim)', fontsize=10)
            axes[row_sim, i].set_title(f't={idx}' if cam_idx == 0 else '')
            axes[row_sim, i].axis('off')
            
            # Real image
            row_real = cam_idx * 2 + 1
            if idx < len(real_images):
                real_img = real_images[idx]
                if real_img.dtype == np.uint8:
                    real_img = real_img.astype(np.float32) / 255.0
                axes[row_real, i].imshow(real_img)
            else:
                axes[row_real, i].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[row_real, i].transAxes)
            if i == 0:
                axes[row_real, i].set_ylabel(f'{cam_name}\n(Real)', fontsize=10)
            axes[row_real, i].axis('off')
    
    plt.suptitle(f'Sim vs Real Camera Comparison (Episode {episode})', fontsize=14)
    plt.tight_layout()
    
    plot_path = output_path / f'all_cameras_comparison_ep{episode}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved all cameras comparison: {plot_path}")
    return plot_path


def plot_comparison(dataset_joints, real_joints, real_target_joints, sim_target_joints, times, output_path, episode):
    """Plot comparison between dataset, real, and target joint trajectories."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    n_joints = 6
    
    fig, axes = plt.subplots(n_joints, 3, figsize=(20, 3 * n_joints))
    
    for j in range(n_joints):
        # Left plot: Overlay all trajectories
        ax = axes[j, 0]
        ax.plot(times, np.rad2deg(dataset_joints[:, j]), 'b-', label='Sim Position', linewidth=2)
        ax.plot(times, np.rad2deg(sim_target_joints[:, j]), 'c--', label='Sim Target', linewidth=1.5, alpha=0.7)
        ax.plot(times, np.rad2deg(real_joints[:, j]), 'r-', label='Real Position', linewidth=2)
        ax.plot(times, np.rad2deg(real_target_joints[:, j]), 'm--', label='Real Target', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'{joint_names[j]} - Trajectory Comparison')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Middle plot: Target tracking errors (position - target for both sim and real)
        ax = axes[j, 1]
        sim_tracking_error = np.rad2deg(dataset_joints[:, j] - sim_target_joints[:, j])
        real_tracking_error = np.rad2deg(real_joints[:, j] - real_target_joints[:, j])
        ax.plot(times, sim_tracking_error, 'b-', label='Sim Tracking (Pos-Target)', linewidth=2)
        ax.plot(times, real_tracking_error, 'r-', label='Real Tracking (Pos-Target)', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (deg)')
        ax.set_title(f'{joint_names[j]} - Target Tracking Error')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Right plot: Sim vs Real position error
        ax = axes[j, 2]
        position_error = np.rad2deg(real_joints[:, j] - dataset_joints[:, j])
        ax.plot(times, position_error, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(times, position_error, alpha=0.3, color='green')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (deg)')
        ax.set_title(f'{joint_names[j]} - Sim vs Real Position Error')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / f'comparison_ep{episode}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_path}")
    
    return plot_path


def plot_ee_comparison(sim_ee, real_ee, times, output_path, episode):
    """Plot EE pose comparison between sim and real."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    labels = ['X', 'Y', 'Z']
    
    # Position comparison (top row)
    for i in range(3):
        ax = axes[0, i]
        ax.plot(times, sim_ee[:, i] * 1000, 'b-', label='Sim', linewidth=2)
        ax.plot(times, real_ee[:, i] * 1000, 'r-', label='Real', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (mm)')
        ax.set_title(f'EE Position - {labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Rotation comparison (bottom row)
    rot_labels = ['RX', 'RY', 'RZ']
    for i in range(3):
        ax = axes[1, i]
        ax.plot(times, np.degrees(sim_ee[:, i+3]), 'b-', label='Sim', linewidth=2)
        ax.plot(times, np.degrees(real_ee[:, i+3]), 'r-', label='Real', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotation (deg)')
        ax.set_title(f'EE Rotation - {rot_labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'End-Effector Pose Comparison (Episode {episode})', fontsize=16)
    plt.tight_layout()
    
    plot_path = output_path / f'ee_comparison_ep{episode}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved EE comparison plot: {plot_path}")
    return plot_path


def compute_metrics(dataset_joints, real_joints, target_joints=None):
    """Compute tracking error metrics."""
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    
    metrics = {
        'dataset_error': {'per_joint': {}, 'overall': {}},
        'target_error': {'per_joint': {}, 'overall': {}} if target_joints is not None else None
    }
    
    def compute_error_metrics(errors_rad, label):
        errors_deg = np.rad2deg(errors_rad)
        
        print(f"\n{label} Per-Joint Errors (degrees):")
        print("-" * 60)
        print(f"{'Joint':<15} {'Mean':>10} {'Std':>10} {'Max':>10} {'RMSE':>10}")
        print("-" * 60)
        
        per_joint = {}
        for j, name in enumerate(joint_names):
            mean_err = np.mean(errors_deg[:, j])
            std_err = np.std(errors_deg[:, j])
            max_err = np.max(np.abs(errors_deg[:, j]))
            rmse = np.sqrt(np.mean(errors_deg[:, j]**2))
            
            per_joint[name] = {
                'mean_deg': mean_err,
                'std_deg': std_err,
                'max_deg': max_err,
                'rmse_deg': rmse
            }
            
            print(f"{name:<15} {mean_err:>10.3f} {std_err:>10.3f} {max_err:>10.3f} {rmse:>10.3f}")
        
        overall_rmse = np.sqrt(np.mean(errors_deg**2))
        overall_max = np.max(np.abs(errors_deg))
        overall_mean = np.mean(np.abs(errors_deg))
        
        overall = {
            'rmse_deg': overall_rmse,
            'max_deg': overall_max,
            'mean_abs_deg': overall_mean
        }
        
        print("-" * 60)
        print(f"{'OVERALL':<15} {overall_mean:>10.3f} {'-':>10} {overall_max:>10.3f} {overall_rmse:>10.3f}")
        
        return per_joint, overall
    
    print("\n" + "="*70)
    print("TRACKING ERROR METRICS")
    print("="*70)
    
    # Target tracking error (how well real tracks commanded targets)
    if target_joints is not None:
        target_errors = real_joints - target_joints
        per_joint, overall = compute_error_metrics(target_errors, "TARGET TRACKING (Real - Target)")
        metrics['target_error']['per_joint'] = per_joint
        metrics['target_error']['overall'] = overall
    
    # Dataset error (how well real matches dataset/sim)
    dataset_errors = real_joints - dataset_joints
    per_joint, overall = compute_error_metrics(dataset_errors, "DATASET ERROR (Real - Dataset)")
    metrics['dataset_error']['per_joint'] = per_joint
    metrics['dataset_error']['overall'] = overall
    
    print("="*70)
    
    return metrics


@click.command()
@click.option('--dataset', '-d', required=True, 
              help='Path to zarr dataset directory')
@click.option('--robot_ip', default='192.168.1.10',
              help='Robot IP address')
@click.option('--episode', '-e', default=0, type=int,
              help='Episode index to replay')
@click.option('--frequency', '-f', default=10.0, type=float,
              help='Control frequency in Hz')
@click.option('--action_scale', default=1.0, type=float,
              help='Scale factor for actions (for relative actions)')
@click.option('--relative_actions', is_flag=True, default=False,
              help='Treat actions as relative joint deltas (add to current real pos)')
@click.option('--sim_relative', is_flag=True, default=False,
              help='Compute targets relative to sim joint pos (same targets as sim)')
@click.option('--dry_run', is_flag=True, default=False,
              help='Print actions without executing on robot')
@click.option('--output', '-o', default='replay_output',
              help='Output directory for replay data')
def main(dataset, robot_ip, episode, frequency, action_scale, relative_actions, sim_relative, 
         dry_run, output):
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
    
    # Load EE pose for comparison (if available)
    sim_ee_pose = None
    if 'end_effector_pose' in root['data/obs']:
        sim_ee_pose = root['data/obs/end_effector_pose'][start_idx:end_idx]
        print(f"Loaded end_effector_pose: shape={sim_ee_pose.shape}")
    
    # Load sim images for comparison
    sim_images = {}
    for cam_key in ['front_rgb', 'side_rgb', 'wrist_rgb']:
        if cam_key in root['data/obs']:
            sim_images[cam_key] = root['data/obs'][cam_key][start_idx:end_idx]
            print(f"Loaded {cam_key}: shape={sim_images[cam_key].shape}")
    
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
        mode = f"SIM_RELATIVE={sim_relative}, RELATIVE={relative_actions}"
        print(f"\n[DRY RUN] {mode}. Not executing on robot. Exiting.")
        return
    
    # Execute on robot
    print(f"\n{'='*60}")
    print("Starting robot execution...")
    if sim_relative:
        print("Mode: SIM_RELATIVE - targets = sim_pos[i] + action[i] (same as sim)")
    elif relative_actions:
        print("Mode: RELATIVE - targets = current_real_pos + action[i]")
    else:
        print("Mode: ABSOLUTE - targets = action[i]")
    print(f"{'='*60}")
    
    dt = 1.0 / frequency
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Storage for comparison
    real_joint_positions = []
    target_joint_positions = []
    real_ee_poses = []  # Real EE poses for comparison
    real_timestamps = []
    real_images = {'front_rgb': [], 'side_rgb': [], 'wrist_rgb': []}
    
    # Load camera configs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs = [
        json.load(open(os.path.join(script_dir, "diffusion_policy/real_world/realsense_config/455_front.json"))),
        json.load(open(os.path.join(script_dir, "diffusion_policy/real_world/realsense_config/435_side.json"))),
        json.load(open(os.path.join(script_dir, "diffusion_policy/real_world/realsense_config/415_wrist.json")))
    ]
    
    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=2,
            obs_image_resolution=(224, 224),  # Match sim image resolution
            max_obs_buffer_size=30,
            obs_float32=False,
            init_joints=True,
            custom_init_joints=init_joints.tolist(),
            # Recording
            record_raw_video=True,
            enable_multi_cam_vis=True,
            camera_serial_numbers=['215122255213', '832112070487', '746112060198'],
            camera_configs=configs,
            thread_per_video=3,
            video_crf=21,
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
                
                # Get current state and observations
                robot_state = env.get_robot_state()
                current_joints = np.array(robot_state['ActualQ'])
                obs = env.get_obs()
                
                # Record current joint position for comparison
                real_joint_positions.append(current_joints.copy())
                real_timestamps.append(time.time() - t_start)
                
                # Get real EE pose from RTDE (uses robot's configured TCP)
                real_tcp = np.array(robot_state.get('ActualTCPPose', robot_state.get('TargetTCPPose', np.zeros(6))))
                real_ee_poses.append(real_tcp.copy())
                
                # Capture real images (take the latest frame)
                for cam_key in ['front_rgb', 'side_rgb', 'wrist_rgb']:
                    if cam_key in obs:
                        # obs[cam_key] shape is (n_obs_steps, H, W, C), take latest
                        img = obs[cam_key][-1]
                        real_images[cam_key].append(img.copy())
                
                # Joint position mode
                if sim_relative:
                    # Use same targets as sim: sim_pos[i] + action[i]
                    joint_targets = arm_joint_pos[i] + action[:6] * action_scale
                elif relative_actions:
                    # Treat as relative: add scaled delta to current real pos
                    joint_targets = current_joints + action[:6] * action_scale
                else:
                    # Treat as absolute: use action directly (scaled)
                    joint_targets = action[:6] * action_scale
                
                target_joint_positions.append(joint_targets.copy())
                gripper_action = action[6]
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
            
            final_joints = np.array(env.get_robot_state()['ActualQ'])
            print(f"\nFinal joint positions: {final_joints}")
            print(f"Dataset final joints:  {arm_joint_pos[-1]}")
    
    # Convert to arrays
    real_joint_positions = np.array(real_joint_positions)
    real_target_positions = np.array(target_joint_positions)
    real_ee_poses = np.array(real_ee_poses)
    real_timestamps = np.array(real_timestamps)
    
    # Compute sim targets: sim_pos[i] + action[i] (what sim was commanded)
    # Note: sim_target[i] should result in sim ending up at sim_pos[i+1]
    sim_target_positions = arm_joint_pos[:-1] + actions[:-1, :6] * action_scale
    
    # Align lengths (use shorter of all)
    min_len = min(len(arm_joint_pos) - 1, len(real_joint_positions), len(real_target_positions))
    dataset_joints_aligned = arm_joint_pos[:min_len]
    real_joints_aligned = real_joint_positions[:min_len]
    real_target_aligned = real_target_positions[:min_len]
    sim_target_aligned = sim_target_positions[:min_len]
    real_ee_aligned = real_ee_poses[:min_len]
    times_aligned = np.arange(min_len) * dt
    
    # Align sim EE poses if available
    sim_ee_aligned = None
    if sim_ee_pose is not None:
        sim_ee_aligned = sim_ee_pose[:min_len]
    
    print(f"\nComparing {min_len} timesteps...")
    
    # Compute joint metrics (both target tracking and dataset error)
    metrics = compute_metrics(dataset_joints_aligned, real_joints_aligned, real_target_aligned)
    
    # Compute EE pose metrics if sim EE pose available
    if sim_ee_aligned is not None:
        print("\n" + "="*70)
        print("END-EFFECTOR POSE COMPARISON")
        print("="*70)
        
        ee_pos_error = real_ee_aligned[:, :3] - sim_ee_aligned[:, :3]
        ee_rot_error = real_ee_aligned[:, 3:6] - sim_ee_aligned[:, 3:6]
        
        pos_rmse = np.sqrt(np.mean(ee_pos_error**2)) * 1000  # mm
        pos_max = np.max(np.abs(ee_pos_error)) * 1000  # mm
        rot_rmse = np.sqrt(np.mean(ee_rot_error**2))  # rad
        rot_max = np.max(np.abs(ee_rot_error))  # rad
        
        print(f"Position error (mm):")
        print(f"  RMSE: {pos_rmse:.2f} mm")
        print(f"  Max:  {pos_max:.2f} mm")
        print(f"  Per-axis RMSE: X={np.sqrt(np.mean(ee_pos_error[:, 0]**2))*1000:.2f}, "
              f"Y={np.sqrt(np.mean(ee_pos_error[:, 1]**2))*1000:.2f}, "
              f"Z={np.sqrt(np.mean(ee_pos_error[:, 2]**2))*1000:.2f}")
        
        print(f"\nRotation error (deg):")
        print(f"  RMSE: {np.degrees(rot_rmse):.2f} deg")
        print(f"  Max:  {np.degrees(rot_max):.2f} deg")
        
        metrics['ee_pose'] = {
            'pos_rmse_mm': pos_rmse,
            'pos_max_mm': pos_max,
            'rot_rmse_deg': np.degrees(rot_rmse),
            'rot_max_deg': np.degrees(rot_max),
        }
        print("="*70)
        
        # Plot EE pose comparison
        plot_ee_comparison(sim_ee_aligned, real_ee_aligned, times_aligned, output_path, episode)
    
    # Compute sim tracking metrics for comparison
    print("\nSIM TARGET TRACKING (Sim Position - Sim Target):")
    sim_tracking_errors = np.rad2deg(dataset_joints_aligned - sim_target_aligned)
    print(f"  Overall RMSE: {np.sqrt(np.mean(sim_tracking_errors**2)):.3f} deg")
    print(f"  Overall Max:  {np.max(np.abs(sim_tracking_errors)):.3f} deg")
    
    # Generate joint comparison plot
    plot_path = plot_comparison(
        dataset_joints_aligned, real_joints_aligned, real_target_aligned, sim_target_aligned,
        times_aligned, output_path, episode
    )
    
    # Generate image comparison plots
    if sim_images and real_images:
        # Convert real_images lists to arrays
        real_images_arrays = {}
        for cam_key in real_images:
            if real_images[cam_key]:
                real_images_arrays[cam_key] = np.array(real_images[cam_key])
                print(f"Captured {len(real_images[cam_key])} real {cam_key} images")
        
        # Plot all cameras comparison
        plot_all_cameras_comparison(sim_images, real_images_arrays, output_path, episode, num_samples=6)
        
        # Also plot individual camera comparisons
        for cam_key in sim_images:
            if cam_key in real_images_arrays:
                plot_image_comparison(
                    sim_images[cam_key], real_images_arrays[cam_key], 
                    output_path, episode, camera_name=cam_key, num_samples=6
                )
    
    # Save data for further analysis
    save_data = {
        'dataset_joints': torch.from_numpy(dataset_joints_aligned).float(),
        'real_joints': torch.from_numpy(real_joints_aligned).float(),
        'real_target_joints': torch.from_numpy(real_target_aligned).float(),
        'sim_target_joints': torch.from_numpy(sim_target_aligned).float(),
        'real_ee_poses': torch.from_numpy(real_ee_aligned).float(),
        'sim_ee_poses': torch.from_numpy(sim_ee_aligned).float() if sim_ee_aligned is not None else None,
        'times': torch.from_numpy(times_aligned).float(),
        'actions': torch.from_numpy(actions[:min_len]).float(),
        'metrics': metrics,
        'episode': episode,
        'frequency': frequency,
        'action_scale': action_scale,
        'relative_actions': relative_actions,
        'sim_relative': sim_relative,
    }
    save_path = output_path / f'comparison_ep{episode}.pt'
    torch.save(save_data, save_path)
    print(f"Saved comparison data: {save_path}")


if __name__ == '__main__':
    main()
