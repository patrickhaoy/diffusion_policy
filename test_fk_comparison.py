"""
Test FK comparison: Move robot to dataset joint positions and compare EE poses.
Verifies that our FK implementation matches the simulation.
"""
import os
import time
import click
import numpy as np
import zarr

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Import calibrated FK implementation (REP-103 frame, matching simulation)
from diffusion_policy.real_world.ur5e_kinematics import (
    forward_kinematics_calibrated, get_ee_pose, quat_to_axis_angle,
)


@click.command()
@click.option('--dataset', '-d', required=True, help='Path to zarr dataset directory')
@click.option('--robot_ip', default='192.168.1.10', help='Robot IP address')
@click.option('--episode', '-e', default=0, type=int, help='Episode index')
@click.option('--num_samples', '-n', default=5, type=int, help='Number of positions to test')
@click.option('--tcp_offset', default='0,0,0,0,0,0', type=str, help='TCP offset [x,y,z,rx,ry,rz]')
@click.option('--dry_run', is_flag=True, default=False, help='Only compute, do not move robot')
@click.option('--calibrate', is_flag=True, default=False,
              help='Calibration mode: run over multiple episodes, collect (sim, real) pairs, print mean offset')
@click.option('--num_episodes', default=10, type=int, help='Number of episodes to use in calibrate mode')
def main(dataset, robot_ip, episode, num_samples, tcp_offset, dry_run, calibrate, num_episodes):
    # Z offset between sim world frame and real robot base frame
    SIM_TO_REAL_Z_OFFSET = 0.150  # meters (sim world origin is ~150mm below robot base)
    
    # Parse TCP offset
    tcp_offset_parsed = [float(x) for x in tcp_offset.split(',')]
    
    # Load zarr dataset
    zarr_path = os.path.join(dataset, 'rgb0.zarr')
    if not os.path.exists(zarr_path):
        zarr_path = dataset
    
    print(f"Loading dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    episode_ends = root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    
    if calibrate:
        # Calibration: collect (sim_ee, real_adj) over many poses, then compute mean offset
        episodes_to_use = min(num_episodes, n_episodes)
        print(f"Calibration mode: {episodes_to_use} episodes, {num_samples} samples/episode")
        print("="*80)
    else:
        # Single-episode mode
        if episode >= n_episodes:
            print(f"Error: Episode {episode} does not exist (max: {n_episodes - 1})")
            return
        start_idx = 0 if episode == 0 else episode_ends[episode - 1]
        end_idx = episode_ends[episode]
        episode_length = end_idx - start_idx
        print(f"Episode {episode}: {episode_length} steps")
    
    # Load joint positions and EE poses (for single episode; calibrate loads per-episode below)
    if not calibrate:
        start_idx = 0 if episode == 0 else episode_ends[episode - 1]
        end_idx = episode_ends[episode]
        episode_length = end_idx - start_idx
        arm_joint_pos = root['data/obs/arm_joint_pos'][start_idx:end_idx]
        sim_ee_pose = None
        if 'end_effector_pose' in root['data/obs']:
            sim_ee_pose = root['data/obs/end_effector_pose'][start_idx:end_idx]
        indices = np.linspace(0, episode_length - 1, num_samples, dtype=int)
    
    print(f"TCP offset: {tcp_offset_parsed}")
    print("="*80)
    
    if not calibrate:
        # First, just compute FK and compare with sim (no robot movement)
        print("\n--- FK Comparison (computed vs sim) ---")
        print(f"{'Idx':>5} | {'Pos Error (mm)':>15} | {'Rot Error (deg)':>15} | {'Notes'}")
        print("-"*80)
        
        for idx in indices:
            joints = arm_joint_pos[idx]
            pos, quat = get_ee_pose(joints)
            aa = quat_to_axis_angle(quat)
            computed_ee = np.concatenate([pos, aa])
            if sim_ee_pose is not None:
                sim_ee = sim_ee_pose[idx]
                pos_err = np.linalg.norm(computed_ee[:3] - sim_ee[:3]) * 1000
                rot_err = np.linalg.norm(computed_ee[3:] - sim_ee[3:])
                print(f"{idx:5d} | {pos_err:15.2f} | {np.degrees(rot_err):15.2f} | "
                      f"sim_pos={sim_ee[:3]*1000}, computed={computed_ee[:3]*1000}")
            else:
                print(f"{idx:5d} | {'N/A':>15} | {'N/A':>15} | No sim EE pose in dataset")
    
    if dry_run:
        print("\n[DRY RUN] Not moving robot.")
        return
    
    # Move robot and compare (single episode or calibrate over many)
    print("\n--- Moving Robot and Comparing ---")
    print(f"Using Z offset: {SIM_TO_REAL_Z_OFFSET*1000:.1f} mm (sim world origin below robot base)")
    
    rtde_c = RTDEControlInterface(robot_ip, 500,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
    rtde_r = RTDEReceiveInterface(robot_ip)
    
    # Collect (sim_ee, real_adj) for offset estimation in calibrate mode
    sim_positions = []
    real_adj_positions = []
    
    try:
        if calibrate:
            sample_count = 0
            for ep in range(episodes_to_use):
                s0 = 0 if ep == 0 else episode_ends[ep - 1]
                s1 = episode_ends[ep]
                ep_len = s1 - s0
                if ep_len < 2:
                    continue
                arm_joints = root['data/obs/arm_joint_pos'][s0:s1]
                sim_ee = root['data/obs/end_effector_pose'][s0:s1] if 'end_effector_pose' in root['data/obs'] else None
                if sim_ee is None:
                    continue
                inds = np.linspace(0, ep_len - 1, num_samples, dtype=int)
                for idx in inds:
                    joints = arm_joints[idx]
                    ok = rtde_c.moveJ(joints.tolist(), 0.5, 0.5)
                    if not ok:
                        continue
                    time.sleep(0.4)
                    real_tcp = np.array(rtde_r.getActualTCPPose())
                    real_adj = real_tcp.copy()
                    real_adj[2] -= SIM_TO_REAL_Z_OFFSET
                    sim_positions.append(sim_ee[idx][:3].copy())
                    real_adj_positions.append(real_adj[:3].copy())
                    sample_count += 1
                    if sample_count % 20 == 0:
                        print(f"  Calibration samples: {sample_count}")
            
            if sim_positions and real_adj_positions:
                sim_positions = np.array(sim_positions)
                real_adj_positions = np.array(real_adj_positions)
                # offset such that sim ≈ real_adj + offset  =>  offset = mean(sim - real_adj)
                offset = np.mean(sim_positions - real_adj_positions, axis=0)
                err_per_dim_mm = (sim_positions - real_adj_positions) * 1000  # (N, 3)
                std_per_axis = np.std(err_per_dim_mm, axis=0)
                mean_abs_per_dim = np.mean(np.abs(err_per_dim_mm), axis=0)
                err_after = np.linalg.norm(sim_positions - (real_adj_positions + offset), axis=1) * 1000
                residual_per_dim_mm = (sim_positions - (real_adj_positions + offset)) * 1000
                mean_abs_per_dim_after = np.mean(np.abs(residual_per_dim_mm), axis=0)
                print("\n" + "="*80)
                print("CALIBRATION RESULT (constant offset: sim = real_adj + offset)")
                print(f"  Samples: {len(sim_positions)}")
                print(f"  Offset (m):  [{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}]")
                print(f"  Offset (mm): [{offset[0]*1000:.2f}, {offset[1]*1000:.2f}, {offset[2]*1000:.2f}]")
                print(f"  Mean absolute error per dimension (mm), before offset: [X: {mean_abs_per_dim[0]:.2f}, Y: {mean_abs_per_dim[1]:.2f}, Z: {mean_abs_per_dim[2]:.2f}]")
                print(f"  Mean absolute error per dimension (mm), after offset:  [X: {mean_abs_per_dim_after[0]:.2f}, Y: {mean_abs_per_dim_after[1]:.2f}, Z: {mean_abs_per_dim_after[2]:.2f}]")
                print(f"  Std of (sim - real_adj) per axis (mm): [X: {std_per_axis[0]:.2f}, Y: {std_per_axis[1]:.2f}, Z: {std_per_axis[2]:.2f}]")
                print(f"  Mean 3D error after applying offset: {np.mean(err_after):.2f} mm")
                print("\n  To use in replay_dataset_actions.py, set:")
                print(f"    SIM_TO_REAL_OFFSET = np.array([{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}])  # x,y,z in m")
                print("  Then: real_ee_in_sim_frame = real_tcp[:3] - [0, 0, Z_OFFSET] + SIM_TO_REAL_OFFSET")
                print("  (or real_ee_in_sim_frame = real_adj_xyz + SIM_TO_REAL_OFFSET)")
            else:
                print("  No samples collected (missing end_effector_pose or moveJ failed).")
        else:
            print(f"\n{'Idx':>5} | {'Raw Error (mm)':>15} | {'Z-Adj Error (mm)':>17} | {'XY Error (mm)':>13}")
            print("-"*80)
            all_errors = []
            for idx in indices:
                joints = arm_joint_pos[idx]
                print(f"\nMoving to position {idx}...")
                ok = rtde_c.moveJ(joints.tolist(), 0.5, 0.5)
                if not ok:
                    print(f"  moveJ failed for idx {idx}")
                    continue
                time.sleep(0.5)
                real_joints = np.array(rtde_r.getActualQ())
                real_tcp = np.array(rtde_r.getActualTCPPose())
                real_tcp_sim_frame = real_tcp.copy()
                real_tcp_sim_frame[2] -= SIM_TO_REAL_Z_OFFSET
                if sim_ee_pose is not None:
                    sim_ee = sim_ee_pose[idx]
                    raw_error = np.linalg.norm(real_tcp[:3] - sim_ee[:3]) * 1000
                    z_adj_error = np.linalg.norm(real_tcp_sim_frame[:3] - sim_ee[:3]) * 1000
                    xy_error = np.linalg.norm(real_tcp_sim_frame[:2] - sim_ee[:2]) * 1000
                    z_error = abs(real_tcp_sim_frame[2] - sim_ee[2]) * 1000
                    all_errors.append([raw_error, z_adj_error, xy_error, z_error])
                else:
                    raw_error = z_adj_error = xy_error = z_error = float('nan')
                print(f"{idx:5d} | {raw_error:15.2f} | {z_adj_error:17.2f} | {xy_error:13.2f}")
                print(f"       Joint diff (deg): max={np.max(np.abs(np.degrees(real_joints - joints))):.4f}")
                if sim_ee_pose is not None:
                    print(f"       Sim EE:       [{sim_ee[0]*1000:8.1f}, {sim_ee[1]*1000:8.1f}, {sim_ee[2]*1000:8.1f}] mm")
                    print(f"       Real (raw):   [{real_tcp[0]*1000:8.1f}, {real_tcp[1]*1000:8.1f}, {real_tcp[2]*1000:8.1f}] mm")
                    print(f"       Real (adj):   [{real_tcp_sim_frame[0]*1000:8.1f}, {real_tcp_sim_frame[1]*1000:8.1f}, {real_tcp_sim_frame[2]*1000:8.1f}] mm")
                    print(f"       Error:        [X:{(real_tcp_sim_frame[0]-sim_ee[0])*1000:6.1f}, Y:{(real_tcp_sim_frame[1]-sim_ee[1])*1000:6.1f}, Z:{(real_tcp_sim_frame[2]-sim_ee[2])*1000:6.1f}] mm")
            if all_errors:
                all_errors = np.array(all_errors)
                print("\n" + "="*80)
                print("SUMMARY (with Z offset adjustment):")
                print(f"  Mean XY error:  {np.mean(all_errors[:, 2]):.2f} mm")
                print(f"  Mean Z error:   {np.mean(all_errors[:, 3]):.2f} mm")
                print(f"  Mean 3D error:  {np.mean(all_errors[:, 1]):.2f} mm")
        
        print("\n" + "="*80)
        print("Done!")
        
    finally:
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == '__main__':
    main()
