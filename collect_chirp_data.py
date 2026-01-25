"""
Collect random waypoint data for PACE system identification.

Usage:
python collect_chirp_data.py --robot_ip 192.168.1.10 --output data/ur5e_pace

This script:
1. Samples random EE waypoints within workspace bounds
2. Uses IK to get joint targets for each waypoint
3. Interpolates between waypoints in joint space
4. Executes with PD torque control at 500Hz (controller frequency)
5. Sends actions at 100Hz (every 5th control step)
6. Logs data at 100Hz (matches action frequency)

This approach provides much better joint space coverage than chirp trajectories
since the joints move through their full range while traversing the workspace.

Note: PD controller matches IdealPDActuator behavior (no position error clipping, only torque clipping).
"""

import numpy as np
import time
import torch
import click
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


# Workspace bounds (measured from robot workspace corners, rounded for safety)
WORKSPACE_BOUNDS = {
    'x_min': -0.70,    # Rightmost (rounded down from -0.721)
    'x_max': -0.40,    # Leftmost (rounded down from -0.393)
    'y_min': -0.30,    # Front right (rounded down from -0.349 to avoid stretching)
    'y_max': 0.12,     # Back left (rounded down from 0.139)
    'z_min': 0.03,     # Bottom (rounded up from 0.022 for safety)
    'z_max': 0.30,     # Top (rounded down from 0.308)
}


def compute_pd_torque(target_joints: np.ndarray, 
                      curr_joints: np.ndarray, 
                      curr_vel: np.ndarray,
                      torque_kp: np.ndarray,
                      torque_kd: np.ndarray,
                      torque_max: np.ndarray) -> np.ndarray:
    """
    Compute PD torque command from joint position error.
    Matches DCMotor behavior: no position error clipping, only torque clipping.
    """
    # Position error with clipping (same limits as rtde_interpolation_controller.py)
    q_err = target_joints - curr_joints
    
    # PD control: torque = Kp * pos_error - Kd * vel
    torque_d = -torque_kd * curr_vel
    torque_target = torque_kp * q_err + torque_d
    
    # Clamp total torque to max limits (only clipping)
    torque_target = np.clip(torque_target, -torque_max, torque_max)
    
    return torque_target.astype(float)


def pose_to_list(pos: np.ndarray, rot: R) -> list:
    """Convert position and rotation to UR pose format [x,y,z,rx,ry,rz]."""
    rotvec = rot.as_rotvec()
    return [pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2]]


def generate_random_waypoints(n_waypoints: int, home_rot: R, max_rot_angle: float, seed: int = 0):
    """Generate random EE waypoints within workspace bounds.
    
    Args:
        n_waypoints: Number of waypoints to generate
        home_rot: Reference rotation (home orientation)
        max_rot_angle: Maximum rotation from home in radians
        seed: Random seed for reproducibility
        
    Returns:
        waypoint_positions: List of [x, y, z] positions
        waypoint_rotations: List of scipy Rotation objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    waypoint_positions = []
    waypoint_rotations = []
    
    # Get home Euler angles for direct offset
    home_euler = home_rot.as_euler('xyz')
    
    for _ in range(n_waypoints):
        # Random position within workspace bounds
        pos = np.array([
            np.random.uniform(WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['x_max']),
            np.random.uniform(WORKSPACE_BOUNDS['y_min'], WORKSPACE_BOUNDS['y_max']),
            np.random.uniform(WORKSPACE_BOUNDS['z_min'], WORKSPACE_BOUNDS['z_max']),
        ])
        waypoint_positions.append(pos)
        
        # Random orientation: add offset directly in Euler space
        # This ensures each Euler angle stays within ±max_rot_angle of home
        target_euler = home_euler + np.array([
            np.random.uniform(-max_rot_angle, max_rot_angle),
            np.random.uniform(-max_rot_angle, max_rot_angle),
            np.random.uniform(-max_rot_angle, max_rot_angle),
        ])
        target_rot = R.from_euler('xyz', target_euler)
        waypoint_rotations.append(target_rot)
    
    return waypoint_positions, waypoint_rotations


def solve_waypoint_ik(rtde_c, waypoint_positions, waypoint_rotations, initial_joints):
    """Solve IK for all waypoints.
    
    Args:
        rtde_c: RTDE control interface
        waypoint_positions: List of [x, y, z] positions
        waypoint_rotations: List of scipy Rotation objects
        initial_joints: Starting joint configuration
        
    Returns:
        waypoint_joints: List of joint configurations (or None for failed IK)
        ik_success: List of booleans indicating IK success
    """
    waypoint_joints = []
    ik_success = []
    last_valid = initial_joints.copy()
    
    for pos, rot in zip(waypoint_positions, waypoint_rotations):
        target_pose = pose_to_list(pos, rot)
        ik_solution = rtde_c.getInverseKinematics(target_pose, last_valid.tolist())
        
        if ik_solution is not None and len(ik_solution) == 6:
            joints = np.array(ik_solution)
            # Check for reasonable solution
            joint_delta = np.abs(joints - last_valid)
            if np.max(joint_delta) < np.pi:  # Allow large moves between waypoints
                waypoint_joints.append(joints)
                ik_success.append(True)
                last_valid = joints.copy()
            else:
                waypoint_joints.append(last_valid.copy())
                ik_success.append(False)
        else:
            waypoint_joints.append(last_valid.copy())
            ik_success.append(False)
    
    return waypoint_joints, ik_success


def interpolate_trajectory(waypoint_joints, waypoint_times, control_frequency):
    """Interpolate joint trajectory between waypoints.
    
    Args:
        waypoint_joints: List of joint configurations at each waypoint
        waypoint_times: Time to reach each waypoint (cumulative)
        control_frequency: Control loop frequency (Hz)
        
    Returns:
        times: Array of timesteps
        joint_targets: Array of joint targets at each timestep
    """
    total_duration = waypoint_times[-1]
    n_steps = int(total_duration * control_frequency)
    
    times = np.linspace(0, total_duration, n_steps)
    joint_targets = np.zeros((n_steps, 6))
    
    for step_idx, t in enumerate(times):
        # Find which segment we're in
        wp_idx = np.searchsorted(waypoint_times, t, side='right') - 1
        wp_idx = max(0, min(wp_idx, len(waypoint_joints) - 2))
        
        # Linear interpolation
        t_start = waypoint_times[wp_idx]
        t_end = waypoint_times[wp_idx + 1]
        alpha = (t - t_start) / (t_end - t_start) if t_end > t_start else 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        
        j_start = waypoint_joints[wp_idx]
        j_end = waypoint_joints[wp_idx + 1]
        joint_targets[step_idx] = j_start + alpha * (j_end - j_start)
    
    return times, joint_targets


def collect_segment(rtde_c, rtde_r, waypoint_joints, waypoint_positions, waypoint_rotations,
                   segment_duration, control_frequency, action_frequency, action_decimation,
                   torque_kp, torque_kd, torque_max, move_speed, move_accel, segment_name):
    """Collect data while moving through waypoints with PD torque control.
    
    Args:
        waypoint_joints: List of joint configurations at each waypoint
        waypoint_positions: List of EE positions for each waypoint
        waypoint_rotations: List of EE rotations for each waypoint
        segment_duration: Total duration for segment (seconds)
        
    Returns:
        dict: Segment data including time, joint_pos, joint_target, etc.
    """
    n_waypoints = len(waypoint_joints)
    
    # Move to first waypoint
    print(f"[INFO]: Moving to first waypoint for {segment_name}...")
    ok = rtde_c.moveJ(waypoint_joints[0].tolist(), move_speed, move_accel)
    if not ok:
        raise RuntimeError(f"moveJ to first waypoint failed for {segment_name}")
    time.sleep(0.5)
    
    # Compute time for each waypoint (evenly distributed)
    waypoint_times = np.linspace(0, segment_duration, n_waypoints)
    
    # Precompute interpolated trajectory at action frequency (100Hz)
    print(f"[INFO]: Precomputing trajectory ({n_waypoints} waypoints over {segment_duration}s)...")
    times, joint_targets = interpolate_trajectory(waypoint_joints, waypoint_times, action_frequency)
    n_steps = len(times)
    print(f"[INFO]: Trajectory ready - {n_steps} steps at {action_frequency}Hz")
    
    # Data storage
    time_data = []
    joint_pos_data = []
    joint_vel_data = []
    joint_target_pos_data = []
    torque_cmd_data = []
    ee_pose_data = []
    
    # Timing
    t_start_traj = time.time()
    iter_idx = 0
    action_counter = 0  # Counter for action decimation
    
    # Initialize target and torque command
    curr_joints_init = np.array(rtde_r.getActualQ(), dtype=float)
    curr_vel_init = np.array(rtde_r.getActualQd(), dtype=float)
    target_joints = joint_targets[0]
    torque_cmd = compute_pd_torque(
        target_joints, curr_joints_init, curr_vel_init,
        torque_kp, torque_kd, torque_max
    )
    
    print(f"[INFO]: Executing trajectory for {segment_name}...")
    print(f"[INFO]: Controller running at {control_frequency}Hz, actions/logging at {action_frequency}Hz")
    
    while True:
        t_loop_start = rtde_c.initPeriod()
        
        # Current time in trajectory
        t_current = time.time()
        t_traj = t_current - t_start_traj
        
        if t_traj >= segment_duration:
            break
        
        # Get current state (always read at 500Hz for accurate control)
        curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
        curr_vel = np.array(rtde_r.getActualQd(), dtype=float)
        curr_pose = rtde_r.getActualTCPPose()
        
        # Only update target and send command at action frequency (100Hz)
        if action_counter % action_decimation == 0:
            # Look up precomputed target at action frequency
            step_idx = min(int(t_traj * action_frequency), n_steps - 1)
            target_joints = joint_targets[step_idx]
            
            # Compute PD torque command
            torque_cmd = compute_pd_torque(
                target_joints, curr_joints, curr_vel,
                torque_kp, torque_kd, torque_max
            )
            
            # Log data at action frequency (100Hz)
            time_data.append(t_traj)
            joint_pos_data.append(curr_joints.copy())
            joint_vel_data.append(curr_vel.copy())
            joint_target_pos_data.append(target_joints.copy())
            torque_cmd_data.append(torque_cmd.copy())
            ee_pose_data.append(np.array(curr_pose))
        
        # Send torque command at control frequency (500Hz) - use last computed command
        rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)
        
        # Progress print
        if iter_idx % 1000 == 0:
            progress = t_traj / segment_duration * 100
            joint_err = np.max(np.abs(target_joints - curr_joints))
            print(f"  [{segment_name}] Progress: {progress:.1f}%, Time: {t_traj:.1f}s, Joint err: {np.degrees(joint_err):.1f}°")
        
        rtde_c.waitPeriod(t_loop_start)
        iter_idx += 1
        action_counter += 1
    
    # Return segment data
    segment_data = {
        "time": torch.from_numpy(np.array(time_data)).float(),
        "dof_pos": torch.from_numpy(np.array(joint_pos_data)).float(),
        "des_dof_pos": torch.from_numpy(np.array(joint_target_pos_data)).float(),
        "joint_pos": torch.from_numpy(np.array(joint_pos_data)).float(),
        "joint_vel": torch.from_numpy(np.array(joint_vel_data)).float(),
        "action": torch.from_numpy(np.array(joint_target_pos_data)).float(),
        "torque_cmd": torch.from_numpy(np.array(torque_cmd_data)).float(),
        "ee_pose": torch.from_numpy(np.array(ee_pose_data)).float(),
        "waypoint_joints": torch.from_numpy(np.array(waypoint_joints)).float(),
        "waypoint_positions": torch.from_numpy(np.array(waypoint_positions)).float(),
        "n_waypoints": n_waypoints,
        "n_samples": len(time_data),
    }
    
    print(f"[INFO]: Collected {len(time_data)} samples for {segment_name}")
    return segment_data


@click.command()
@click.option('--output', '-o', default='data/ur5e_pace', help="Directory to save data.")
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5's IP address")
@click.option('--segment_duration', '-d', default=20.0, type=float, help="Duration per segment (seconds).")
@click.option('--n_waypoints', '-nw', default=15, type=int, help="Number of waypoints per segment.")
@click.option('--max_rot_angle', '-mra', default=45.0, type=float, help="Max rotation angle from home (degrees).")
@click.option('--n_train_segments', '-nt', default=3, type=int, help="Number of training segments.")
@click.option('--n_test_segments', '-nv', default=1, type=int, help="Number of test segments.")
@click.option('--seed', '-s', default=42, type=int, help="Random seed for reproducibility.")
def main(output, robot_ip, segment_duration, n_waypoints, max_rot_angle, 
         n_train_segments, n_test_segments, seed):
    """Collect random waypoint data for PACE system identification."""
    
    np.random.seed(seed)
    
    # Home joint configuration
    j_home_deg = np.array([0, -90, 90, -90, -90, 0], dtype=float)
    j_home = j_home_deg / 180.0 * np.pi
    
    # Control frequency (500Hz required by UR for torque control)
    control_frequency = 500
    # Action/logging frequency (10Hz - matches policy deployment frequency)
    action_frequency = 10
    action_decimation = control_frequency // action_frequency  # 50
    
    # Connect to robot
    rtde_c = RTDEControlInterface(
        robot_ip, 
        control_frequency,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT
    )
    rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)

    try:
        move_speed = 1.05  # rad/s
        move_accel = 1.4   # rad/s^2
        
        # Set TCP offset (gripper offset)
        tcp_offset_x = 0.0
        tcp_offset_y = 0.0
        tcp_offset_z = 0.1500  # 150.0 mm
        tcp_offset_pose = [tcp_offset_x, tcp_offset_y, tcp_offset_z, 0, 0, 0]
        rtde_c.setTcp(tcp_offset_pose)
        print(f"[INFO]: Set TCP offset: X={tcp_offset_x:.4f}, Y={tcp_offset_y:.4f}, Z={tcp_offset_z:.4f} m")
        
        # PD torque control parameters (matches rtde_interpolation_controller.py)
        torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        torque_kp = torque_max / np.array([1, 1, 1, 1, 1, 1])  # Lower stiffness for smoother transfer
        torque_kd = torque_max / (np.pi * 0.5)
        
        print(f"[INFO]: PD gains - Kp: {torque_kp}")
        print(f"[INFO]: PD gains - Kd: {torque_kd}")
        print(f"[INFO]: Control frequency: {control_frequency}Hz (controller)")
        print(f"[INFO]: Action frequency: {action_frequency}Hz (command updates)")
        print(f"[INFO]: Logging frequency: {action_frequency}Hz (matches actions)")
        print(f"[INFO]: Segment duration: {segment_duration}s")
        print(f"[INFO]: Waypoints per segment: {n_waypoints}")
        
        print(f"\n[INFO]: Workspace bounds:")
        print(f"        X: [{WORKSPACE_BOUNDS['x_min']:.3f}, {WORKSPACE_BOUNDS['x_max']:.3f}] m")
        print(f"        Y: [{WORKSPACE_BOUNDS['y_min']:.3f}, {WORKSPACE_BOUNDS['y_max']:.3f}] m")
        print(f"        Z: [{WORKSPACE_BOUNDS['z_min']:.3f}, {WORKSPACE_BOUNDS['z_max']:.3f}] m")
        print(f"        Max rotation: ±{max_rot_angle}°")
        
        # Move to home and get reference orientation
        print(f"\n[INFO]: Moving to home position...")
        ok = rtde_c.moveJ(j_home.tolist(), move_speed, move_accel)
        if not ok:
            raise RuntimeError("moveJ to home failed")
        time.sleep(1.0)
        
        home_pose = rtde_r.getActualTCPPose()
        home_rot = R.from_rotvec(home_pose[3:])
        print(f"[INFO]: Home EE position: {home_pose[:3]}")
        
        max_rot_rad = np.radians(max_rot_angle)
        n_total_segments = n_train_segments + n_test_segments
        
        # Output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "train_segments").mkdir(exist_ok=True)
        (output_path / "test_segments").mkdir(exist_ok=True)
        
        all_segments = []
        
        # Collect all segments
        for seg_idx in range(n_total_segments):
            is_train = seg_idx < n_train_segments
            seg_type = "train" if is_train else "test"
            seg_num = seg_idx if is_train else seg_idx - n_train_segments
            segment_name = f"{seg_type}_{seg_num}"
            
            print(f"\n{'='*60}")
            print(f"SEGMENT {seg_idx + 1}/{n_total_segments}: {segment_name}")
            print(f"{'='*60}")
            
            # Generate random waypoints (different seed for each segment)
            seg_seed = seed + seg_idx * 1000
            waypoint_positions, waypoint_rotations = generate_random_waypoints(
                n_waypoints, home_rot, max_rot_rad, seed=seg_seed
            )
            
            print(f"[INFO]: Generated {n_waypoints} random waypoints")
            
            # Solve IK for all waypoints
            print(f"[INFO]: Solving IK for waypoints...")
            waypoint_joints, ik_success = solve_waypoint_ik(
                rtde_c, waypoint_positions, waypoint_rotations, j_home
            )
            n_success = sum(ik_success)
            print(f"[INFO]: IK success: {n_success}/{n_waypoints}")
            
            if n_success < n_waypoints * 0.8:
                print(f"[WARNING]: Too many IK failures, regenerating waypoints...")
                continue
            
            # Collect segment
            segment = collect_segment(
                rtde_c, rtde_r, waypoint_joints, waypoint_positions, waypoint_rotations,
                segment_duration, control_frequency, action_frequency, action_decimation,
                torque_kp, torque_kd, torque_max, move_speed, move_accel, segment_name
            )
            
            # Add metadata
            segment["segment_seed"] = seg_seed
            segment["is_train"] = is_train
            
            all_segments.append(segment)
            
            # Save individual segment
            seg_dir = "train_segments" if is_train else "test_segments"
            torch.save(segment, output_path / seg_dir / f"segment_{seg_num:02d}.pt")
            
            # Return to home between segments
            print(f"[INFO]: Returning to home...")
            rtde_c.moveJ(j_home.tolist(), move_speed, move_accel)
            time.sleep(0.5)
        
        # Split into train/test
        train_segments = [s for s in all_segments if s["is_train"]]
        test_segments = [s for s in all_segments if not s["is_train"]]
        
        # Concatenate segments
        def concatenate_segments(segments):
            if not segments:
                return None
            
            concat_time = []
            concat_dof_pos = []
            concat_des_dof_pos = []
            concat_joint_vel = []
            concat_torque_cmd = []
            
            t_offset = 0.0
            for seg in segments:
                seg_time = seg["time"].numpy() + t_offset
                concat_time.append(seg_time)
                concat_dof_pos.append(seg["dof_pos"].numpy())
                concat_des_dof_pos.append(seg["des_dof_pos"].numpy())
                concat_joint_vel.append(seg["joint_vel"].numpy())
                concat_torque_cmd.append(seg["torque_cmd"].numpy())
                t_offset = seg_time[-1] + 1.0 / action_frequency
            
            return {
                "time": torch.from_numpy(np.concatenate(concat_time)).float(),
                "dof_pos": torch.from_numpy(np.concatenate(concat_dof_pos)).float(),
                "des_dof_pos": torch.from_numpy(np.concatenate(concat_des_dof_pos)).float(),
                "joint_pos": torch.from_numpy(np.concatenate(concat_dof_pos)).float(),
                "joint_vel": torch.from_numpy(np.concatenate(concat_joint_vel)).float(),
                "action": torch.from_numpy(np.concatenate(concat_des_dof_pos)).float(),
                "torque_cmd": torch.from_numpy(np.concatenate(concat_torque_cmd)).float(),
                "n_segments": len(segments),
                "segment_lengths": [seg["n_samples"] for seg in segments],
            }
        
        # Save concatenated data
        if train_segments:
            train_concat = concatenate_segments(train_segments)
            train_concat["torque_kp"] = torch.from_numpy(torque_kp).float()
            train_concat["torque_kd"] = torch.from_numpy(torque_kd).float()
            train_concat["torque_max"] = torch.from_numpy(torque_max).float()
            train_concat["log_frequency"] = action_frequency  # Logging at action frequency
            train_concat["control_frequency"] = control_frequency  # Controller frequency
            torch.save(train_concat, output_path / "train.pt")
        
        if test_segments:
            test_concat = concatenate_segments(test_segments)
            test_concat["torque_kp"] = torch.from_numpy(torque_kp).float()
            test_concat["torque_kd"] = torch.from_numpy(torque_kd).float()
            test_concat["torque_max"] = torch.from_numpy(torque_max).float()
            test_concat["log_frequency"] = action_frequency  # Logging at action frequency
            test_concat["control_frequency"] = control_frequency  # Controller frequency
            torch.save(test_concat, output_path / "test.pt")
        
        # Save metadata
        metadata = {
            "log_frequency": action_frequency,  # Logging at action frequency
            "control_frequency": control_frequency,  # Controller frequency
            "action_frequency": action_frequency,  # Action update frequency
            "segment_duration": segment_duration,
            "n_waypoints_per_segment": n_waypoints,
            "torque_kp": torque_kp.tolist(),
            "torque_kd": torque_kd.tolist(),
            "torque_max": torque_max.tolist(),
            "max_rot_angle_deg": max_rot_angle,
            "workspace_bounds": WORKSPACE_BOUNDS,
            "random_seed": seed,
            "home_joints_deg": j_home_deg.tolist(),
            "n_train_segments": len(train_segments),
            "n_test_segments": len(test_segments),
            "total_train_samples": sum(s["n_samples"] for s in train_segments) if train_segments else 0,
            "total_test_samples": sum(s["n_samples"] for s in test_segments) if test_segments else 0,
        }
        torch.save(metadata, output_path / "metadata.pt")
        
        print(f"\n{'='*60}")
        print(f"DATA COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Output directory: {output_path}")
        print(f"\nTraining data:")
        print(f"  - {len(train_segments)} segments, {metadata['total_train_samples']} total samples")
        print(f"  - train.pt: Concatenated data for CMA-ES")
        print(f"\nTest data:")
        print(f"  - {len(test_segments)} segments, {metadata['total_test_samples']} total samples")
        print(f"  - test.pt: Concatenated data for validation")
        print(f"{'='*60}")

    finally:
        # Clean shutdown
        try:
            current_joints = rtde_r.getActualQ()
            rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
            rtde_c.servoStop()
        except:
            pass
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
