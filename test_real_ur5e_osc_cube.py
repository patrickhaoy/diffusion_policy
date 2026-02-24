"""
Test script for Operational Space Control (OSC) on UR5e.
Moves the robot in a cube pattern using relative Cartesian commands.

This implements OSC with:
- Pure task-space PD control (no inertial decoupling)
- Impedance control with fixed stiffness/damping
- Commands in meters and radians directly (no scaling)

Usage:
python test_real_ur5e_osc_cube.py --robot_ip <ip_of_ur5>
"""

import numpy as np
import time
import click
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.ur5e_kinematics import (
    CALIBRATED_JOINTS, PAYLOAD_MASS, PAYLOAD_COG, LINK_INERTIAS,
    R_180Z, T_180Z,
    rpy_to_matrix, matrix_to_quat, quat_to_axis_angle, axis_angle_to_quat,
    forward_kinematics_calibrated, compute_jacobian_calibrated,
    get_ee_pose, compute_pose_error, apply_delta_pose,
    compute_ee_velocity_finite_diff, skew,
    compute_mass_matrix, compute_task_space_mass_matrix,
    OperationalSpaceController,
)


# ============================================================================
# Cube Motion Pattern
# ============================================================================

def generate_cube_waypoints(step_size_m=0.10, rot_step_rad=0.3):
    """
    Generate relative Cartesian commands for cube motion pattern with rotations.
    Returns list of (delta_command, name) tuples.
    
    Args:
        step_size_m: Position step size in meters (default 0.10 = 100mm)
        rot_step_rad: Rotation step size in radians (default 0.3 = ~17 deg)
        
    Returns:
        List of (delta_command, description) where delta_command is [dx, dy, dz, drx, dry, drz]
        in meters and radians
    """
    s = step_size_m   # Position step in meters
    r = rot_step_rad  # Rotation step in radians
    
    waypoints = [
        # 4 combined 6-DOF moves — each excites all joints
        (np.array([s, 0, s*0.3, r, 0, r]), "+X +Z +RX +RZ"),
        (np.array([0, s, -s*0.3, -r, r, 0]), "+Y -Z -RX +RY"),
        (np.array([-s, 0, s*0.3, r, 0, -r]), "-X +Z +RX -RZ"),
        (np.array([0, -s, -s*0.3, -r, -r, 0]), "-Y -Z -RX -RY"),
    ]
    return waypoints


def generate_validation_waypoints(step_size_m=0.07, rot_step_rad=0.35):
    """
    Generate validation trajectory with different motion pattern than training.
    Smaller asymmetric steps, different axis ordering, varying hold-like pauses.
    """
    s = step_size_m
    r = rot_step_rad

    waypoints = [
        # 4 asymmetric 6-DOF moves (different combos than train)
        (np.array([0, s, s*0.5, 0, r, 0]), "+Y +Z +RY"),
        (np.array([s*0.6, -s*0.4, 0, r*0.7, 0, -r*0.5]), "+X -Y +RX -RZ"),
        (np.array([-s*0.8, 0, -s*0.4, -r, r*0.3, 0]), "-X -Z -RX +RY"),
        (np.array([0, s*0.5, s*0.3, 0, -r*0.5, r*0.8]), "+Y +Z -RY +RZ"),
    ]
    return waypoints


# ============================================================================
# Chirp (Frequency Sweep) Trajectory
# ============================================================================

def generate_chirp_trajectory(duration=30.0, dt=0.002, f0=0.1, f1=5.0,
                              pos_amp=0.05, rot_amp=0.15):
    """
    Generate a linear chirp (frequency sweep) trajectory in Cartesian space.

    Each of the 6 axes (x,y,z,rx,ry,rz) gets a sinusoidal oscillation whose
    frequency sweeps from f0 to f1 over the duration.  Phase offsets decouple
    the axes so all joints are excited simultaneously.

    Args:
        duration: Total sweep time in seconds
        dt: Timestep (1/control_freq, e.g. 0.002 for 500Hz)
        f0: Start frequency in Hz
        f1: End frequency in Hz
        pos_amp: Position amplitude in meters per axis
        rot_amp: Rotation amplitude in radians per axis

    Returns:
        offsets: (T, 6) array of [dx,dy,dz,drx,dry,drz] offsets from center pose
        t: (T,) time array
    """
    T = int(duration / dt)
    t = np.linspace(0, duration, T)

    # Linear chirp: instantaneous freq = f0 + (f1-f0)*t/duration
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)

    # Amplitude envelope: ramp up over first 2s, hold, ramp down over last 3s.
    # Ensures the robot starts and ends at the center pose smoothly.
    ramp_up_s = 2.0
    ramp_down_s = 3.0
    envelope = np.ones(T)
    ramp_up_n = int(ramp_up_s / dt)
    ramp_down_n = int(ramp_down_s / dt)
    envelope[:ramp_up_n] = np.linspace(0, 1, ramp_up_n)
    envelope[-ramp_down_n:] = np.linspace(1, 0, ramp_down_n)

    # Phase offsets so axes are decoupled (60 deg apart)
    phase_offsets = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    #   dx,dy: full pos_amp  (excites base, shoulder)
    #   dz:    1.5x pos_amp  (reach in/out forces elbow to flex)
    #   drx:   2x rot_amp    (pitch rotation drives shoulder+elbow)
    #   dry:   1x rot_amp
    #   drz:   2x rot_amp    (yaw rotation drives base joint)
    amps = [pos_amp, pos_amp, pos_amp * 1.5,
            rot_amp * 2.0, rot_amp, rot_amp * 2.0]

    offsets = np.zeros((T, 6))
    for i in range(6):
        offsets[:, i] = amps[i] * envelope * np.sin(phase + phase_offsets[i])

    return offsets, t


# ============================================================================
# Main Test Script
# ============================================================================

@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5's IP address")
@click.option('--step_size', '-s', default=0.10, type=float, 
              help="Position step size in meters (default 0.10 = 100mm)")
@click.option('--rot_step', '-rs', default=0.5, type=float,
              help="Rotation step size in radians (default 0.5 = ~29 deg)")
@click.option('--hold_time', '-ht', default=2.0, type=float, help="Time to hold each position (seconds)")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Move to initial joint configuration first")
@click.option('--joints_init_deg', '-jid', default=None, type=str, 
              help="Initial joints in degrees, comma-separated (e.g. '0,-90,90,-90,-90,0')")
@click.option('--print_state', '-ps', is_flag=True, default=False, 
              help="Print current robot state and exit")
@click.option('--verify_fk', '-vfk', is_flag=True, default=False,
              help="Verify FK against RTDE (move robot around manually in freedrive, compare poses)")
@click.option('--safe', is_flag=True, default=False,
              help="Safe mode: very low torque limits (10Nm), high damping, small steps")
@click.option('--hold_only', is_flag=True, default=False,
              help="Hold-only mode: just maintain current position with OSC (no movement)")
@click.option('--hold_duration', '-hd', default=10.0, type=float,
              help="Duration for hold_only mode in seconds")
@click.option('--output_json', '-o', default=None, type=str,
              help="Output JSON file for trajectory logging (for sim2real comparison)")
@click.option('--num_waypoints', '-nw', default=None, type=int,
              help="Limit number of waypoints (default: all)")
@click.option('--kp_pos', default=1000.0, type=float, help="Position stiffness (default 1000)")
@click.option('--kp_rot', default=50.0, type=float, help="Rotation stiffness (default 50)")
@click.option('--damping_ratio_pos', default=1.0, type=float, help="Position damping ratio (default 1)")
@click.option('--damping_ratio_rot', default=1.0, type=float, help="Rotation damping ratio (default 1)")
@click.option('--verbose', '-v', is_flag=True, default=False, help="Print detailed debug info")
@click.option('--collect_sysid', type=str, default=None,
              help="Collect dense per-step data for sys-id and save to .pt file")
@click.option('--payload_mass', '-pm', default=None, type=float,
              help="Override payload mass in kg (default: 0.98 from PAYLOAD_MASS)")
@click.option('--payload_cog', '-pc', default=None, type=str,
              help="Override payload CoG as 'x,y,z' in meters (default: '0.017,-0.007,0.058')")
@click.option('--val', is_flag=True, default=False,
              help="Use validation trajectory (different pattern & init pose) instead of training")
@click.option('--chirp', is_flag=True, default=False,
              help="Chirp mode: frequency-sweep excitation for broadband sysid")
@click.option('--chirp_duration', default=8.0, type=float,
              help="Chirp sweep duration in seconds (default 8)")
@click.option('--chirp_f0', default=0.1, type=float,
              help="Chirp start frequency Hz (default 0.1)")
@click.option('--chirp_f1', default=3.0, type=float,
              help="Chirp end frequency Hz (default 3.0)")
@click.option('--chirp_pos_amp', default=0.10, type=float,
              help="Chirp position amplitude in meters (default 0.10 = 100mm)")
@click.option('--chirp_rot_amp', default=0.25, type=float,
              help="Chirp rotation amplitude in radians (default 0.25 = ~14 deg, RZ gets 2x)")
def main(robot_ip, step_size, rot_step, hold_time, init_joints, joints_init_deg, print_state, verify_fk, safe, hold_only, hold_duration, output_json, num_waypoints, kp_pos, kp_rot, damping_ratio_pos, damping_ratio_rot, verbose, collect_sysid, payload_mass, payload_cog, val, chirp, chirp_duration, chirp_f0, chirp_f1, chirp_pos_amp, chirp_rot_amp):
    """Test Operational Space Control with cube motion pattern."""
    
    # Parse initial joint positions
    if joints_init_deg is not None:
        j_init_deg = np.array([float(x) for x in joints_init_deg.split(',')], dtype=float)
        assert len(j_init_deg) == 6, f"Expected 6 joint values, got {len(j_init_deg)}"
    elif val:
        # Offset init pose for validation (different workspace region)
        j_init_deg = np.array([10, -80, 80, -85, -80, 10], dtype=float)
    else:
        # Default initial joint positions
        j_init_deg = np.array([0, -90, 90, -90, -90, 0], dtype=float)
    j_init = j_init_deg / 180.0 * np.pi
    
    # Resolve payload parameters (CLI overrides or defaults)
    pl_mass = payload_mass if payload_mass is not None else PAYLOAD_MASS
    if payload_cog is not None:
        pl_cog = [float(x) for x in payload_cog.split(',')]
        assert len(pl_cog) == 3, f"Expected 3 CoG values, got {len(pl_cog)}"
    else:
        pl_cog = list(PAYLOAD_COG)
    
    # Control frequency (500Hz for torque control)
    control_frequency = 500
    
    # OSC parameters - choose based on mode
    if safe:
        # SAFE MODE: Very conservative parameters for first-time testing
        # - Very low torque limits (won't hurt anything)
        # - Very high damping (overdamped, slow but stable)
        # - Lower stiffness (compliant)
        # - Smaller step sizes
        motion_stiffness = (50.0, 50.0, 50.0, 1.0, 1.0, 1.0)
        motion_damping_ratio = (10.0, 10.0, 10.0, 5.0, 5.0, 5.0)  # Very overdamped
        torque_max = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])  # Very low limits
        step_size = min(step_size, 0.05)  # Cap at 50mm in safe mode
        rot_step = min(rot_step, 0.1)     # Cap at ~6 deg in safe mode
        print("\n" + "!"*60)
        print("! SAFE MODE ENABLED - Very conservative parameters")
        print("! Low torque limits, high damping, small movements")
        print("! Robot will move slowly and be very compliant")
        print("!"*60)
    else:
        # Normal mode - gains from command line (default matches simulation)
        motion_stiffness = (kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot)
        motion_damping_ratio = (damping_ratio_pos, damping_ratio_pos, damping_ratio_pos, 
                                damping_ratio_rot, damping_ratio_rot, damping_ratio_rot)
        torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    
    # If print_state flag is set, just print current robot state and exit
    if print_state:
        rtde_r = RTDEReceiveInterface(robot_ip)
        current_joints = np.array(rtde_r.getActualQ(), dtype=float)
        current_joints_deg = np.degrees(current_joints)
        current_tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)
        
        # Also compute with our calibrated FK
        pos, quat = get_ee_pose(current_joints)
        jacobian = compute_jacobian_calibrated(current_joints)
        
        print("\n" + "="*60)
        print("Current Robot State")
        print("="*60)
        print(f"Joint positions (rad): {current_joints}")
        print(f"Joint positions (deg): {current_joints_deg}")
        print(f"\nFor --joints_init_deg argument, use:")
        print(f"  --joints_init_deg '{','.join([f'{x:.2f}' for x in current_joints_deg])}'")
        print(f"\nTCP pose (from RTDE): {current_tcp}")
        print(f"  Position: [{current_tcp[0]*1000:.1f}, {current_tcp[1]*1000:.1f}, {current_tcp[2]*1000:.1f}] mm")
        print(f"\nTCP pose (calibrated FK):")
        print(f"  Position: [{pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}] mm")
        print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        
        print(f"\n" + "-"*60)
        print("Jacobian Analysis (calibrated)")
        print("-"*60)
        print("Jacobian matrix (6x6):")
        print("  Rows 0-2: linear velocity (vx, vy, vz)")
        print("  Rows 3-5: angular velocity (wx, wy, wz)")
        print("  Columns: joint 0-5")
        print("")
        for i, row_name in enumerate(['vx', 'vy', 'vz', 'wx', 'wy', 'wz']):
            print(f"  {row_name}: [{', '.join([f'{jacobian[i,j]:8.4f}' for j in range(6)])}]")
        
        # Test: what torques do we get for a pure +X force?
        print(f"\n" + "-"*60)
        print("Test: tau = J^T @ F for pure forces")
        print("-"*60)
        
        F_x = np.array([1, 0, 0, 0, 0, 0])  # Pure +X force
        F_y = np.array([0, 1, 0, 0, 0, 0])  # Pure +Y force  
        F_z = np.array([0, 0, 1, 0, 0, 0])  # Pure +Z force
        
        tau_x = jacobian.T @ F_x
        tau_y = jacobian.T @ F_y
        tau_z = jacobian.T @ F_z
        
        print(f"F = [1,0,0,0,0,0] (pure +X):")
        print(f"  tau = [{', '.join([f'{t:7.4f}' for t in tau_x])}]")
        print(f"F = [0,1,0,0,0,0] (pure +Y):")
        print(f"  tau = [{', '.join([f'{t:7.4f}' for t in tau_y])}]")
        print(f"F = [0,0,1,0,0,0] (pure +Z):")
        print(f"  tau = [{', '.join([f'{t:7.4f}' for t in tau_z])}]")
        
        # Reverse test: if we apply tau_x, what EE acceleration do we get?
        # In ideal case (no inertia), a = J @ q_ddot, and q_ddot ~ tau
        # So if J^T @ F = tau, then J @ tau should be proportional to F
        print(f"\n" + "-"*60)
        print("Reverse test: J @ tau (should be proportional to F)")
        print("-"*60)
        print(f"J @ tau_x = [{', '.join([f'{v:7.4f}' for v in jacobian @ tau_x])}]")
        print(f"J @ tau_y = [{', '.join([f'{v:7.4f}' for v in jacobian @ tau_y])}]")
        print(f"J @ tau_z = [{', '.join([f'{v:7.4f}' for v in jacobian @ tau_z])}]")
        
        print("="*60)
        rtde_r.disconnect()
        return
    
    # Verify FK mode - compare our calibrated FK with RTDE's FK while in freedrive
    if verify_fk:
        rtde_c = RTDEControlInterface(
            robot_ip, 
            control_frequency,
            RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT
        )
        rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)
        
        print("\n" + "="*60)
        print("FK Verification Mode")
        print("="*60)
        print("Robot will enter FREEDRIVE mode.")
        print("Move the robot around and compare FK values.")
        print("")
        print("NOTE: Our FK computes to wrist_3_link (matching sim).")
        print("      RTDE computes to tool0 (different frame).")
        print("      A constant offset is EXPECTED and OK.")
        print("      What matters is that the offset stays constant")
        print("      as you move the robot around.")
        print("")
        print("Press 'q' to quit.")
        print("="*60 + "\n")
        
        try:
            with KeystrokeCounter() as key_counter:
                # Put robot in freedrive
                rtde_c.freedriveMode()
                print("Freedrive enabled. Move the robot around...\n")
                
                # Store initial values for delta tracking
                init_joints = np.array(rtde_r.getActualQ(), dtype=float)
                init_our_pos, _ = get_ee_pose(init_joints)
                
                iter_idx = 0
                while True:
                    # Check for quit
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            print("\nQuitting...")
                            rtde_c.endFreedriveMode()
                            rtde_c.disconnect()
                            rtde_r.disconnect()
                            return
                    
                    # Get joint positions
                    curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                    
                    # Get RTDE's TCP pose (uses robot's internal FK - different frame!)
                    rtde_tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)
                    rtde_pos = rtde_tcp[:3]
                    
                    # Compute our FK (wrist_3_link frame, matching sim)
                    our_pos, our_quat = get_ee_pose(curr_joints)
                    
                    # Compute movement from initial position (this should match between RTDE and ours)
                    our_delta = our_pos - init_our_pos
                    
                    # Print every 100 iterations (~0.2s at 500Hz)
                    if iter_idx % 100 == 0:
                        print(f"Joints (deg): [{', '.join([f'{np.degrees(j):7.2f}' for j in curr_joints])}]")
                        print(f"  RTDE pos (tool0):      [{rtde_pos[0]*1000:8.2f}, {rtde_pos[1]*1000:8.2f}, {rtde_pos[2]*1000:8.2f}] mm")
                        print(f"  Ours pos (wrist3/sim): [{our_pos[0]*1000:8.2f}, {our_pos[1]*1000:8.2f}, {our_pos[2]*1000:8.2f}] mm")
                        print(f"  Our delta from init:   [{our_delta[0]*1000:8.2f}, {our_delta[1]*1000:8.2f}, {our_delta[2]*1000:8.2f}] mm")
                        print()
                    
                    iter_idx += 1
                    time.sleep(0.002)  # ~500Hz
                    
        except Exception as e:
            print(f"Error: {e}")
            rtde_c.endFreedriveMode()
            rtde_c.disconnect()
            rtde_r.disconnect()
            raise
    
    # Create OSC controller (pure task-space PD, no inertial decoupling)
    osc_controller = OperationalSpaceController(
        motion_stiffness=motion_stiffness,
        motion_damping_ratio=motion_damping_ratio,
        torque_max=torque_max,
    )
    
    # Compute actual Kd for display
    kp_sqrt = np.sqrt(np.array(motion_stiffness))
    kd_diag = 2 * kp_sqrt * np.array(motion_damping_ratio)
    
    # Hold-only mode - just maintain current position
    if hold_only:
        print("\n" + "="*60)
        print("Hold-Only Mode - OSC Position Maintenance Test")
        print("="*60)
        print(f"Will hold current position for {hold_duration} seconds")
        print(f"Stiffness (Kp): {motion_stiffness}")
        print(f"Damping (Kd):   [{', '.join([f'{x:.1f}' for x in kd_diag])}]")
        print(f"Torque max:     {torque_max.tolist()}")
        print("Press 'q' to quit early")
        print("="*60 + "\n")
        
        # Connect to robot
        rtde_c = RTDEControlInterface(
            robot_ip, 
            control_frequency,
            RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT
        )
        rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)
        
        # Set payload for proper gravity compensation
        rtde_c.setPayload(pl_mass, pl_cog)
        
        try:
            with KeystrokeCounter() as key_counter:
                # Move to initial position if requested
                if init_joints:
                    print("Moving to initial joint position...")
                    ok = rtde_c.moveJ(j_init.tolist(), 1.05, 1.4)
                    if not ok:
                        raise RuntimeError("moveJ to initial joints failed")
                    print("Initial position reached.")
                
                # Get initial state and set as target
                curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                ee_pos, ee_quat = get_ee_pose(curr_joints)
                
                # Set zero delta command (hold current position)
                osc_controller.set_command(np.zeros(6), ee_pos, ee_quat)
                
                print(f"Holding position at: [{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, {ee_pos[2]*1000:.1f}] mm")
                print(f"Starting OSC control...\n")
                
                start_time = time.time()
                iter_idx = 0
                
                while True:
                    t_start = rtde_c.initPeriod()
                    
                    # Check for quit
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            print("\nQuitting...")
                            raise KeyboardInterrupt
                    
                    # Check duration
                    elapsed = time.time() - start_time
                    if elapsed >= hold_duration:
                        print(f"\nHold duration ({hold_duration}s) completed.")
                        break
                    
                    # Get current state
                    curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                    
                    # Compute FK and Jacobian
                    ee_pos_curr, ee_quat_curr = get_ee_pose(curr_joints)
                    jacobian = compute_jacobian_calibrated(curr_joints)
                    
                    # EE velocity via J @ qdot (matches sim)
                    joint_vel_curr = np.array(rtde_r.getActualQd(), dtype=float)
                    ee_vel = jacobian @ joint_vel_curr
                    
                    # Compute OSC torques
                    torque_cmd = osc_controller.compute(
                        ee_pos_curr, ee_quat_curr, ee_vel, jacobian
                    )
                    
                    # Send torque command
                    rtde_c.directTorque(torque_cmd.tolist(), friction_comp=True)
                    
                    # Print status every second
                    if iter_idx % 500 == 0:
                        pose_error = compute_pose_error(ee_pos_curr, ee_quat_curr,
                                                        osc_controller.ee_pos_des,
                                                        osc_controller.ee_quat_des)
                        pos_error_norm = np.linalg.norm(pose_error[:3])
                        torque_norm = np.linalg.norm(torque_cmd)
                        print(f"[{elapsed:.1f}s] Pos: [{ee_pos_curr[0]*1000:.1f}, {ee_pos_curr[1]*1000:.1f}, {ee_pos_curr[2]*1000:.1f}] mm, "
                              f"Err: {pos_error_norm*1000:.3f} mm, |tau|: {torque_norm:.2f} Nm, tau: [{', '.join([f'{t:.1f}' for t in torque_cmd])}]")
                    
                    rtde_c.waitPeriod(t_start)
                    iter_idx += 1
                    
        except KeyboardInterrupt:
            pass
        finally:
            try:
                rtde_c.directTorque([0.0]*6, friction_comp=False)
                time.sleep(0.1)
                current_joints = rtde_r.getActualQ()
                rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
                rtde_c.servoStop()
            except Exception as e:
                print(f"Cleanup error: {e}")
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            print("Disconnected from robot.")
        return
    
    # ================================================================
    # Chirp mode — frequency-sweep excitation for broadband sysid
    # ================================================================
    if chirp:
        chirp_offsets, chirp_t = generate_chirp_trajectory(
            duration=chirp_duration, dt=1.0/control_frequency,
            f0=chirp_f0, f1=chirp_f1,
            pos_amp=chirp_pos_amp, rot_amp=chirp_rot_amp)
        chirp_steps = len(chirp_t)
        dt = 1.0 / control_frequency

        print("\n" + "="*60)
        print("Chirp Mode — Frequency Sweep Excitation")
        print("="*60)
        print(f"Duration: {chirp_duration:.1f}s  ({chirp_steps} steps at {control_frequency}Hz)")
        print(f"Frequency: {chirp_f0:.2f} → {chirp_f1:.1f} Hz")
        print(f"Amplitude: pos={chirp_pos_amp*1000:.0f}mm  rot={np.degrees(chirp_rot_amp):.1f}deg")
        print(f"Stiffness (Kp): {motion_stiffness}")
        kp_sqrt = np.sqrt(np.array(motion_stiffness))
        kd_diag = 2 * kp_sqrt * np.array(motion_damping_ratio)
        print(f"Damping (Kd):   [{', '.join([f'{x:.1f}' for x in kd_diag])}]")
        print(f"Torque max:     {torque_max.tolist()}")
        if collect_sysid:
            print(f"Sysid output:   {collect_sysid}")
        print("="*60)
        print("\nPress 'q' to abort.")
        print("="*60 + "\n")

        rtde_c = RTDEControlInterface(
            robot_ip, control_frequency,
            RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
        rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)

        rtde_c.setPayload(pl_mass, pl_cog)

        try:
            with KeystrokeCounter() as key_counter:
                if init_joints:
                    print("Moving to initial joint position...")
                    ok = rtde_c.moveJ(j_init.tolist(), 1.05, 1.4)
                    if not ok:
                        raise RuntimeError("moveJ to initial joints failed")
                    print("Initial position reached.")

                current_joints = np.array(rtde_r.getActualQ(), dtype=float)
                center_pos, center_quat = get_ee_pose(current_joints)
                print(f"Center EE: [{center_pos[0]*1000:.1f}, {center_pos[1]*1000:.1f}, {center_pos[2]*1000:.1f}] mm\n")

                sysid_joint_positions = []
                sysid_joint_torques = []
                sysid_tcp_forces = []
                sysid_initial_joint_pos = current_joints.copy()
                sysid_waypoints = []

                # Set initial target to center pose
                osc_controller.set_target(center_pos, center_quat)

                stop = False
                for step_idx in range(chirp_steps):
                    t_start = rtde_c.initPeriod()

                    press_events = key_counter.get_press_events()
                    for ks in press_events:
                        if ks == KeyCode(char='q'):
                            print("\nAborted by user.")
                            stop = True
                            break
                    if stop:
                        break

                    # Compute chirp target: center pose + chirp offset
                    offset = chirp_offsets[step_idx]
                    target_pos = center_pos + offset[:3]
                    target_quat_delta = axis_angle_to_quat(offset[3:6])
                    w1, x1, y1, z1 = target_quat_delta
                    w2, x2, y2, z2 = center_quat
                    target_quat = np.array([
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2])
                    osc_controller.set_target(target_pos, target_quat)

                    curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                    ee_pos, ee_quat = get_ee_pose(curr_joints)
                    jacobian = compute_jacobian_calibrated(curr_joints)
                    joint_vel = np.array(rtde_r.getActualQd(), dtype=float)
                    ee_vel = jacobian @ joint_vel

                    torque_cmd = osc_controller.compute(ee_pos, ee_quat, ee_vel, jacobian)
                    rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)

                    if collect_sysid is not None:
                        sysid_joint_positions.append(curr_joints.copy())
                        sysid_joint_torques.append(torque_cmd.copy())
                        sysid_tcp_forces.append(np.array(rtde_r.getActualTCPForce(), dtype=float))
                        sysid_waypoints.append({
                            "step_idx": step_idx,
                            "target_pos": target_pos.copy(),
                            "target_quat": target_quat.copy(),
                        })

                    if step_idx % (control_frequency * 2) == 0:
                        elapsed = step_idx * dt
                        inst_freq = chirp_f0 + (chirp_f1 - chirp_f0) * elapsed / chirp_duration
                        pos_err = np.linalg.norm(ee_pos - target_pos)
                        print(f"  [{elapsed:.1f}s] freq={inst_freq:.2f}Hz  "
                              f"pos_err={pos_err*1000:.1f}mm  "
                              f"|tau|={np.linalg.norm(torque_cmd):.1f}Nm  "
                              f"EE=[{ee_pos[0]*1000:.0f},{ee_pos[1]*1000:.0f},{ee_pos[2]*1000:.0f}]mm")

                    rtde_c.waitPeriod(t_start)

                print(f"\nChirp completed ({chirp_steps} steps).")

                if collect_sysid is not None and len(sysid_joint_positions) > 0:
                    import torch as _torch
                    wp_step_indices = [wp["step_idx"] for wp in sysid_waypoints]
                    wp_target_pos = [wp["target_pos"] for wp in sysid_waypoints]
                    wp_target_quat = [wp["target_quat"] for wp in sysid_waypoints]
                    sysid_data = {
                        "joint_positions": _torch.tensor(np.array(sysid_joint_positions), dtype=_torch.float32),
                        "joint_torques": _torch.tensor(np.array(sysid_joint_torques), dtype=_torch.float32),
                        "tcp_forces": _torch.tensor(np.array(sysid_tcp_forces), dtype=_torch.float32),
                        "initial_joint_pos": _torch.tensor(sysid_initial_joint_pos, dtype=_torch.float32),
                        "dt": dt,
                        "control_freq": control_frequency,
                        "osc_params": {
                            "motion_stiffness": list(motion_stiffness),
                            "motion_damping_ratio": list(motion_damping_ratio),
                            "torque_max": torque_max.tolist(),
                        },
                        "chirp_params": {
                            "duration": chirp_duration,
                            "f0": chirp_f0, "f1": chirp_f1,
                            "pos_amp": chirp_pos_amp, "rot_amp": chirp_rot_amp,
                        },
                        "num_waypoints": len(sysid_waypoints),
                        "waypoint_step_indices": _torch.tensor(wp_step_indices, dtype=_torch.long),
                        "waypoint_target_pos": _torch.tensor(np.array(wp_target_pos), dtype=_torch.float32),
                        "waypoint_target_quat": _torch.tensor(np.array(wp_target_quat), dtype=_torch.float32),
                    }
                    _torch.save(sysid_data, collect_sysid)
                    print(f"Saved chirp sysid data ({len(sysid_joint_positions)} steps, "
                          f"{len(sysid_waypoints)} waypoints) to: {collect_sysid}")

        finally:
            try:
                rtde_c.directTorque([0.0]*6, friction_comp=False)
                time.sleep(0.1)
                current_joints = rtde_r.getActualQ()
                rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
                rtde_c.servoStop()
            except Exception as e:
                print(f"Cleanup error: {e}")
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            print("Disconnected from robot.")
        return
    
    # Generate waypoints
    if val:
        waypoints = generate_validation_waypoints(step_size, rot_step)
        traj_label = "Validation"
    else:
        waypoints = generate_cube_waypoints(step_size, rot_step)
        traj_label = "Training"
    if num_waypoints is not None:
        waypoints = waypoints[:num_waypoints]
    
    print("\n" + "="*60)
    print(f"Operational Space Control Test - {traj_label} Trajectory")
    print("="*60)
    print(f"Robot IP: {robot_ip}")
    print(f"Step size: {step_size*1000:.1f} mm")
    print(f"Rot step: {np.degrees(rot_step):.1f} deg")
    print(f"Hold time: {hold_time} seconds per waypoint")
    print(f"Control freq: {control_frequency} Hz")
    print(f"\nOSC Parameters:")
    print(f"  Stiffness (Kp): {motion_stiffness}")
    print(f"  Damping ratio:  {motion_damping_ratio}")
    print(f"  Damping (Kd):   [{', '.join([f'{x:.1f}' for x in kd_diag])}]")
    print(f"  Torque max:     {torque_max.tolist()}")
    print("="*60)
    print("\nWaypoints (in meters/radians):")
    for i, (delta, name) in enumerate(waypoints):
        pos_mm = delta[:3] * 1000  # Convert to mm for display
        rot_deg = np.degrees(delta[3:6])  # Convert to deg for display
        print(f"  {i+1:2d}. {name:20s} -> pos: [{pos_mm[0]:.0f}, {pos_mm[1]:.0f}, {pos_mm[2]:.0f}] mm, "
              f"rot: [{rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}] deg")
    print("="*60)
    print("\nPress 'q' to quit, 'n' for next waypoint, 's' to skip to end")
    print("="*60 + "\n")
    
    # Connect to robot
    rtde_c = RTDEControlInterface(
        robot_ip, 
        control_frequency,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT
    )
    rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)
    
    # Set payload for proper gravity compensation (gripper + camera)
    rtde_c.setPayload(pl_mass, pl_cog)
    print(f"Payload set: {pl_mass} kg at CoG [{pl_cog[0]*1000:.1f}, {pl_cog[1]*1000:.1f}, {pl_cog[2]*1000:.1f}] mm")
    
    try:
        with KeystrokeCounter() as key_counter:
            # Move to initial position
            if init_joints:
                print("Moving to initial joint position...")
                ok = rtde_c.moveJ(j_init.tolist(), 1.05, 1.4)
                if not ok:
                    raise RuntimeError("moveJ to initial joints failed")
                print("Initial position reached.")
            
            # Get initial state
            current_joints = np.array(rtde_r.getActualQ(), dtype=float)
            
            # Print initial EE pose
            initial_pos, initial_quat = get_ee_pose(current_joints)
            initial_axis_angle = quat_to_axis_angle(initial_quat)
            print(f"\nInitial EE pose (calibrated FK):")
            print(f"  Position: [{initial_pos[0]*1000:.1f}, {initial_pos[1]*1000:.1f}, {initial_pos[2]*1000:.1f}] mm")
            print(f"  Rotation: [{np.degrees(initial_axis_angle[0]):.1f}, {np.degrees(initial_axis_angle[1]):.1f}, {np.degrees(initial_axis_angle[2]):.1f}] deg")
            
            # Main control loop
            waypoint_idx = 0
            waypoint_start_time = None
            stop = False
            iter_idx = 0
            auto_advance = True
            dt = 1.0 / control_frequency
            
            # Trajectory logging (comprehensive for debugging)
            trajectory = {
                "sim_dt": dt,
                "control_freq": control_frequency,
                "initial_joints_deg": np.degrees(current_joints).tolist(),
                "initial_ee_pos": initial_pos.tolist(),
                "osc_params": {
                    "motion_stiffness": motion_stiffness.tolist() if hasattr(motion_stiffness, 'tolist') else list(motion_stiffness),
                    "motion_damping_ratio": motion_damping_ratio.tolist() if hasattr(motion_damping_ratio, 'tolist') else list(motion_damping_ratio),
                    "torque_max": torque_max.tolist(),
                },
                "waypoints": [],
                # Per-step data
                "timestamps": [],
                "joint_positions": [],
                "joint_velocities": [],
                "ee_positions": [],
                "ee_quaternions": [],
                "ee_velocities": [],
                "target_positions": [],
                "pose_errors": [],
                "desired_accelerations": [],
                "jacobians": [],
                "joint_torques": [],
                "mass_matrices": [],
            }
            log_counter = 0
            
            # Dense sys-id data collection (every step)
            sysid_joint_positions = []
            sysid_joint_torques = []
            sysid_tcp_forces = []  # F/T sensor readings [Fx,Fy,Fz,Tx,Ty,Tz]
            sysid_initial_joint_pos = current_joints.copy()
            sysid_waypoints = []  # (step_idx, target_pos, target_quat)
            
            while not stop and waypoint_idx < len(waypoints):
                t_start = rtde_c.initPeriod()
                
                # Handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='n'):
                        # Force advance to next waypoint
                        waypoint_start_time = None
                    elif key_stroke == KeyCode(char='s'):
                        # Skip to end
                        waypoint_idx = len(waypoints)
                        break
                    elif key_stroke == KeyCode(char='a'):
                        # Toggle auto-advance
                        auto_advance = not auto_advance
                        print(f"Auto-advance: {'ON' if auto_advance else 'OFF'}")
                
                # Get current state
                curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                
                # Compute FK and Jacobian
                ee_pos, ee_quat = get_ee_pose(curr_joints)
                jacobian = compute_jacobian_calibrated(curr_joints)
                
                # EE velocity via J @ qdot (matches sim, cleaner than finite difference)
                joint_vel_curr = np.array(rtde_r.getActualQd(), dtype=float)
                ee_vel = jacobian @ joint_vel_curr
                
                # Check if we need to set new target
                if waypoint_start_time is None:
                    # New waypoint - set command
                    delta_command, waypoint_name = waypoints[waypoint_idx]
                    
                    print(f"\n--- Waypoint {waypoint_idx + 1}/{len(waypoints)}: {waypoint_name} ---")
                    
                    # Apply delta relative to current pose (matches sim / Isaac Lab)
                    osc_controller.set_command(delta_command, ee_pos, ee_quat)
                    
                    # Log absolute targets for closed-loop sys-id replay
                    if collect_sysid is not None:
                        sysid_waypoints.append({
                            "step_idx": len(sysid_joint_positions),
                            "target_pos": osc_controller.ee_pos_des.copy(),
                            "target_quat": osc_controller.ee_quat_des.copy(),
                        })
                    
                    # Log waypoint
                    if output_json:
                        trajectory["waypoints"].append({
                            "name": waypoint_name,
                            "delta": delta_command.tolist(),
                            "target_pos": osc_controller.ee_pos_des.tolist(),
                        })
                    
                    if verbose:
                        print(f"Current EE: [{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, {ee_pos[2]*1000:.1f}] mm")
                        print(f"Delta: [{delta_command[0]*1000:.1f}, {delta_command[1]*1000:.1f}, {delta_command[2]*1000:.1f}] mm")
                        print(f"Target EE: [{osc_controller.ee_pos_des[0]*1000:.1f}, {osc_controller.ee_pos_des[1]*1000:.1f}, {osc_controller.ee_pos_des[2]*1000:.1f}] mm")
                    
                    waypoint_start_time = time.time()
                
                # Compute OSC torques
                torque_cmd = osc_controller.compute(
                    ee_pos, ee_quat, ee_vel, jacobian
                )
                
                # Send torque command
                rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)
                
                # Dense sys-id logging (every step)
                if collect_sysid is not None:
                    sysid_joint_positions.append(curr_joints.copy())
                    sysid_joint_torques.append(torque_cmd.copy())
                    sysid_tcp_forces.append(np.array(rtde_r.getActualTCPForce(), dtype=float))
                
                # Check if hold time elapsed
                elapsed = time.time() - waypoint_start_time
                if auto_advance and elapsed >= hold_time:
                    # Check convergence
                    pose_error = compute_pose_error(ee_pos, ee_quat, 
                                                    osc_controller.ee_pos_des, 
                                                    osc_controller.ee_quat_des)
                    pos_error_norm = np.linalg.norm(pose_error[:3])
                    rot_error_norm = np.linalg.norm(pose_error[3:])
                    
                    if (pos_error_norm < 0.002 and rot_error_norm < 0.02) or elapsed > hold_time * 2:
                        # Print final pose and move to next waypoint
                        print(f"Reached: [{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, {ee_pos[2]*1000:.1f}] mm "
                              f"(pos_err: {pos_error_norm*1000:.2f} mm, rot_err: {np.degrees(rot_error_norm):.2f} deg)")
                        waypoint_idx += 1
                        waypoint_start_time = None
                
                # Log trajectory every 50 iterations (~100ms) - comprehensive logging
                if output_json and iter_idx % 50 == 0:
                    # Compute pose error for logging
                    pose_error_log = compute_pose_error(ee_pos, ee_quat, 
                                                        osc_controller.ee_pos_des, 
                                                        osc_controller.ee_quat_des)
                    # Compute desired acceleration for logging
                    kp_sqrt = np.sqrt(np.array(motion_stiffness))
                    kd_diag = 2 * kp_sqrt * np.array(motion_damping_ratio)
                    des_ee_acc_log = np.array(motion_stiffness) * pose_error_log + kd_diag * (-ee_vel)
                    
                    trajectory["timestamps"].append(iter_idx * dt)
                    trajectory["joint_positions"].append(curr_joints.tolist())
                    trajectory["joint_velocities"].append(joint_vel_curr.tolist())
                    trajectory["ee_positions"].append(ee_pos.tolist())
                    trajectory["ee_quaternions"].append(ee_quat.tolist())
                    trajectory["ee_velocities"].append(ee_vel.tolist())
                    trajectory["target_positions"].append(osc_controller.ee_pos_des.tolist())
                    trajectory["pose_errors"].append(pose_error_log.tolist())
                    trajectory["desired_accelerations"].append(des_ee_acc_log.tolist())
                    trajectory["jacobians"].append(jacobian.tolist())
                    trajectory["joint_torques"].append(torque_cmd.tolist())
                    # Log mass matrix (always compute for comparison)
                    M = compute_mass_matrix(curr_joints)
                    trajectory["mass_matrices"].append(M.tolist())
                
                # Status print every 500 iterations (1 second at 500Hz)
                if iter_idx % 500 == 0 and waypoint_start_time is not None:
                    pose_error = compute_pose_error(ee_pos, ee_quat, 
                                                    osc_controller.ee_pos_des, 
                                                    osc_controller.ee_quat_des)
                    pos_error_norm = np.linalg.norm(pose_error[:3])
                    torque_norm = np.linalg.norm(torque_cmd)
                    print(f"  [{elapsed:.1f}s] Pos: [{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, {ee_pos[2]*1000:.1f}] mm, "
                          f"Err: {pos_error_norm*1000:.2f} mm, |tau|: {torque_norm:.1f} Nm")
                
                rtde_c.waitPeriod(t_start)
                iter_idx += 1
            
            # Print final pose
            final_joints = np.array(rtde_r.getActualQ(), dtype=float)
            final_pos, final_quat = get_ee_pose(final_joints)
            final_axis_angle = quat_to_axis_angle(final_quat)
            print(f"\n" + "="*60)
            print(f"Test completed!")
            print(f"Final EE pose:")
            print(f"  Position: [{final_pos[0]*1000:.1f}, {final_pos[1]*1000:.1f}, {final_pos[2]*1000:.1f}] mm")
            print(f"  Rotation: [{np.degrees(final_axis_angle[0]):.1f}, {np.degrees(final_axis_angle[1]):.1f}, {np.degrees(final_axis_angle[2]):.1f}] deg")
            print(f"Distance from initial: {np.linalg.norm(final_pos - initial_pos)*1000:.2f} mm")
            print("="*60)
            
            # Save trajectory if requested
            if output_json:
                import json
                with open(output_json, 'w') as f:
                    json.dump(trajectory, f, indent=2)
                print(f"\nSaved trajectory to: {output_json}")
            
            # Save dense sys-id data
            if collect_sysid is not None and len(sysid_joint_positions) > 0:
                import torch as _torch
                # Build waypoint schedule: list of (step_idx, target_pos(3), target_quat(4))
                wp_step_indices = []
                wp_target_pos = []
                wp_target_quat = []
                for wp in sysid_waypoints:
                    wp_step_indices.append(wp["step_idx"])
                    wp_target_pos.append(wp["target_pos"])
                    wp_target_quat.append(wp["target_quat"])
                
                sysid_data = {
                    "joint_positions": _torch.tensor(np.array(sysid_joint_positions), dtype=_torch.float32),
                    "joint_torques": _torch.tensor(np.array(sysid_joint_torques), dtype=_torch.float32),
                    "tcp_forces": _torch.tensor(np.array(sysid_tcp_forces), dtype=_torch.float32),
                    "initial_joint_pos": _torch.tensor(sysid_initial_joint_pos, dtype=_torch.float32),
                    "dt": dt,
                    "control_freq": control_frequency,
                    "osc_params": {
                        "motion_stiffness": list(motion_stiffness),
                        "motion_damping_ratio": list(motion_damping_ratio),
                        "torque_max": torque_max.tolist(),
                    },
                    "num_waypoints": len(waypoints),
                    "hold_time": hold_time,
                    # Closed-loop replay data
                    "waypoint_step_indices": _torch.tensor(wp_step_indices, dtype=_torch.long),
                    "waypoint_target_pos": _torch.tensor(np.array(wp_target_pos), dtype=_torch.float32),
                    "waypoint_target_quat": _torch.tensor(np.array(wp_target_quat), dtype=_torch.float32),
                }
                _torch.save(sysid_data, collect_sysid)
                print(f"\nSaved sys-id data ({len(sysid_joint_positions)} steps, "
                      f"{len(sysid_waypoints)} waypoints) to: {collect_sysid}")
    
    finally:
        # Clean shutdown
        try:
            # Send zero torque
            rtde_c.directTorque([0.0]*6, friction_comp=False)
            time.sleep(0.1)
            
            # Hold current position
            current_joints = rtde_r.getActualQ()
            rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
            rtde_c.servoStop()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Disconnected from robot.")


if __name__ == "__main__":
    main()
