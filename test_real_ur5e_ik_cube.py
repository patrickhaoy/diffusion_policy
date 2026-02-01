"""
Test script for one-shot differential IK with PD torque control.
Moves the robot in a cube pattern using relative Cartesian commands.

This implements the same IK approach as the simulation:
- One-shot differential IK: compute IK once per command step, hold joint target
- Damped Least Squares (DLS) with configurable lambda
- PD torque control to track joint targets

Usage:
python test_real_ur5e_ik_cube.py --robot_ip <ip_of_ur5>
"""

import numpy as np
import time
import click
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.common.precise_sleep import precise_wait


# ============================================================================
# UR5e Kinematics (DH Parameters)
# ============================================================================

# UR5e DH parameters (meters, radians)
# From Universal Robots documentation
UR5E_DH = {
    'd': np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996]),
    'a': np.array([0, -0.425, -0.3922, 0, 0, 0]),
    'alpha': np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
}


def dh_transform(theta, d, a, alpha):
    """Compute homogeneous transformation matrix from DH parameters."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])


def forward_kinematics(joint_angles, tcp_offset=None):
    """
    Compute forward kinematics for UR5e.
    
    Args:
        joint_angles: 6 joint angles in radians
        tcp_offset: Optional TCP offset [x, y, z, rx, ry, rz] (axis-angle)
        
    Returns:
        T: 4x4 homogeneous transformation matrix (base to TCP)
    """
    T = np.eye(4)
    for i in range(6):
        Ti = dh_transform(
            joint_angles[i],
            UR5E_DH['d'][i],
            UR5E_DH['a'][i],
            UR5E_DH['alpha'][i]
        )
        T = T @ Ti
    
    if tcp_offset is not None:
        # Apply TCP offset
        tcp_pos = tcp_offset[:3]
        tcp_rot = tcp_offset[3:6]  # axis-angle
        T_tcp = pose_to_matrix(np.concatenate([tcp_pos, tcp_rot]))
        T = T @ T_tcp
    
    return T


def get_all_transforms(joint_angles):
    """Get transformation matrices from base to each joint frame."""
    transforms = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        Ti = dh_transform(
            joint_angles[i],
            UR5E_DH['d'][i],
            UR5E_DH['a'][i],
            UR5E_DH['alpha'][i]
        )
        T = T @ Ti
        transforms.append(T.copy())
    return transforms


def compute_jacobian(joint_angles, tcp_offset=None):
    """
    Compute geometric Jacobian for UR5e.
    
    Args:
        joint_angles: 6 joint angles in radians
        tcp_offset: Optional TCP offset [x, y, z, rx, ry, rz]
        
    Returns:
        J: 6x6 Jacobian matrix [linear; angular]
    """
    transforms = get_all_transforms(joint_angles)
    
    # End-effector position
    T_ee = transforms[-1]
    if tcp_offset is not None:
        tcp_pos = tcp_offset[:3]
        tcp_rot = tcp_offset[3:6]
        T_tcp = pose_to_matrix(np.concatenate([tcp_pos, tcp_rot]))
        T_ee = T_ee @ T_tcp
    
    p_ee = T_ee[:3, 3]
    
    J = np.zeros((6, 6))
    
    for i in range(6):
        # z-axis of joint i frame (rotation axis for revolute joint)
        z_i = transforms[i][:3, 2]
        # Position of joint i origin
        p_i = transforms[i][:3, 3]
        
        # Linear velocity component: z_i x (p_ee - p_i)
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        # Angular velocity component: z_i
        J[3:, i] = z_i
    
    return J


def pose_to_matrix(pose):
    """Convert pose [x, y, z, rx, ry, rz] (axis-angle) to 4x4 matrix."""
    pos = pose[:3]
    axis_angle = pose[3:6]
    
    # Axis-angle to rotation matrix
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        R = np.eye(3)
    else:
        axis = axis_angle / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def matrix_to_pose(T):
    """Convert 4x4 matrix to pose [x, y, z, rx, ry, rz] (axis-angle)."""
    pos = T[:3, 3]
    R = T[:3, :3]
    
    # Rotation matrix to axis-angle
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    
    if angle < 1e-10:
        axis_angle = np.zeros(3)
    elif np.abs(angle - np.pi) < 1e-10:
        # Handle 180 degree rotation
        # Find eigenvector corresponding to eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1))
        axis = np.real(eigvecs[:, idx])
        axis_angle = axis * angle
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        axis_angle = axis * angle
    
    return np.concatenate([pos, axis_angle])


def quat_to_axis_angle(quat):
    """Convert quaternion [w, x, y, z] to axis-angle [rx, ry, rz]."""
    w, x, y, z = quat
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    if angle < 1e-10:
        return np.zeros(3)
    
    s = np.sin(angle / 2)
    if s < 1e-10:
        return np.zeros(3)
    
    axis = np.array([x, y, z]) / s
    return axis * angle


def axis_angle_to_quat(axis_angle):
    """Convert axis-angle [rx, ry, rz] to quaternion [w, x, y, z]."""
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = axis_angle / angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])


# ============================================================================
# Differential IK (matching simulation)
# ============================================================================

def compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des):
    """
    Compute pose error between current and desired end-effector pose.
    Returns position error and axis-angle rotation error.
    
    Args:
        ee_pos: Current EE position [x, y, z]
        ee_quat: Current EE quaternion [w, x, y, z]
        ee_pos_des: Desired EE position [x, y, z]
        ee_quat_des: Desired EE quaternion [w, x, y, z]
        
    Returns:
        pos_error: Position error [dx, dy, dz]
        rot_error: Rotation error in axis-angle [drx, dry, drz]
    """
    # Position error
    pos_error = ee_pos_des - ee_pos
    
    # Rotation error (axis-angle)
    # q_error = q_des * q_current^-1
    # q^-1 = [w, -x, -y, -z] for unit quaternion
    q_curr_inv = np.array([ee_quat[0], -ee_quat[1], -ee_quat[2], -ee_quat[3]])
    
    # Quaternion multiplication: q_des * q_curr_inv
    w1, x1, y1, z1 = ee_quat_des
    w2, x2, y2, z2 = q_curr_inv
    
    q_error = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # Normalize
    q_error = q_error / (np.linalg.norm(q_error) + 1e-10)
    
    # Convert to axis-angle
    rot_error = quat_to_axis_angle(q_error)
    
    return pos_error, rot_error


def apply_delta_pose(ee_pos, ee_quat, delta_pose):
    """
    Apply delta pose to current end-effector pose.
    Matches isaaclab.utils.math.apply_delta_pose
    
    Args:
        ee_pos: Current position [x, y, z]
        ee_quat: Current quaternion [w, x, y, z]
        delta_pose: Delta pose [dx, dy, dz, drx, dry, drz] (axis-angle)
        
    Returns:
        ee_pos_des: Desired position [x, y, z]
        ee_quat_des: Desired quaternion [w, x, y, z]
    """
    # Position: simple addition
    ee_pos_des = ee_pos + delta_pose[:3]
    
    # Rotation: multiply quaternions
    # delta_quat * current_quat (apply delta in world frame)
    delta_quat = axis_angle_to_quat(delta_pose[3:6])
    
    w1, x1, y1, z1 = delta_quat
    w2, x2, y2, z2 = ee_quat
    
    ee_quat_des = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # Normalize
    ee_quat_des = ee_quat_des / (np.linalg.norm(ee_quat_des) + 1e-10)
    
    return ee_pos_des, ee_quat_des


def differential_ik_dls(delta_pose, jacobian, joint_pos, lambda_val=0.1):
    """
    Compute joint position target using damped least squares IK.
    Matches simulation: isaaclab/controllers/differential_ik.py
    
    Args:
        delta_pose: Pose error [dx, dy, dz, drx, dry, drz]
        jacobian: 6x6 geometric Jacobian
        joint_pos: Current joint positions (6,)
        lambda_val: Damping factor (default 0.1, matching sim config)
        
    Returns:
        joint_pos_target: Target joint positions (6,)
    """
    # DLS: delta_q = J^T @ (J @ J^T + lambda^2 * I)^-1 @ delta_x
    J = jacobian
    J_T = J.T
    
    lambda_matrix = (lambda_val ** 2) * np.eye(6)
    
    # Compute pseudo-inverse with damping
    delta_joint_pos = J_T @ np.linalg.inv(J @ J_T + lambda_matrix) @ delta_pose
    
    return joint_pos + delta_joint_pos


class OneShotDifferentialIK:
    """
    One-shot differential IK controller matching simulation behavior.
    
    Workflow:
    1. Receive 6-DOF Cartesian delta command [dx, dy, dz, drx, dry, drz]
    2. Apply scaling
    3. Compute desired EE pose
    4. Compute pose error
    5. Apply DLS IK to get joint position target (computed ONCE)
    6. Return joint target for PD torque control
    """
    
    def __init__(self, 
                 scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 lambda_val=0.1,
                 tcp_offset=None):
        """
        Args:
            scale: Scaling factors for [x, y, z, rx, ry, rz] commands
            lambda_val: DLS damping factor
            tcp_offset: TCP offset from flange [x, y, z, rx, ry, rz]
        """
        self.scale = np.array(scale)
        self.lambda_val = lambda_val
        self.tcp_offset = np.array(tcp_offset) if tcp_offset is not None else None
        
    def compute(self, delta_command, current_joint_pos):
        """
        Compute joint position target from Cartesian delta command.
        
        Args:
            delta_command: Raw delta command [dx, dy, dz, drx, dry, drz]
            current_joint_pos: Current joint positions (6,)
            
        Returns:
            target_joint_pos: Target joint positions (6,)
        """
        # Step 1: Apply scaling
        scaled_delta = delta_command * self.scale
        
        # Step 2: Get current EE pose
        T_ee = forward_kinematics(current_joint_pos, self.tcp_offset)
        ee_pos = T_ee[:3, 3]
        ee_axis_angle = matrix_to_pose(T_ee)[3:6]
        ee_quat = axis_angle_to_quat(ee_axis_angle)
        
        # Step 3: Compute desired EE pose
        ee_pos_des, ee_quat_des = apply_delta_pose(ee_pos, ee_quat, scaled_delta)
        
        # Step 4: Compute pose error
        pos_error, rot_error = compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des)
        pose_error = np.concatenate([pos_error, rot_error])
        
        # Step 5: Compute Jacobian
        jacobian = compute_jacobian(current_joint_pos, self.tcp_offset)
        
        # Step 6: Apply DLS IK
        target_joint_pos = differential_ik_dls(
            pose_error, jacobian, current_joint_pos, self.lambda_val
        )
        
        return target_joint_pos
    
    def get_current_pose(self, current_joint_pos):
        """Get current EE pose [x, y, z, rx, ry, rz]."""
        T_ee = forward_kinematics(current_joint_pos, self.tcp_offset)
        return matrix_to_pose(T_ee)


# ============================================================================
# PD Torque Control (matching simulation/existing controller)
# ============================================================================

def compute_pd_torque(target_joints, curr_joints, curr_vel,
                      torque_kp, torque_kd, torque_max):
    """
    Compute PD torque command from joint position error.
    Matches DCMotor behavior in simulation.
    """
    # Position error (no clipping - matching sim)
    q_err = target_joints - curr_joints
    
    # PD control: torque = Kp * pos_error - Kd * vel
    torque_d = -torque_kd * curr_vel
    torque_target = torque_kp * q_err + torque_d
    
    # Clamp total torque to max limits (only torque clipping, not error)
    torque_target = np.clip(torque_target, -torque_max, torque_max)
    
    return torque_target.astype(float)


# ============================================================================
# Cube Motion Pattern
# ============================================================================

def generate_cube_waypoints(step_size=0.05, rot_step=0.1):
    """
    Generate relative Cartesian commands for cube motion pattern with rotations.
    Returns list of (delta_command, name) tuples.
    
    Args:
        step_size: Step size in meters for position moves
        rot_step: Step size in radians for rotation moves (~5.7 degrees per 0.1 rad)
        
    Returns:
        List of (delta_command, description) where delta_command is [dx, dy, dz, drx, dry, drz]
    """
    waypoints = [
        # === Position moves (cube pattern) ===
        # Move in +X
        (np.array([step_size, 0, 0, 0, 0, 0]), "+X"),
        # Move in +Y
        (np.array([0, step_size, 0, 0, 0, 0]), "+Y"),
        # Move in -X
        (np.array([-step_size, 0, 0, 0, 0, 0]), "-X"),
        # Move in -Y
        (np.array([0, -step_size, 0, 0, 0, 0]), "-Y"),
        # Move in +Z
        (np.array([0, 0, step_size, 0, 0, 0]), "+Z"),
        # Move in -Z (back)
        (np.array([0, 0, -step_size, 0, 0, 0]), "-Z"),
        
        # === Rotation moves ===
        # Rotate around X (roll)
        (np.array([0, 0, 0, rot_step, 0, 0]), "+RX (roll)"),
        (np.array([0, 0, 0, -rot_step, 0, 0]), "-RX (roll back)"),
        # Rotate around Y (pitch)
        (np.array([0, 0, 0, 0, rot_step, 0]), "+RY (pitch)"),
        (np.array([0, 0, 0, 0, -rot_step, 0]), "-RY (pitch back)"),
        # Rotate around Z (yaw)
        (np.array([0, 0, 0, 0, 0, rot_step]), "+RZ (yaw)"),
        (np.array([0, 0, 0, 0, 0, -rot_step]), "-RZ (yaw back)"),
        
        # === Combined position + rotation ===
        (np.array([step_size, 0, 0, 0, 0, rot_step]), "+X +RZ"),
        (np.array([-step_size, 0, 0, 0, 0, -rot_step]), "-X -RZ (back)"),
    ]
    return waypoints


# ============================================================================
# Main Test Script
# ============================================================================

@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5's IP address")
@click.option('--step_size', '-s', default=0.03, type=float, help="Step size in meters for position moves")
@click.option('--rot_step', '-rs', default=0.1, type=float, help="Step size in radians for rotation moves (~5.7 deg)")
@click.option('--hold_time', '-ht', default=2.0, type=float, help="Time to hold each position (seconds)")
@click.option('--lambda_val', '-l', default=0.1, type=float, help="DLS damping factor (matching sim)")
@click.option('--tcp_offset_z', '-tcp', default=0.1345, type=float, help="TCP offset in Z (matching robotiq gripper)")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Move to initial joint configuration first")
@click.option('--joints_init_deg', '-jid', default=None, type=str, help="Initial joints in degrees, comma-separated (e.g. '0,-90,90,-90,-90,0')")
@click.option('--print_state', '-ps', is_flag=True, default=False, help="Print current robot state and exit (for getting current pose)")
@click.option('--verbose', '-v', is_flag=True, default=False, help="Print detailed debug info")
def main(robot_ip, step_size, rot_step, hold_time, lambda_val, tcp_offset_z, init_joints, joints_init_deg, print_state, verbose):
    """Test one-shot differential IK with cube motion pattern."""
    
    # Parse initial joint positions
    if joints_init_deg is not None:
        j_init_deg = np.array([float(x) for x in joints_init_deg.split(',')], dtype=float)
        assert len(j_init_deg) == 6, f"Expected 6 joint values, got {len(j_init_deg)}"
    else:
        # Default initial joint positions (same as other scripts)
        j_init_deg = np.array([0, -90, 90, -90, -90, 0], dtype=float)
    j_init = j_init_deg / 180.0 * np.pi
    
    # Control frequency (500Hz for torque control)
    control_frequency = 500
    
    # If print_state flag is set, just print current robot state and exit
    if print_state:
        rtde_r = RTDEReceiveInterface(robot_ip)
        current_joints = np.array(rtde_r.getActualQ(), dtype=float)
        current_joints_deg = np.degrees(current_joints)
        current_tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)
        
        print("\n" + "="*60)
        print("Current Robot State")
        print("="*60)
        print(f"Joint positions (rad): {current_joints}")
        print(f"Joint positions (deg): {current_joints_deg}")
        print(f"\nFor --joints_init_deg argument, use:")
        print(f"  --joints_init_deg '{','.join([f'{x:.2f}' for x in current_joints_deg])}'")
        print(f"\nTCP pose (from RTDE): {current_tcp}")
        print(f"  Position: [{current_tcp[0]*1000:.1f}, {current_tcp[1]*1000:.1f}, {current_tcp[2]*1000:.1f}] mm")
        print(f"  Rotation (axis-angle): [{np.degrees(current_tcp[3]):.1f}, {np.degrees(current_tcp[4]):.1f}, {np.degrees(current_tcp[5]):.1f}] deg")
        print("="*60)
        rtde_r.disconnect()
        return
    
    # TCP offset (from flange to fingertip - matching sim body_offset)
    tcp_offset = np.array([tcp_offset_z, 0, 0, 0, 0, 0])  # gripper points along flange X
    
    # PD torque control parameters (matching rtde_interpolation_controller.py)
    torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    torque_kp = torque_max / np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    torque_kd = torque_max / (np.pi * 0.5)
    
    # Create IK controller
    ik_controller = OneShotDifferentialIK(
        scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),  # No scaling for test (direct meters/radians)
        lambda_val=lambda_val,
        tcp_offset=tcp_offset
    )
    
    # Generate cube waypoints
    waypoints = generate_cube_waypoints(step_size, rot_step)
    
    print("\n" + "="*60)
    print("One-Shot Differential IK Test - Cube Motion")
    print("="*60)
    print(f"Robot IP: {robot_ip}")
    print(f"Position step: {step_size*1000:.1f} mm")
    print(f"Rotation step: {np.degrees(rot_step):.1f} deg ({rot_step:.2f} rad)")
    print(f"Hold time: {hold_time} seconds per waypoint")
    print(f"DLS lambda: {lambda_val}")
    print(f"TCP offset: {tcp_offset}")
    print(f"Control freq: {control_frequency} Hz")
    print(f"Torque Kp: {torque_kp}")
    print(f"Torque Kd: {torque_kd}")
    print("="*60)
    print("\nWaypoints:")
    for i, (delta, name) in enumerate(waypoints):
        pos_str = f"pos: [{delta[0]*1000:.0f}, {delta[1]*1000:.0f}, {delta[2]*1000:.0f}] mm"
        rot_str = f"rot: [{np.degrees(delta[3]):.1f}, {np.degrees(delta[4]):.1f}, {np.degrees(delta[5]):.1f}] deg"
        print(f"  {i+1:2d}. {name:20s} {pos_str}, {rot_str}")
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
            target_joints = current_joints.copy()
            
            # Print initial EE pose
            initial_pose = ik_controller.get_current_pose(current_joints)
            print(f"\nInitial EE pose:")
            print(f"  Position: [{initial_pose[0]*1000:.1f}, {initial_pose[1]*1000:.1f}, {initial_pose[2]*1000:.1f}] mm")
            print(f"  Rotation: [{np.degrees(initial_pose[3]):.1f}, {np.degrees(initial_pose[4]):.1f}, {np.degrees(initial_pose[5]):.1f}] deg")
            
            # Main control loop
            waypoint_idx = 0
            waypoint_start_time = None
            stop = False
            iter_idx = 0
            auto_advance = True
            
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
                curr_vel = np.array(rtde_r.getActualQd(), dtype=float)
                
                # Check if we need to compute new IK target
                if waypoint_start_time is None:
                    # New waypoint - compute IK once
                    delta_command, waypoint_name = waypoints[waypoint_idx]
                    
                    print(f"\n--- Waypoint {waypoint_idx + 1}/{len(waypoints)}: {waypoint_name} ---")
                    
                    # Compute joint target using one-shot IK
                    target_joints = ik_controller.compute(delta_command, curr_joints)
                    
                    if verbose:
                        current_pose = ik_controller.get_current_pose(curr_joints)
                        print(f"Current EE: [{current_pose[0]*1000:.1f}, {current_pose[1]*1000:.1f}, {current_pose[2]*1000:.1f}] mm")
                        print(f"Delta cmd:  [{delta_command[0]*1000:.1f}, {delta_command[1]*1000:.1f}, {delta_command[2]*1000:.1f}] mm")
                        print(f"Joint delta: {np.degrees(target_joints - curr_joints)} deg")
                    
                    waypoint_start_time = time.time()
                
                # Compute PD torque
                torque_cmd = compute_pd_torque(
                    target_joints, curr_joints, curr_vel,
                    torque_kp, torque_kd, torque_max
                )
                
                # Send torque command
                rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)
                
                # Check if hold time elapsed
                elapsed = time.time() - waypoint_start_time
                if auto_advance and elapsed >= hold_time:
                    # Check if we've converged
                    joint_error = np.max(np.abs(target_joints - curr_joints))
                    if joint_error < np.radians(1.0) or elapsed > hold_time * 2:
                        # Print final pose and move to next waypoint
                        current_pose = ik_controller.get_current_pose(curr_joints)
                        print(f"Reached: [{current_pose[0]*1000:.1f}, {current_pose[1]*1000:.1f}, {current_pose[2]*1000:.1f}] mm "
                              f"(error: {joint_error*1000:.2f} mrad)")
                        waypoint_idx += 1
                        waypoint_start_time = None
                
                # Status print every 500 iterations (1 second at 500Hz)
                if iter_idx % 500 == 0 and waypoint_start_time is not None:
                    joint_error = np.max(np.abs(target_joints - curr_joints))
                    current_pose = ik_controller.get_current_pose(curr_joints)
                    print(f"  [{elapsed:.1f}s] Pos: [{current_pose[0]*1000:.1f}, {current_pose[1]*1000:.1f}, {current_pose[2]*1000:.1f}] mm, "
                          f"Joint err: {np.degrees(joint_error):.2f} deg")
                
                rtde_c.waitPeriod(t_start)
                iter_idx += 1
            
            # Print final pose
            final_joints = np.array(rtde_r.getActualQ(), dtype=float)
            final_pose = ik_controller.get_current_pose(final_joints)
            print(f"\n" + "="*60)
            print(f"Test completed!")
            print(f"Final EE pose:")
            print(f"  Position: [{final_pose[0]*1000:.1f}, {final_pose[1]*1000:.1f}, {final_pose[2]*1000:.1f}] mm")
            print(f"  Rotation: [{np.degrees(final_pose[3]):.1f}, {np.degrees(final_pose[4]):.1f}, {np.degrees(final_pose[5]):.1f}] deg")
            print(f"Distance from initial: {np.linalg.norm(final_pose[:3] - initial_pose[:3])*1000:.2f} mm")
            print("="*60)
    
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
