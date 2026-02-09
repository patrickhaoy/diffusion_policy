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


# ============================================================================
# UR5e Calibrated Kinematics (from ~/ur5e_calibrated.urdf)
# ============================================================================

# Joint transforms from calibrated URDF
# Each joint is defined by (xyz, rpy) relative to parent
# Note: URDF has base_link_inertia rotated by pi from base_link (REP-103 convention)
# We work in the UR controller frame (base_link_inertia), not REP-103 frame

CALIBRATED_JOINTS = [
    # shoulder_pan_joint: base_link_inertia -> shoulder_link
    {
        'xyz': np.array([0.0, 0.0, 0.1625269213099744]),
        'rpy': np.array([0.0, 0.0, -2.9298064682709875e-08]),
    },
    # shoulder_lift_joint: shoulder_link -> upper_arm_link
    {
        'xyz': np.array([0.0001655127603476787, 0.0, 0.0]),
        'rpy': np.array([1.569956524571092, 0.0, -6.769377576441778e-10]),
    },
    # elbow_joint: upper_arm_link -> forearm_link
    {
        'xyz': np.array([-0.42543123286729523, 0.0, 0.0]),
        'rpy': np.array([3.1401294729209073, 3.1401513871885234, 3.1415926282859967]),
    },
    # wrist_1_joint: forearm_link -> wrist_1_link
    {
        'xyz': np.array([-0.3926255442838572, -0.00018118202218157404, 0.13378567426143276]),
        'rpy': np.array([0.001354269897852176, -0.0010953222406931517, 2.9544587825348556e-06]),
    },
    # wrist_2_joint: wrist_1_link -> wrist_2_link
    {
        'xyz': np.array([7.719320843298656e-05, -0.0997722260056821, 6.2471135291889e-05]),
        'rpy': np.array([1.570170189345338, 0.0, -3.311599503007332e-08]),
    },
    # wrist_3_joint: wrist_2_link -> wrist_3_link
    {
        'xyz': np.array([9.607876543233622e-05, 0.09944214059801317, 0.00012685702594143576]),
        'rpy': np.array([1.5720720129010473, 3.141592653589793, 3.141592595496277]),
    },
]

# Payload parameters (Robotiq 2F-85 gripper + RealSense camera)
# Measured with UR pendant
PAYLOAD_MASS = 0.98  # kg
PAYLOAD_COG = [0.017, -0.007, 0.058]  # meters, relative to tool flange

# 180° rotation around Z-axis to convert from UR controller's base_link_inertia frame
# to REP-103 base_link frame (which matches simulation)
R_180Z = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])
T_180Z = np.eye(4)
T_180Z[:3, :3] = R_180Z

# Link inertial parameters from URDF (for mass matrix computation)
# Format: mass, center of mass (in link frame), inertia tensor (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
# NOTE: Using bare arm inertias (no gripper payload) for simpler, more compliant behavior
LINK_INERTIAS = [
    # shoulder_link (link 1)
    {'mass': 3.7, 'com': np.array([0, 0, 0]), 
     'I': np.array([0.010267495893, 0.010267495893, 0.00666, 0, 0, 0])},
    # upper_arm_link (link 2)
    {'mass': 8.393, 'com': np.array([-0.2125, 0.0, 0.138]),
     'I': np.array([0.1338857818623325, 0.1338857818623325, 0.0151074, 0, 0, 0])},
    # forearm_link (link 3)
    {'mass': 2.275, 'com': np.array([-0.1961, 0.0, 0.007]),
     'I': np.array([0.031209355099586295, 0.031209355099586295, 0.004095, 0, 0, 0])},
    # wrist_1_link (link 4)
    {'mass': 1.219, 'com': np.array([0, 0, 0]),
     'I': np.array([0.0025598989760400002, 0.0025598989760400002, 0.0021942, 0, 0, 0])},
    # wrist_2_link (link 5)
    {'mass': 1.219, 'com': np.array([0, 0, 0]),
     'I': np.array([0.0025598989760400002, 0.0025598989760400002, 0.0021942, 0, 0, 0])},
    # wrist_3_link (link 6) - bare link only
    {'mass': 0.1879, 'com': np.array([0.0, 0.0, -0.0229]),
     'I': np.array([9.890410052167731e-05, 9.890410052167731e-05, 0.0001321171875, 0, 0, 0])},
]


def rpy_to_matrix(rpy):
    """Convert roll-pitch-yaw angles to rotation matrix."""
    roll, pitch, yaw = rpy
    
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Rotation order: Z(yaw) * Y(pitch) * X(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R


def matrix_to_quat(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


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


def compute_ee_velocity_finite_diff(ee_pos_curr, ee_quat_curr, ee_pos_prev, ee_quat_prev, dt):
    """
    Compute EE velocity using finite difference.
    
    Args:
        ee_pos_curr: Current EE position [x, y, z]
        ee_quat_curr: Current EE quaternion [w, x, y, z]
        ee_pos_prev: Previous EE position [x, y, z]
        ee_quat_prev: Previous EE quaternion [w, x, y, z]
        dt: Time step (seconds)
        
    Returns:
        ee_vel: 6D velocity [vx, vy, vz, wx, wy, wz]
    """
    # Linear velocity: simple finite difference
    vel_lin = (ee_pos_curr - ee_pos_prev) / dt
    
    # Angular velocity from quaternion difference
    # q_delta = q_curr * q_prev^-1, then convert to axis-angle and divide by dt
    q_prev_inv = np.array([ee_quat_prev[0], -ee_quat_prev[1], -ee_quat_prev[2], -ee_quat_prev[3]])
    
    # Quaternion multiplication: q_curr * q_prev_inv
    w1, x1, y1, z1 = ee_quat_curr
    w2, x2, y2, z2 = q_prev_inv
    
    q_delta = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # Ensure positive w for consistent axis-angle conversion
    if q_delta[0] < 0:
        q_delta = -q_delta
    
    # Convert to axis-angle
    axis_angle_delta = quat_to_axis_angle(q_delta)
    
    # Angular velocity = axis_angle_delta / dt
    vel_ang = axis_angle_delta / dt
    
    return np.concatenate([vel_lin, vel_ang])


def forward_kinematics_calibrated(joint_angles, apply_base_rotation=True):
    """
    Compute forward kinematics using calibrated URDF parameters.
    Computes to wrist_3_link frame (matching simulation).
    
    Args:
        joint_angles: 6 joint angles in radians
        apply_base_rotation: If True, apply 180° Z rotation to convert from
            UR controller's base_link_inertia frame to REP-103 base_link frame.
            Default True for sim2real alignment.
        
    Returns:
        T: 4x4 homogeneous transformation matrix (base to wrist_3_link)
        transforms: List of transforms to each joint frame
    """
    T = np.eye(4)
    transforms = [T.copy()]
    
    for i in range(6):
        # Get calibrated joint parameters
        xyz = CALIBRATED_JOINTS[i]['xyz']
        rpy = CALIBRATED_JOINTS[i]['rpy']
        
        # Fixed transform from URDF (parent to joint origin)
        R_fixed = rpy_to_matrix(rpy)
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz
        
        # Joint rotation (around local Z axis)
        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint = np.eye(4)
        T_joint[:3, :3] = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]
        ])
        
        # Combine: T_cumulative = T_prev @ T_fixed @ T_joint
        T = T @ T_fixed @ T_joint
        transforms.append(T.copy())
    
    # Apply 180° Z rotation to convert from base_link_inertia to base_link (REP-103)
    if apply_base_rotation:
        T = T_180Z @ T
        transforms = [T_180Z @ t for t in transforms]
    
    return T, transforms


def compute_jacobian_calibrated(joint_angles, apply_base_rotation=True):
    """
    Compute geometric Jacobian using calibrated kinematics.
    Computes to wrist_3_link frame (matching simulation).
    
    Args:
        joint_angles: 6 joint angles in radians
        apply_base_rotation: If True, apply 180° Z rotation to convert from
            UR controller's base_link_inertia frame to REP-103 base_link frame.
            Default True for sim2real alignment.
        
    Returns:
        J: 6x6 Jacobian matrix [linear; angular]
    """
    # First get the EE position (without base rotation for internal computation)
    T_ee, _ = forward_kinematics_calibrated(joint_angles, apply_base_rotation=False)
    p_ee = T_ee[:3, 3]
    
    J = np.zeros((6, 6))
    
    # Recompute transforms, keeping track of frames at each joint
    # For joint i, we need the frame AFTER the fixed transform but BEFORE the joint rotation
    T = np.eye(4)  # Base frame
    
    for i in range(6):
        # Get the fixed transform for joint i
        xyz = CALIBRATED_JOINTS[i]['xyz']
        rpy = CALIBRATED_JOINTS[i]['rpy']
        R_fixed = rpy_to_matrix(rpy)
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz
        
        # Frame at joint i (after fixed transform, before joint rotation)
        # This is where the joint axis is defined
        T_joint_frame = T @ T_fixed
        
        # Z-axis of joint i (rotation axis) in world frame
        z_i = T_joint_frame[:3, 2]
        
        # Position of joint i origin in world frame
        p_i = T_joint_frame[:3, 3]
        
        # Jacobian column for joint i
        # Linear: z_i x (p_ee - p_i)
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        # Angular: z_i
        J[3:, i] = z_i
        
        # Apply joint rotation to get transform for next joint's fixed transform
        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint_rot = np.eye(4)
        T_joint_rot[:3, :3] = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]
        ])
        T = T_joint_frame @ T_joint_rot
    
    # Apply 180° Z rotation to Jacobian to convert from base_link_inertia to base_link (REP-103)
    # The Jacobian maps joint velocities to EE velocities in the base frame
    # When we rotate the base frame, we need to rotate both linear and angular velocity components
    if apply_base_rotation:
        J[:3, :] = R_180Z @ J[:3, :]  # Rotate linear velocity part
        J[3:, :] = R_180Z @ J[3:, :]  # Rotate angular velocity part
    
    return J


def get_ee_pose(joint_angles):
    """Get current EE pose as position and quaternion.
    Computes to wrist_3_link frame (matching simulation).
    
    Args:
        joint_angles: 6 joint angles in radians
    """
    T, _ = forward_kinematics_calibrated(joint_angles)
    pos = T[:3, 3]
    quat = matrix_to_quat(T[:3, :3])
    return pos, quat


def skew(v):
    """Compute skew-symmetric matrix from 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_mass_matrix(joint_angles):
    """
    Compute the 6x6 joint-space mass/inertia matrix M(q).
    Uses the Composite Rigid Body Algorithm (CRBA).
    
    Args:
        joint_angles: 6 joint angles in radians
        
    Returns:
        M: 6x6 mass matrix
    """
    # Get transforms to each link frame (no base rotation — mass matrix is a
    # joint-space quantity so it's frame-invariant, and the inner loop below
    # rebuilds joint frames from scratch without R_180Z, so we must be consistent)
    _, transforms = forward_kinematics_calibrated(joint_angles, apply_base_rotation=False)
    
    # Initialize mass matrix
    M = np.zeros((6, 6))
    
    # For each joint i, compute its contribution to the mass matrix
    # Using the formula: M_ij = sum_k(m_k * J_vi^T @ J_vk + J_wi^T @ I_k @ J_wk)
    # where sum is over links k that are affected by both joints i and j
    
    # Compute link Jacobians (6xn for each link, but we only need up to that link's joint)
    # We'll use a simplified approach: compute the contribution of each link to M
    
    for link_idx in range(6):
        link_info = LINK_INERTIAS[link_idx]
        m = link_info['mass']
        com_local = link_info['com']
        I_local = link_info['I']  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        
        # Inertia tensor in local frame
        I_tensor = np.array([
            [I_local[0], I_local[3], I_local[4]],
            [I_local[3], I_local[1], I_local[5]],
            [I_local[4], I_local[5], I_local[2]]
        ])
        
        # Transform from base to link frame (after joint rotation)
        T_link = transforms[link_idx + 1]  # +1 because transforms[0] is base
        R_link = T_link[:3, :3]
        p_link = T_link[:3, 3]
        
        # Center of mass in world frame
        p_com = p_link + R_link @ com_local
        
        # Inertia tensor in world frame
        I_world = R_link @ I_tensor @ R_link.T
        
        # Compute Jacobian for this link's COM (only joints 0 to link_idx affect it)
        T = np.eye(4)
        for j in range(link_idx + 1):
            # Get joint j's axis and position
            xyz = CALIBRATED_JOINTS[j]['xyz']
            rpy = CALIBRATED_JOINTS[j]['rpy']
            R_fixed = rpy_to_matrix(rpy)
            T_fixed = np.eye(4)
            T_fixed[:3, :3] = R_fixed
            T_fixed[:3, 3] = xyz
            
            T_joint_frame = T @ T_fixed
            z_j = T_joint_frame[:3, 2]  # Joint axis in world frame
            p_j = T_joint_frame[:3, 3]  # Joint position in world frame
            
            # Linear Jacobian column for COM: z_j x (p_com - p_j)
            J_v_j = np.cross(z_j, p_com - p_j)
            # Angular Jacobian column: z_j
            J_w_j = z_j
            
            # Contribution to mass matrix
            for k in range(j + 1):
                # Get joint k's contribution
                T_k = np.eye(4)
                for kk in range(k + 1):
                    xyz_k = CALIBRATED_JOINTS[kk]['xyz']
                    rpy_k = CALIBRATED_JOINTS[kk]['rpy']
                    R_fixed_k = rpy_to_matrix(rpy_k)
                    T_fixed_k = np.eye(4)
                    T_fixed_k[:3, :3] = R_fixed_k
                    T_fixed_k[:3, 3] = xyz_k
                    
                    T_joint_frame_k = T_k @ T_fixed_k
                    if kk < k:
                        theta_k = joint_angles[kk]
                        ct_k, st_k = np.cos(theta_k), np.sin(theta_k)
                        T_joint_rot_k = np.eye(4)
                        T_joint_rot_k[:3, :3] = np.array([
                            [ct_k, -st_k, 0],
                            [st_k, ct_k, 0],
                            [0, 0, 1]
                        ])
                        T_k = T_joint_frame_k @ T_joint_rot_k
                    else:
                        T_k = T_joint_frame_k
                
                z_k = T_k[:3, 2]
                p_k = T_k[:3, 3]
                J_v_k = np.cross(z_k, p_com - p_k)
                J_w_k = z_k
                
                # M_jk += m * J_v_j^T @ J_v_k + J_w_j^T @ I @ J_w_k
                M[j, k] += m * np.dot(J_v_j, J_v_k) + np.dot(J_w_j, I_world @ J_w_k)
                if j != k:
                    M[k, j] = M[j, k]  # Symmetric
            
            # Update transform for next joint
            theta = joint_angles[j]
            ct, st = np.cos(theta), np.sin(theta)
            T_joint_rot = np.eye(4)
            T_joint_rot[:3, :3] = np.array([
                [ct, -st, 0],
                [st, ct, 0],
                [0, 0, 1]
            ])
            T = T_joint_frame @ T_joint_rot
    
    # Add small regularization for numerical stability
    M += np.eye(6) * 1e-6
    
    return M


def compute_task_space_mass_matrix(jacobian, mass_matrix, partial=False):
    """
    Compute the task-space (operational space) mass matrix.
    
    Args:
        jacobian: 6x6 Jacobian matrix
        mass_matrix: 6x6 joint-space mass matrix
        partial: If True, compute block-diagonal Lambda (no pos-rot coupling).
            Lambda_pos = (J_pos @ M^-1 @ J_pos^T)^-1  (3x3)
            Lambda_rot = (J_rot @ M^-1 @ J_rot^T)^-1  (3x3)
        
    Returns:
        Lambda: 6x6 task-space mass matrix
    """
    M_inv = np.linalg.inv(mass_matrix)
    
    if partial:
        # Block-diagonal: decouple position and rotation
        Lambda = np.zeros((6, 6))
        
        J_pos = jacobian[:3, :]  # 3x6
        J_rot = jacobian[3:, :]  # 3x6
        
        Lambda_pos_inv = J_pos @ M_inv @ J_pos.T + np.eye(3) * 1e-6
        Lambda_rot_inv = J_rot @ M_inv @ J_rot.T + np.eye(3) * 1e-6
        
        Lambda[:3, :3] = np.linalg.inv(Lambda_pos_inv)
        Lambda[3:, 3:] = np.linalg.inv(Lambda_rot_inv)
        return Lambda
    else:
        # Full 6x6 with coupling
        Lambda_inv = jacobian @ M_inv @ jacobian.T
        Lambda_inv += np.eye(6) * 1e-6
        return np.linalg.inv(Lambda_inv)


# ============================================================================
# Operational Space Controller
# ============================================================================

def compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des):
    """
    Compute pose error between current and desired end-effector pose.
    
    Args:
        ee_pos: Current EE position [x, y, z]
        ee_quat: Current EE quaternion [w, x, y, z]
        ee_pos_des: Desired EE position [x, y, z]
        ee_quat_des: Desired EE quaternion [w, x, y, z]
        
    Returns:
        pose_error: 6D pose error [pos_error, rot_error_axis_angle]
    """
    # Position error
    pos_error = ee_pos_des - ee_pos
    
    # Rotation error (axis-angle representation)
    # q_error = q_des * q_current^-1
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
    
    # Ensure positive w for shorter rotation path (matches Isaac Lab's axis_angle_from_quat)
    if q_error[0] < 0:
        q_error = -q_error
    
    # Convert to axis-angle
    rot_error = quat_to_axis_angle(q_error)
    
    return np.concatenate([pos_error, rot_error])


def apply_delta_pose(ee_pos, ee_quat, delta_pose):
    """
    Apply delta pose to current end-effector pose.
    
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
    
    # Rotation: multiply quaternions (delta in world frame)
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


class OperationalSpaceController:
    """
    Operational Space Controller matching simulation config.
    
    Pure task-space PD (no inertial decoupling):
        tau = J^T @ (Kp @ pose_error + Kd @ vel_error)
    
    Where Kd = 2 * sqrt(Kp) * damping_ratio
    """
    
    def __init__(self,
                 motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
                 motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 torque_max=None):
        """
        Args:
            motion_stiffness: Stiffness gains [x, y, z, rx, ry, rz]
            motion_damping_ratio: Damping ratios [x, y, z, rx, ry, rz]
            torque_max: Maximum torque per joint
        """
        
        # Stiffness (Kp) as diagonal matrix
        self.Kp = np.diag(motion_stiffness)
        
        # Damping (Kd) = 2 * sqrt(Kp) * damping_ratio
        kp_sqrt = np.sqrt(np.array(motion_stiffness))
        kd_diag = 2 * kp_sqrt * np.array(motion_damping_ratio)
        self.Kd = np.diag(kd_diag)
        
        # Torque limits
        if torque_max is None:
            torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        self.torque_max = np.array(torque_max)
        
        # Desired pose (set by set_command)
        self.ee_pos_des = None
        self.ee_quat_des = None
        
    def set_command(self, delta_command, ee_pos_curr, ee_quat_curr):
        """
        Set the target pose from a delta command.
        
        Args:
            delta_command: Delta command [dx, dy, dz, drx, dry, drz] in meters and radians
            ee_pos_curr: Current EE position
            ee_quat_curr: Current EE quaternion [w, x, y, z]
        """
        # Compute desired pose (no scaling - delta is in meters/radians directly)
        self.ee_pos_des, self.ee_quat_des = apply_delta_pose(
            ee_pos_curr, ee_quat_curr, delta_command
        )
    
    def set_target(self, ee_pos, ee_quat):
        """Set target EE pose directly (matches sim's set_target)."""
        self.ee_pos_des = ee_pos.copy()
        self.ee_quat_des = ee_quat.copy()
    
    def apply_delta(self, delta_pos, delta_rot):
        """Apply delta to current target (matches sim's apply_delta).
        
        Unlike set_command which applies delta to current pose,
        this accumulates deltas on the stored target.
        """
        self.ee_pos_des = self.ee_pos_des + delta_pos
        
        # Convert axis-angle to quaternion
        delta_quat = axis_angle_to_quat(delta_rot)
        
        # Quaternion multiply: delta * current_target (rotation in world frame)
        w1, x1, y1, z1 = delta_quat
        w2, x2, y2, z2 = self.ee_quat_des
        self.ee_quat_des = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        self.ee_quat_des /= np.linalg.norm(self.ee_quat_des)
        
    def compute(self, ee_pos_curr, ee_quat_curr, ee_vel_curr, jacobian):
        """
        Compute joint torques: tau = J^T @ (Kp @ err + Kd @ vel_err).
        
        Args:
            ee_pos_curr: Current EE position [x, y, z]
            ee_quat_curr: Current EE quaternion [w, x, y, z]
            ee_vel_curr: Current EE velocity [vx, vy, vz, wx, wy, wz]
            jacobian: 6x6 Jacobian matrix
            
        Returns:
            joint_torques: 6D joint torque command
        """
        if self.ee_pos_des is None:
            return np.zeros(6)
        
        # Pose error
        pose_error = compute_pose_error(
            ee_pos_curr, ee_quat_curr,
            self.ee_pos_des, self.ee_quat_des
        )
        
        # Velocity error (target velocity is zero)
        vel_error = -ee_vel_curr
        
        # Task-space force (spring-damper)
        task_force = self.Kp @ pose_error + self.Kd @ vel_error
        
        # Joint torques: tau = J^T @ F
        joint_torques = jacobian.T @ task_force
        
        # Clamp torques
        joint_torques = np.clip(joint_torques, -self.torque_max, self.torque_max)
        
        return joint_torques
    
    def get_current_pose(self, joint_pos):
        """Get current EE pose [x, y, z, rx, ry, rz]."""
        pos, quat = get_ee_pose(joint_pos)
        axis_angle = quat_to_axis_angle(quat)
        return np.concatenate([pos, axis_angle])


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
def main(robot_ip, step_size, rot_step, hold_time, init_joints, joints_init_deg, print_state, verify_fk, safe, hold_only, hold_duration, output_json, num_waypoints, kp_pos, kp_rot, damping_ratio_pos, damping_ratio_rot, verbose, collect_sysid, payload_mass, payload_cog, val):
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
