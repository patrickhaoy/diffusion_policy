"""
UR5e Calibrated Kinematics and Operational Space Controller.

Shared module for sim2real-aligned forward kinematics, Jacobian computation,
and OSC torque control. All computations use calibrated URDF parameters and
output in the REP-103 base_link frame (matching Isaac Lab simulation).

The 180deg Z rotation (R_180Z) converts from the UR controller's native
base_link_inertia frame to the REP-103 base_link frame used in simulation.
"""

import numpy as np


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

# 180deg rotation around Z-axis to convert from UR controller's base_link_inertia frame
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


# ============================================================================
# Rotation / Quaternion Utilities
# ============================================================================

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


# ============================================================================
# Forward Kinematics and Jacobian
# ============================================================================

def forward_kinematics_calibrated(joint_angles, apply_base_rotation=True):
    """
    Compute forward kinematics using calibrated URDF parameters.
    Computes to wrist_3_link frame (matching simulation).
    
    Args:
        joint_angles: 6 joint angles in radians
        apply_base_rotation: If True, apply 180deg Z rotation to convert from
            UR controller's base_link_inertia frame to REP-103 base_link frame.
            Default True for sim2real alignment.
        
    Returns:
        T: 4x4 homogeneous transformation matrix (base to wrist_3_link)
        transforms: List of transforms to each joint frame
    """
    T = np.eye(4)
    transforms = [T.copy()]
    
    for i in range(6):
        xyz = CALIBRATED_JOINTS[i]['xyz']
        rpy = CALIBRATED_JOINTS[i]['rpy']
        
        R_fixed = rpy_to_matrix(rpy)
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz
        
        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint = np.eye(4)
        T_joint[:3, :3] = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]
        ])
        
        T = T @ T_fixed @ T_joint
        transforms.append(T.copy())
    
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
        apply_base_rotation: If True, apply 180deg Z rotation to convert from
            UR controller's base_link_inertia frame to REP-103 base_link frame.
            Default True for sim2real alignment.
        
    Returns:
        J: 6x6 Jacobian matrix [linear; angular]
    """
    T_ee, _ = forward_kinematics_calibrated(joint_angles, apply_base_rotation=False)
    p_ee = T_ee[:3, 3]
    
    J = np.zeros((6, 6))
    T = np.eye(4)
    
    for i in range(6):
        xyz = CALIBRATED_JOINTS[i]['xyz']
        rpy = CALIBRATED_JOINTS[i]['rpy']
        R_fixed = rpy_to_matrix(rpy)
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz
        
        T_joint_frame = T @ T_fixed
        z_i = T_joint_frame[:3, 2]
        p_i = T_joint_frame[:3, 3]
        
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
        
        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint_rot = np.eye(4)
        T_joint_rot[:3, :3] = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1]
        ])
        T = T_joint_frame @ T_joint_rot
    
    if apply_base_rotation:
        J[:3, :] = R_180Z @ J[:3, :]
        J[3:, :] = R_180Z @ J[3:, :]
    
    return J


def get_ee_pose(joint_angles):
    """Get current EE pose as position and quaternion.
    Computes to wrist_3_link frame (matching simulation).
    
    Args:
        joint_angles: 6 joint angles in radians
    Returns:
        pos: [x, y, z] in REP-103 base frame
        quat: [w, x, y, z] quaternion
    """
    T, _ = forward_kinematics_calibrated(joint_angles)
    pos = T[:3, 3]
    quat = matrix_to_quat(T[:3, :3])
    return pos, quat


# ============================================================================
# Pose Utilities
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
    pos_error = ee_pos_des - ee_pos
    
    q_curr_inv = np.array([ee_quat[0], -ee_quat[1], -ee_quat[2], -ee_quat[3]])
    
    w1, x1, y1, z1 = ee_quat_des
    w2, x2, y2, z2 = q_curr_inv
    
    q_error = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    q_error = q_error / (np.linalg.norm(q_error) + 1e-10)
    
    if q_error[0] < 0:
        q_error = -q_error
    
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
    ee_pos_des = ee_pos + delta_pose[:3]
    
    delta_quat = axis_angle_to_quat(delta_pose[3:6])
    
    w1, x1, y1, z1 = delta_quat
    w2, x2, y2, z2 = ee_quat
    
    ee_quat_des = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    ee_quat_des = ee_quat_des / (np.linalg.norm(ee_quat_des) + 1e-10)
    
    return ee_pos_des, ee_quat_des


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
    vel_lin = (ee_pos_curr - ee_pos_prev) / dt
    
    q_prev_inv = np.array([ee_quat_prev[0], -ee_quat_prev[1], -ee_quat_prev[2], -ee_quat_prev[3]])
    
    w1, x1, y1, z1 = ee_quat_curr
    w2, x2, y2, z2 = q_prev_inv
    
    q_delta = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    if q_delta[0] < 0:
        q_delta = -q_delta
    
    axis_angle_delta = quat_to_axis_angle(q_delta)
    vel_ang = axis_angle_delta / dt
    
    return np.concatenate([vel_lin, vel_ang])


# ============================================================================
# Dynamics (Mass Matrix)
# ============================================================================

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
    _, transforms = forward_kinematics_calibrated(joint_angles, apply_base_rotation=False)
    
    M = np.zeros((6, 6))
    
    for link_idx in range(6):
        link_info = LINK_INERTIAS[link_idx]
        m = link_info['mass']
        com_local = link_info['com']
        I_local = link_info['I']
        
        I_tensor = np.array([
            [I_local[0], I_local[3], I_local[4]],
            [I_local[3], I_local[1], I_local[5]],
            [I_local[4], I_local[5], I_local[2]]
        ])
        
        T_link = transforms[link_idx + 1]
        R_link = T_link[:3, :3]
        p_link = T_link[:3, 3]
        
        p_com = p_link + R_link @ com_local
        I_world = R_link @ I_tensor @ R_link.T
        
        T = np.eye(4)
        for j in range(link_idx + 1):
            xyz = CALIBRATED_JOINTS[j]['xyz']
            rpy = CALIBRATED_JOINTS[j]['rpy']
            R_fixed = rpy_to_matrix(rpy)
            T_fixed = np.eye(4)
            T_fixed[:3, :3] = R_fixed
            T_fixed[:3, 3] = xyz
            
            T_joint_frame = T @ T_fixed
            z_j = T_joint_frame[:3, 2]
            p_j = T_joint_frame[:3, 3]
            
            J_v_j = np.cross(z_j, p_com - p_j)
            J_w_j = z_j
            
            for k in range(j + 1):
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
                
                M[j, k] += m * np.dot(J_v_j, J_v_k) + np.dot(J_w_j, I_world @ J_w_k)
                if j != k:
                    M[k, j] = M[j, k]
            
            theta = joint_angles[j]
            ct, st = np.cos(theta), np.sin(theta)
            T_joint_rot = np.eye(4)
            T_joint_rot[:3, :3] = np.array([
                [ct, -st, 0],
                [st, ct, 0],
                [0, 0, 1]
            ])
            T = T_joint_frame @ T_joint_rot
    
    M += np.eye(6) * 1e-6
    
    return M


def compute_task_space_mass_matrix(jacobian, mass_matrix, partial=False):
    """
    Compute the task-space (operational space) mass matrix.
    
    Args:
        jacobian: 6x6 Jacobian matrix
        mass_matrix: 6x6 joint-space mass matrix
        partial: If True, compute block-diagonal Lambda (no pos-rot coupling).
        
    Returns:
        Lambda: 6x6 task-space mass matrix
    """
    M_inv = np.linalg.inv(mass_matrix)
    
    if partial:
        Lambda = np.zeros((6, 6))
        
        J_pos = jacobian[:3, :]
        J_rot = jacobian[3:, :]
        
        Lambda_pos_inv = J_pos @ M_inv @ J_pos.T + np.eye(3) * 1e-6
        Lambda_rot_inv = J_rot @ M_inv @ J_rot.T + np.eye(3) * 1e-6
        
        Lambda[:3, :3] = np.linalg.inv(Lambda_pos_inv)
        Lambda[3:, 3:] = np.linalg.inv(Lambda_rot_inv)
        return Lambda
    else:
        Lambda_inv = jacobian @ M_inv @ jacobian.T
        Lambda_inv += np.eye(6) * 1e-6
        return np.linalg.inv(Lambda_inv)


# ============================================================================
# Operational Space Controller
# ============================================================================

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
        self.Kp = np.diag(motion_stiffness)
        
        kp_sqrt = np.sqrt(np.array(motion_stiffness))
        kd_diag = 2 * kp_sqrt * np.array(motion_damping_ratio)
        self.Kd = np.diag(kd_diag)
        
        if torque_max is None:
            torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
        self.torque_max = np.array(torque_max)
        
        self.ee_pos_des = None
        self.ee_quat_des = None
        
    def set_command(self, delta_command, ee_pos_curr, ee_quat_curr):
        """Set the target pose from a delta command (meters/radians, no scaling)."""
        self.ee_pos_des, self.ee_quat_des = apply_delta_pose(
            ee_pos_curr, ee_quat_curr, delta_command
        )
    
    def set_target(self, ee_pos, ee_quat):
        """Set target EE pose directly (matches sim's set_target)."""
        self.ee_pos_des = ee_pos.copy()
        self.ee_quat_des = ee_quat.copy()
    
    def apply_delta(self, delta_pos, delta_rot):
        """Apply delta to current target (accumulates on stored target)."""
        self.ee_pos_des = self.ee_pos_des + delta_pos
        
        delta_quat = axis_angle_to_quat(delta_rot)
        
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
        
        Returns:
            joint_torques: 6D joint torque command
        """
        if self.ee_pos_des is None:
            return np.zeros(6)
        
        pose_error = compute_pose_error(
            ee_pos_curr, ee_quat_curr,
            self.ee_pos_des, self.ee_quat_des
        )
        
        vel_error = -ee_vel_curr
        
        task_force = self.Kp @ pose_error + self.Kd @ vel_error
        
        joint_torques = jacobian.T @ task_force
        
        joint_torques = np.clip(joint_torques, -self.torque_max, self.torque_max)
        
        return joint_torques
    
    def get_current_pose(self, joint_pos):
        """Get current EE pose [x, y, z, rx, ry, rz]."""
        pos, quat = get_ee_pose(joint_pos)
        axis_angle = quat_to_axis_angle(quat)
        return np.concatenate([pos, axis_angle])
