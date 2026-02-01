import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer)
from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


# ============================================================================
# UR5e Kinematics (DH Parameters)
# ============================================================================

# UR5e DH parameters (meters, radians)
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
    """Compute forward kinematics for UR5e."""
    T = np.eye(4)
    for i in range(6):
        Ti = dh_transform(joint_angles[i], UR5E_DH['d'][i], 
                          UR5E_DH['a'][i], UR5E_DH['alpha'][i])
        T = T @ Ti
    if tcp_offset is not None:
        T_tcp = pose_to_matrix(tcp_offset)
        T = T @ T_tcp
    return T


def get_all_transforms(joint_angles):
    """Get transformation matrices from base to each joint frame."""
    transforms = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        Ti = dh_transform(joint_angles[i], UR5E_DH['d'][i],
                          UR5E_DH['a'][i], UR5E_DH['alpha'][i])
        T = T @ Ti
        transforms.append(T.copy())
    return transforms


def compute_jacobian(joint_angles, tcp_offset=None):
    """Compute geometric Jacobian for UR5e."""
    transforms = get_all_transforms(joint_angles)
    T_ee = transforms[-1]
    if tcp_offset is not None:
        T_tcp = pose_to_matrix(tcp_offset)
        T_ee = T_ee @ T_tcp
    p_ee = T_ee[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
    return J


def pose_to_matrix(pose):
    """Convert pose [x, y, z, rx, ry, rz] (axis-angle) to 4x4 matrix."""
    pos = pose[:3]
    axis_angle = pose[3:6]
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        R = np.eye(3)
    else:
        axis = axis_angle / angle
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def matrix_to_pose(T):
    """Convert 4x4 matrix to pose [x, y, z, rx, ry, rz] (axis-angle)."""
    pos = T[:3, 3]
    R = T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        axis_angle = np.zeros(3)
    elif np.abs(angle - np.pi) < 1e-10:
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1))
        axis = np.real(eigvecs[:, idx])
        axis_angle = axis * angle
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
        axis_angle = axis * angle
    return np.concatenate([pos, axis_angle])


def axis_angle_to_quat(axis_angle):
    """Convert axis-angle to quaternion [w, x, y, z]."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis_angle / angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_to_axis_angle(quat):
    """Convert quaternion [w, x, y, z] to axis-angle."""
    w, x, y, z = quat
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    if angle < 1e-10:
        return np.zeros(3)
    s = np.sin(angle / 2)
    if s < 1e-10:
        return np.zeros(3)
    axis = np.array([x, y, z]) / s
    return axis * angle


def apply_delta_pose(ee_pos, ee_quat, delta_pose):
    """Apply delta pose to current EE pose. Returns (pos_des, quat_des)."""
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


def compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des):
    """Compute pose error (position + axis-angle rotation error)."""
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
    rot_error = quat_to_axis_angle(q_error)
    return pos_error, rot_error


def differential_ik_dls(delta_pose, jacobian, joint_pos, lambda_val=0.1):
    """Compute joint target using damped least squares IK (matching simulation)."""
    J = jacobian
    J_T = J.T
    lambda_matrix = (lambda_val ** 2) * np.eye(6)
    delta_joint_pos = J_T @ np.linalg.inv(J @ J_T + lambda_matrix) @ delta_pose
    return joint_pos + delta_joint_pos


class OneShotDifferentialIK:
    """One-shot differential IK controller matching simulation behavior."""
    
    def __init__(self, scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), lambda_val=0.1, tcp_offset=None):
        self.scale = np.array(scale)
        self.lambda_val = lambda_val
        self.tcp_offset = np.array(tcp_offset) if tcp_offset is not None else None
        
    def compute(self, delta_command, current_joint_pos):
        """Compute joint position target from Cartesian delta command."""
        scaled_delta = delta_command * self.scale
        T_ee = forward_kinematics(current_joint_pos, self.tcp_offset)
        ee_pos = T_ee[:3, 3]
        ee_axis_angle = matrix_to_pose(T_ee)[3:6]
        ee_quat = axis_angle_to_quat(ee_axis_angle)
        ee_pos_des, ee_quat_des = apply_delta_pose(ee_pos, ee_quat, scaled_delta)
        pos_error, rot_error = compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des)
        pose_error = np.concatenate([pos_error, rot_error])
        jacobian = compute_jacobian(current_joint_pos, self.tcp_offset)
        target_joint_pos = differential_ik_dls(pose_error, jacobian, current_joint_pos, self.lambda_val)
        return target_joint_pos
    
    def get_current_pose(self, current_joint_pos):
        """Get current EE pose [x, y, z, rx, ry, rz]."""
        T_ee = forward_kinematics(current_joint_pos, self.tcp_offset)
        return matrix_to_pose(T_ee)


class Command(enum.Enum):
    STOP = 0
    JointTorqueControl = 1  # Joint Torque PD Control
    CartesianIKControl = 2  # Cartesian delta -> IK -> Joint Torque PD


class RTDEInterpolationController(mp.Process):
    """
    Joint torque PD control for UR robot with optional Cartesian IK.
    This controller runs in a separate process to ensure predictable latency.
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 gripper_port=63352,
                 frequency=500,
                 launch_timeout=3,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128,
                 # IK parameters (matching simulation)
                 ik_scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 ik_lambda=0.1,
                 tcp_offset=None,  # [x, y, z, rx, ry, rz] TCP offset from flange
                 ):
        """
        Args:
            frequency: Control frequency in Hz (500Hz for UR torque control)
            joints_init: Initial joint positions in radians (6D array)
            joints_init_speed: Speed for initial joint movement (rad/s)
            soft_real_time: Enable round-robin scheduling and real-time priority
            verbose: Print debug messages
            ik_scale: Scale factors for Cartesian IK commands [x, y, z, rx, ry, rz]
            ik_lambda: DLS damping factor (default 0.1, matching sim)
            tcp_offset: TCP offset from flange [x, y, z, rx, ry, rz]
        """
        # verify
        assert 0 < frequency <= 500
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDETorqueController")
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        
        # IK parameters
        self.ik_scale = np.array(ik_scale)
        self.ik_lambda = ik_lambda
        self.tcp_offset = np.array(tcp_offset) if tcp_offset is not None else None
        
        # PD torque control parameters (same as collect_chirp_data.py)
        self.torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)
        self.torque_kp = self.torque_max / np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
        self.torque_kd = self.torque_max / (np.pi * 0.5)
        
        # build input queue (supports both joint and Cartesian commands)
        example = {
            'cmd': Command.JointTorqueControl.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
            'cartesian_delta': np.zeros((6,), dtype=np.float64),  # For Cartesian IK control
            'close_gripper': np.zeros((1,), dtype=np.bool_),
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualQ',
                'ActualQd',
                'ActualTCPPose',  # EE pose from robot's FK
            ]
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDETorqueController] Controller process "
                  f"spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': np.array([Command.STOP.value], dtype=np.int32),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'cartesian_delta': np.zeros((6,), dtype=np.float64),
            'close_gripper': np.zeros((1,), dtype=np.bool_),
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def joint_torque_control(self, target_joints, close_gripper):
        """
        Send joint torque PD control command to the robot.
        
        Args:
            target_joints: Array of 6 target joint positions in radians
            close_gripper: Boolean for gripper state
        """
        assert self.is_alive()
        target_joints = np.array(target_joints)
        assert target_joints.shape == (6,)

        message = {
            'cmd': np.array([Command.JointTorqueControl.value], dtype=np.int32),
            'target_joints': target_joints.astype(np.float64),
            'cartesian_delta': np.zeros((6,), dtype=np.float64),
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
        }
        self.input_queue.put(message)
    
    def cartesian_ik_control(self, cartesian_delta, close_gripper):
        """
        Send Cartesian delta command with one-shot IK (matching simulation).
        
        Args:
            cartesian_delta: Array of 6 Cartesian deltas [dx, dy, dz, drx, dry, drz]
                           (position in meters, rotation in axis-angle radians)
            close_gripper: Boolean for gripper state
        """
        assert self.is_alive()
        cartesian_delta = np.array(cartesian_delta)
        assert cartesian_delta.shape == (6,)

        message = {
            'cmd': np.array([Command.CartesianIKControl.value], dtype=np.int32),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'cartesian_delta': cartesian_delta.astype(np.float64),
            'close_gripper': np.array([close_gripper], dtype=np.bool_),
        }
        self.input_queue.put(message)

    def reset_to_initial_position(self, duration=3.0):
        """
        Reset robot to initial joint position using moveJ.
        
        Args:
            duration: Time to reach initial position (seconds)
        """
        if self.joints_init is not None:
            print(f"Resetting robot to initial position: {self.joints_init}")
            # Note: This requires direct access to rtde_c, so it's handled in run()
            # For now, just set target to initial joints
            self.joint_torque_control(
                target_joints=self.joints_init,
                close_gripper=False
            )
            time.sleep(duration + 0.5)
        else:
            print("Warning: No initial joint positions defined for reset")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def compute_pd_torque(self, target_joints: np.ndarray, 
                          curr_joints: np.ndarray, 
                          curr_vel: np.ndarray) -> np.ndarray:
        """
        Compute PD torque command from joint position error.
        Matches DCMotor behavior: no position error clipping, only torque clipping.
        Same implementation as collect_chirp_data.py
        """
        # Convert to numpy arrays if needed
        target_joints = np.array(target_joints, dtype=np.float64)
        curr_joints = np.array(curr_joints, dtype=np.float64)
        curr_vel = np.array(curr_vel, dtype=np.float64)
        
        # Position error
        q_err = target_joints - curr_joints
        
        # PD control: torque = Kp * pos_error - Kd * vel
        torque_d = -self.torque_kd * curr_vel
        torque_target = self.torque_kp * q_err + torque_d
        
        # Clamp total torque to max limits (only clipping)
        torque_target = np.clip(torque_target, -self.torque_max, self.torque_max)
        
        return torque_target.astype(float)

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start gripper
        gripper = RobotiqGripper()
        gripper.connect(self.robot_ip, self.gripper_port)
        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip, frequency=self.frequency,
                                      flags=RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip, frequency=self.frequency)

        try:
            if self.verbose:
                print(f"[RTDETorqueController] Connect to robot: "
                      f"{robot_ip}")

            # init joints
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init.tolist(),
                                    self.joints_init_speed, 1.4)

            gripper.activate()

            # Initialize IK controller
            ik_controller = OneShotDifferentialIK(
                scale=self.ik_scale,
                lambda_val=self.ik_lambda,
                tcp_offset=self.tcp_offset
            )

            # main loop
            curr_joints = rtde_r.getActualQ()
            current_target_joints = np.array(curr_joints, dtype=np.float64)
            current_gripper_close = False
            current_gripper_state = 'open'

            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                curr_joints = np.array(rtde_r.getActualQ(), dtype=np.float64)
                curr_vel = np.array(rtde_r.getActualQd(), dtype=np.float64)
                
                # Compute PD torque command
                torque_cmd = self.compute_pd_torque(
                    current_target_joints, 
                    curr_joints, 
                    curr_vel
                )
                
                # Send torque command
                ok = rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)
                if not ok:
                    if self.verbose:
                        print("[RTDETorqueController] directTorque failed")

                # update gripper state
                if (current_gripper_close and
                        current_gripper_state == 'open'):
                    gripper.move(gripper.get_closed_position(), 128, 128)
                    current_gripper_state = 'closed'
                elif (not current_gripper_close and
                      current_gripper_state == 'closed'):
                    gripper.move(gripper.get_open_position(), 128, 128)
                    current_gripper_state = 'open'

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['robot_receive_timestamp'] = time.time()
                
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                    commands = None

                # execute commands
                if commands is not None:
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd'][0] if isinstance(command['cmd'], np.ndarray) else command['cmd']

                        if cmd == Command.STOP.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.JointTorqueControl.value:
                            # Update target joint positions for torque control
                            current_target_joints = np.array(command['target_joints'], dtype=np.float64)
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            if self.verbose:
                                print("[RTDETorqueController] New torque control "
                                      f"target:{current_target_joints}")
                        elif cmd == Command.CartesianIKControl.value:
                            # Compute joint target from Cartesian delta using one-shot IK
                            cartesian_delta = np.array(command['cartesian_delta'], dtype=np.float64)
                            current_target_joints = ik_controller.compute(cartesian_delta, curr_joints)
                            current_gripper_close = command['close_gripper'][0] if isinstance(command['close_gripper'], np.ndarray) else command['close_gripper']
                            if self.verbose:
                                print(f"[RTDETorqueController] Cartesian IK: delta={cartesian_delta[:3]}, "
                                      f"joint_target={current_target_joints}")
                        else:
                            keep_running = False
                            break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    freq = 1/(time.perf_counter() - t_start)
                    print(f"[RTDETorqueController] Actual frequency "
                          f"{freq}")

        finally:
            # mandatory cleanup
            try:
                # Send zero torque to stop
                zero_torque = np.zeros(6)
                rtde_c.directTorque(zero_torque.tolist(), friction_comp=False)
                time.sleep(0.1)
                
                # Hold current position briefly to prevent drift
                current_joints = rtde_r.getActualQ()
                rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
                
                # decelerate
                rtde_c.servoStop()
            except Exception as e:
                if self.verbose:
                    print(f"[RTDETorqueController] Cleanup error: {e}")

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDETorqueController] Disconnected from "
                      f"robot: {robot_ip}")
