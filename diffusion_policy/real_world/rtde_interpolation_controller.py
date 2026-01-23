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


class Command(enum.Enum):
    STOP = 0
    JointTorqueControl = 1  # Joint Torque PD Control


class RTDEInterpolationController(mp.Process):
    """
    Joint torque PD control for UR robot.
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
                 ):
        """
        Args:
            frequency: Control frequency in Hz (500Hz for UR torque control)
            joints_init: Initial joint positions in radians (6D array)
            joints_init_speed: Speed for initial joint movement (rad/s)
            soft_real_time: Enable round-robin scheduling and real-time priority
            verbose: Print debug messages
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
        
        # PD torque control parameters (same as collect_chirp_data.py)
        self.torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], dtype=np.float64)
        self.torque_kp = self.torque_max / np.array([0.25, 0.25, 0.5, 1, 1, 1], dtype=np.float64)
        self.torque_kd = self.torque_max / (np.pi * 0.5)
        
        # build input queue
        example = {
            'cmd': Command.JointTorqueControl.value,
            'target_joints': np.zeros((6,), dtype=np.float64),
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
