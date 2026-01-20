"""
Usage:
python test_real_ur5e_torque_pd.py --robot_ip <ip_of_ur5> --mello_port <port>

Control robot using Mello teleop with PD torque control for sim-transferable data collection.

Recording control:
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

import numpy as np
import time
import torch
import click
import cv2
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface
from diffusion_policy.common.precise_sleep import precise_wait


def compute_pd_torque(target_joints: np.ndarray, 
                      curr_joints: np.ndarray, 
                      curr_vel: np.ndarray,
                      torque_kp: np.ndarray,
                      torque_kd: np.ndarray,
                      torque_max: np.ndarray) -> np.ndarray:
    """
    Compute PD torque command from joint position error.
    Same implementation as rtde_interpolation_controller.py TORQUE_PD mode.
    """
    # Position error with clipping (same limits as rtde_interpolation_controller.py)
    q_err = target_joints - curr_joints
    q_err = np.clip(q_err, 
                    [-np.pi / 6, -np.pi / 6, -np.pi / 6, -np.pi / 4, -np.pi / 4, -np.pi / 4], 
                    [np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4])
    
    # PD control: torque = Kp * pos_error - Kd * vel
    torque_d = -torque_kd * curr_vel
    # Clip derivative term to prevent spikes
    torque_d_clipped = np.clip(torque_d, -0.2 * torque_max, 0.2 * torque_max)
    torque_target = torque_kp * q_err + torque_d_clipped
    
    # Clamp total torque to max limits
    torque_target = np.clip(torque_target, -torque_max, torque_max)
    
    return torque_target.astype(float)


@click.command()
@click.option('--output', '-o', default='data/ur5e_torque_pd_teleop', help="Directory to save trajectory data.")
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5's IP address")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--log_frequency', '-lf', default=120, type=float, help="Logging frequency in Hz (should match sim: dt=1/120, decimation=1).")
@click.option('--max_joint_delta', '-mjd', default=0.5, type=float, help="Max allowed joint delta from current position (rad). ~30 degrees.")
@click.option('--debug', is_flag=True, help="Use dummy Mello interface with fixed positions for testing.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration.")
def main(output, robot_ip, mello_port, log_frequency, max_joint_delta, debug, init_joints):
    # Default initial joint positions
    j_init_deg = np.array([0, -90, 90, -90, -90, 0], dtype=float)
    j_init = j_init_deg / 180.0 * np.pi

    # Control loop frequency for torque control (500Hz required)
    control_frequency = 500  # Hz - required for torque control
    
    # Connect to robot using RTDE with torque control flags
    rtde_c = RTDEControlInterface(
        robot_ip, 
        control_frequency,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT
    )
    rtde_r = RTDEReceiveInterface(robot_ip, control_frequency)

    # Choose between real and dummy Mello interface
    MelloInterface = DummyMelloTeleopInterface if debug else MelloTeleopInterface
    mello_kwargs = {} if debug else {'port': mello_port}

    try:
        with KeystrokeCounter() as key_counter, MelloInterface(**mello_kwargs) as mello:
            cv2.setNumThreads(1)
            
            if init_joints:
                # Use the same style of speed as joints_init_speed in RTDEInterpolationController
                move_speed = 1.05  # rad/s
                move_accel = 1.4   # rad/s^2
                ok = rtde_c.moveJ(j_init, move_speed, move_accel)
                if not ok:
                    raise RuntimeError("moveJ to initial joints failed")
            
            # PD torque control parameters (matching rtde_interpolation_controller.py)
            torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
            torque_kp = torque_max / np.array([0.25, 0.25, 0.5, 1, 1, 1])
            torque_kd = torque_max / (np.pi * 0.5)
            
            print(f"[INFO]: PD gains - Kp: {torque_kp}, Kd: {torque_kd}")
            print(f"[INFO]: Max torques: {torque_max}")
            
            # Timing - run main loop at 500Hz
            # Log at log_frequency (default 120Hz to match sim: dt=1/120, decimation=1)
            log_interval_steps = control_frequency / log_frequency  # 500/120 = 4.166... steps per log
            log_accumulator = 0.0  # Fractional accumulator for precise timing
            
            print(f"[INFO]: Control loop: {control_frequency}Hz")
            print(f"[INFO]: Logging at: {log_frequency}Hz (for sim replay with dt=1/{int(log_frequency)}, decimation=1)")
            print(f"[INFO]: Max joint delta: {np.degrees(max_joint_delta):.1f} degrees")
            
            # Data storage
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            all_episodes = []
            
            # Current episode data
            episode_time_data = []
            episode_joint_pos_data = []
            episode_joint_vel_data = []
            episode_torque_cmd_data = []
            episode_action_data = []  # Target joints from Mello
            episode_joint_current_as_torque_data = []
            episode_gripper_data = []
            
            # State
            iter_idx = 0
            stop = False
            is_recording = False
            episode_start_time = None
            target_joints = np.array(rtde_r.getActualQ(), dtype=float)  # Start at current position
            gripper_cmd = 1  # Default open
            mello_valid = True  # Track if Mello position is valid
            
            print("\n" + "="*60)
            print("Mello Teleop with PD Torque Control")
            print("="*60)
            print("Press 'C' to start recording")
            print("Press 'S' to stop recording and save episode")
            print("Press 'Q' to quit")
            print("Press 'Backspace' to discard current episode")
            if debug:
                print("(DEBUG MODE - using dummy Mello interface)")
            print("="*60 + "\n")
            
            # Main control loop at 500Hz
            while not stop:
                t_start = rtde_c.initPeriod()
                
                # Handle key presses (check every iteration for responsiveness)
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        episode_start_time = time.time()
                        episode_time_data = []
                        episode_joint_pos_data = []
                        episode_joint_vel_data = []
                        episode_torque_cmd_data = []
                        episode_action_data = []
                        episode_joint_current_as_torque_data = []
                        episode_gripper_data = []
                        log_accumulator = 0.0  # Reset accumulator
                        is_recording = True
                        print('[Recording started!]')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording and save
                        if is_recording and len(episode_time_data) > 0:
                            episode_dict = {
                                "time": torch.from_numpy(np.array(episode_time_data)).float(),
                                "joint_pos": torch.from_numpy(np.array(episode_joint_pos_data)).float(),
                                "joint_vel": torch.from_numpy(np.array(episode_joint_vel_data)).float(),
                                "torque_cmd": torch.from_numpy(np.array(episode_torque_cmd_data)).float(),
                                "action": torch.from_numpy(np.array(episode_action_data)).float(),
                                "joint_current_as_torque": torch.from_numpy(np.array(episode_joint_current_as_torque_data)).float(),
                                "gripper": torch.from_numpy(np.array(episode_gripper_data)).float(),
                                "torque_kp": torch.from_numpy(torque_kp).float(),
                                "torque_kd": torch.from_numpy(torque_kd).float(),
                                "torque_max": torch.from_numpy(torque_max).float(),
                                "log_frequency": log_frequency,  # For sim replay reference
                            }
                            all_episodes.append(episode_dict)
                            
                            # Save individual episode
                            ep_idx = len(all_episodes) - 1
                            ep_file = output_path / f"episode_{ep_idx:04d}.pt"
                            torch.save(episode_dict, ep_file)
                            print(f'[Episode {ep_idx} saved: {len(episode_time_data)} samples at {log_frequency}Hz]')
                        is_recording = False
                        print('[Recording stopped]')
                    elif key_stroke == Key.backspace:
                        # Discard current episode
                        if is_recording:
                            episode_time_data = []
                            episode_joint_pos_data = []
                            episode_joint_vel_data = []
                            episode_torque_cmd_data = []
                            episode_action_data = []
                            episode_joint_current_as_torque_data = []
                            episode_gripper_data = []
                            is_recording = False
                            print('[Episode discarded]')
                
                # Get latest Mello values (every iteration for smooth tracking)
                mello_values = mello.get_latest_values()
                new_target_joints = np.array(mello_values[:6], dtype=float)
                gripper_cmd = mello_values[6]
                
                # Get current state
                curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                curr_vel = np.array(rtde_r.getActualQd(), dtype=float)
                curr_current_as_torque = np.array(rtde_r.getActualCurrentAsTorque(), dtype=float)
                
                # Safety check: Mello target must be within max_joint_delta of current position
                joint_delta = np.abs(new_target_joints - curr_joints)
                max_delta = np.max(joint_delta)
                if max_delta > max_joint_delta:
                    if mello_valid:  # Only print once when becoming invalid
                        print(f"[WARNING] Mello target too far from current position!")
                        print(f"          Max delta: {np.degrees(max_delta):.1f}deg > limit: {np.degrees(max_joint_delta):.1f}deg")
                        print(f"          Joint deltas (deg): {np.degrees(joint_delta)}")
                        print(f"          Holding current position until Mello is closer.")
                    mello_valid = False
                    # Don't update target_joints - keep tracking current position
                else:
                    if not mello_valid:
                        print(f"[INFO] Mello target valid again, resuming tracking.")
                    mello_valid = True
                    target_joints = new_target_joints
                
                # Compute PD torque command
                torque_cmd = compute_pd_torque(
                    target_joints, curr_joints, curr_vel,
                    torque_kp, torque_kd, torque_max
                )
                
                # Send torque command
                rtde_c.directTorque(torque_cmd.tolist())
                
                # Log data at log_frequency using fractional accumulator
                log_accumulator += 1.0
                if log_accumulator >= log_interval_steps:
                    log_accumulator -= log_interval_steps  # Keep fractional remainder
                    
                    if is_recording and episode_start_time is not None:
                        t_current = time.time()
                        t_episode = t_current - episode_start_time
                        
                        episode_time_data.append(t_episode)
                        episode_joint_pos_data.append(curr_joints.copy())
                        episode_joint_vel_data.append(curr_vel.copy())
                        episode_torque_cmd_data.append(torque_cmd.copy())
                        episode_action_data.append(target_joints.copy())
                        episode_joint_current_as_torque_data.append(curr_current_as_torque.copy())
                        episode_gripper_data.append(gripper_cmd)
                
                # Status print every 2 seconds (every 1000 iterations at 500Hz)
                if iter_idx % 1000 == 0:
                    joint_err = np.max(np.abs(target_joints - curr_joints))
                    status = "RECORDING" if is_recording else "idle"
                    valid_str = "" if mello_valid else " [MELLO INVALID]"
                    n_samples = len(episode_time_data) if is_recording else 0
                    print(f"[{status}] Ep: {len(all_episodes)}, Samples: {n_samples}, "
                          f"Err: {np.degrees(joint_err):.1f}deg{valid_str}")
                
                rtde_c.waitPeriod(t_start)
                iter_idx += 1
            
            # Save all episodes summary
            if len(all_episodes) > 0:
                summary_file = output_path / "all_episodes.pt"
                torch.save({
                    "episodes": all_episodes,
                    "n_episodes": len(all_episodes),
                    "log_frequency": log_frequency,
                    "control_frequency": control_frequency,
                    "torque_kp": torque_kp,
                    "torque_kd": torque_kd,
                    "torque_max": torque_max,
                }, summary_file)
                
                total_samples = sum(len(ep['time']) for ep in all_episodes)
                print(f"\n{'='*60}")
                print(f"Saved {len(all_episodes)} episodes to {output_path}")
                print(f"Total samples: {total_samples}")
                print(f"Log frequency: {log_frequency}Hz")
                print(f"\nFor sim replay (dt=1/{int(log_frequency)}, decimation=1):")
                print(f"  - Use 'action' as target joint positions")
                print(f"  - Use torque_kp, torque_kd for PD gains")
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
