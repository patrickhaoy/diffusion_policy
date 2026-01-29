"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Control robot joint positions using Mello device.
Gripper control is handled through Mello's 7th axis.

Debug mode (--debug flag):
When debug flag is set, uses fixed joint positions instead of Mello device.
The robot will move to a "home" position and stay there.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.

PACE data collection (for sim2real alignment):
Press "T" to start/stop PACE TRAIN trajectory recording.
Press "V" to start/stop PACE VALIDATION trajectory recording.
Data is auto-saved when recording stops.

Macro controls:
Press "R" to go to the above peg tcp position
Press "Down Arrow" to use the screw macro
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface


def compute_pd_torque(target_joints, curr_joints, curr_vel, torque_kp, torque_kd, torque_max):
    """Compute PD torque for logging (matches rtde_interpolation_controller.py)."""
    q_err = np.array(target_joints) - np.array(curr_joints)
    torque_d = -torque_kd * np.array(curr_vel)
    torque_target = torque_kp * q_err + torque_d
    return np.clip(torque_target, -torque_max, torque_max)

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--debug', is_flag=True, help="Use dummy Mello interface with fixed joint positions for testing.")
@click.option('--pace_output', '-po', default='data/pace_teleop', help="Directory to save PACE trajectory data.")
def main(output, robot_ip, mello_port, vis_camera_idx, init_joints, frequency, command_latency, debug, pace_output):

    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "415_wrist.json"))
    ]

    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        # Choose between real and dummy Mello interface
        MelloInterface = DummyMelloTeleopInterface if debug else MelloTeleopInterface
        mello_kwargs = {} if debug else {'port': mello_port}
        with KeystrokeCounter() as key_counter, \
            MelloInterface(**mello_kwargs) as mello, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip,
                # recording resolution
                obs_image_resolution=(640,480),
                camera_serial_numbers=['215122255213', '832112070487',
                        '746112060198'],
                camera_configs=configs,
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            # PACE data logging setup
            pace_output_path = Path(pace_output)
            pace_output_path.mkdir(parents=True, exist_ok=True)
            
            # PD control parameters (must match rtde_interpolation_controller.py)
            torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
            torque_kp = torque_max / np.array([1, 1, 1, 1, 1, 1])  # Lower stiffness
            torque_kd = torque_max / (np.pi * 0.5)
            
            # PACE recording state
            is_pace_recording = False
            pace_record_type = None  # 'train' or 'val'
            pace_data = {
                'time': [], 'joint_pos': [], 'joint_vel': [],
                'joint_target': [], 'torque_cmd': []
            }
            pace_train_count = len(list(pace_output_path.glob('train_*.pt')))
            pace_val_count = len(list(pace_output_path.glob('val_*.pt')))

            time.sleep(1.0)
            print('Ready!')
            print(f'PACE output directory: {pace_output_path}')
            print(f'  Existing train trajectories: {pace_train_count}')
            print(f'  Existing val trajectories: {pace_val_count}')
            stage = key_counter[Key.space]
            t_start = time.monotonic()
            pace_t_start = None
            iter_idx = 0
            stop = False
            is_recording = False
            is_twisting = False
            is_reverse_twisting = False
            twist_step_count = 0
            reverse_twist_step_count = 0
            initial_wrist_position = 0.0
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()
                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                    elif key_stroke == Key.down:
                        # Start twisting macro (screw motion)
                        is_twisting = True
                        twist_step_count = 0
                    elif key_stroke == Key.up:
                        # Stop twisting macro
                        is_twisting = False
                    elif key_stroke == Key.left:
                        # Start reverse twisting macro (counter-clockwise)
                        is_reverse_twisting = True
                        reverse_twist_step_count = 0
                    elif key_stroke == Key.right:
                        # Stop reverse twisting macro
                        is_reverse_twisting = False
                    elif key_stroke == KeyCode(char='t'):
                        # Toggle PACE TRAIN recording
                        if not is_pace_recording:
                            is_pace_recording = True
                            pace_record_type = 'train'
                            pace_t_start = time.time()
                            pace_data = {'time': [], 'joint_pos': [], 'joint_vel': [],
                                        'joint_target': [], 'torque_cmd': []}
                            print('PACE TRAIN recording started!')
                        else:
                            # Save and stop
                            is_pace_recording = False
                            if len(pace_data['time']) > 0:
                                save_path = pace_output_path / f'train_{pace_train_count:02d}.pt'
                                save_data = {
                                    'time': torch.from_numpy(np.array(pace_data['time'])).float(),
                                    'dof_pos': torch.from_numpy(np.array(pace_data['joint_pos'])).float(),
                                    'des_dof_pos': torch.from_numpy(np.array(pace_data['joint_target'])).float(),
                                    'joint_pos': torch.from_numpy(np.array(pace_data['joint_pos'])).float(),
                                    'joint_vel': torch.from_numpy(np.array(pace_data['joint_vel'])).float(),
                                    'action': torch.from_numpy(np.array(pace_data['joint_target'])).float(),
                                    'torque_cmd': torch.from_numpy(np.array(pace_data['torque_cmd'])).float(),
                                    'torque_kp': torch.from_numpy(torque_kp).float(),
                                    'torque_kd': torch.from_numpy(torque_kd).float(),
                                    'torque_max': torch.from_numpy(torque_max).float(),
                                    'log_frequency': frequency,
                                    'n_samples': len(pace_data['time']),
                                }
                                torch.save(save_data, save_path)
                                pace_train_count += 1
                                print(f'PACE TRAIN saved: {save_path} ({len(pace_data["time"])} samples)')
                            pace_record_type = None
                    elif key_stroke == KeyCode(char='v'):
                        # Toggle PACE VALIDATION recording
                        if not is_pace_recording:
                            is_pace_recording = True
                            pace_record_type = 'val'
                            pace_t_start = time.time()
                            pace_data = {'time': [], 'joint_pos': [], 'joint_vel': [],
                                        'joint_target': [], 'torque_cmd': []}
                            print('PACE VALIDATION recording started!')
                        else:
                            # Save and stop
                            is_pace_recording = False
                            if len(pace_data['time']) > 0:
                                save_path = pace_output_path / f'val_{pace_val_count:02d}.pt'
                                save_data = {
                                    'time': torch.from_numpy(np.array(pace_data['time'])).float(),
                                    'dof_pos': torch.from_numpy(np.array(pace_data['joint_pos'])).float(),
                                    'des_dof_pos': torch.from_numpy(np.array(pace_data['joint_target'])).float(),
                                    'joint_pos': torch.from_numpy(np.array(pace_data['joint_pos'])).float(),
                                    'joint_vel': torch.from_numpy(np.array(pace_data['joint_vel'])).float(),
                                    'action': torch.from_numpy(np.array(pace_data['joint_target'])).float(),
                                    'torque_cmd': torch.from_numpy(np.array(pace_data['torque_cmd'])).float(),
                                    'torque_kp': torch.from_numpy(torque_kp).float(),
                                    'torque_kd': torch.from_numpy(torque_kd).float(),
                                    'torque_max': torch.from_numpy(torque_max).float(),
                                    'log_frequency': frequency,
                                    'n_samples': len(pace_data['time']),
                                }
                                torch.save(save_data, save_path)
                                pace_val_count += 1
                                print(f'PACE VALIDATION saved: {save_path} ({len(pace_data["time"])} samples)')
                            pace_record_type = None



                # visualize
                # vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                # episode_id = env.replay_buffer.n_episodes
                # text = f'Episode: {episode_id}, Stage: {stage}'
                # if is_recording:
                #     text += ', Recording!'
                # if debug:
                #     text += ' (DEBUG MODE)'
                # cv2.putText(
                #     vis_img,
                #     text,
                #     (10,30),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #     fontScale=1,
                #     thickness=2,
                #     color=(255,255,255)
                # )

                # cv2.imshow('default', vis_img)
                # cv2.pollKey()

                precise_wait(t_sample)
                
                # Get latest Mello values
                mello_values = mello.get_latest_values()
                mello_joints = mello_values[:6]  # First 6 values are joints
                gripper_command = mello_values[6]  # 7th value is gripper (1 for open, -1 for closed)
                unified_action = np.concatenate([mello_joints, [gripper_command]])
                
                current_joints = env.robot.get_state()["ActualQ"]

                # Handle twisting macro behavior
                if is_twisting:
                    # Store initial wrist position on first twist step
                    if twist_step_count == 0:
                        initial_wrist_position = current_joints[5]
                    
                    # Execute screw motion: move TCP down and rotate wrist
                    modified_joints = current_joints.copy()
                    modified_joints[1] += 0.0025  # Joint 1 adjustment
                    modified_joints[2] += 0.0025  # Joint 2 adjustment  
                    modified_joints[3] -= 0.005   # Joint 3 adjustment
                    modified_joints[5] += 0.5     # Wrist rotation
                    
                    unified_action = np.concatenate([modified_joints, [-1]])  # Close gripper
                    twist_step_count += 1
                
                elif is_reverse_twisting:
                    # Store initial wrist position on first reverse twist step
                    if reverse_twist_step_count == 0:
                        initial_wrist_position = current_joints[5]
                    
                    # Execute reverse screw motion: move TCP down and rotate wrist opposite direction
                    modified_joints = current_joints.copy()
                    modified_joints[1] += 0.0025  # Joint 1 adjustment
                    modified_joints[2] += 0.0025  # Joint 2 adjustment  
                    modified_joints[3] -= 0.005   # Joint 3 adjustment
                    modified_joints[5] -= 0.5     # Reverse wrist rotation
                    
                    unified_action = np.concatenate([modified_joints, [-1]])  # Close gripper
                    reverse_twist_step_count += 1
                
                elif twist_step_count > 0 or reverse_twist_step_count > 0:
                    # Return wrist to original position after twisting
                    wrist_error = initial_wrist_position - current_joints[5]
                    max_correction = np.pi/8
                    
                    if abs(wrist_error) > max_correction:
                        # Apply gradual correction back to initial position
                        wrist_correction = np.clip(wrist_error, -max_correction, max_correction)
                        modified_joints = current_joints.copy()
                        modified_joints[5] += wrist_correction
                        unified_action = np.concatenate([modified_joints, [1]])  # Open gripper
                    else:
                        # Close enough to initial position, reset twist states
                        twist_step_count = 0
                        reverse_twist_step_count = 0

                # Log PACE data if recording
                if is_pace_recording and pace_t_start is not None:
                    robot_state = env.robot.get_state()
                    curr_joints = np.array(robot_state["ActualQ"])
                    curr_vel = np.array(robot_state["ActualQd"])
                    target_joints = unified_action[:6]  # Joint targets being sent
                    
                    # Compute torque command (for logging, actual torque computed in controller)
                    torque_cmd = compute_pd_torque(
                        target_joints, curr_joints, curr_vel,
                        torque_kp, torque_kd, torque_max
                    )
                    
                    pace_data['time'].append(time.time() - pace_t_start)
                    pace_data['joint_pos'].append(curr_joints.copy())
                    pace_data['joint_vel'].append(curr_vel.copy())
                    pace_data['joint_target'].append(target_joints.copy())
                    pace_data['torque_cmd'].append(torque_cmd.copy())
                    
                    # Print progress every 50 samples
                    if len(pace_data['time']) % 50 == 0:
                        print(f'  PACE {pace_record_type}: {len(pace_data["time"])} samples')

                # execute teleop command
                env.exec_actions(
                    actions=[unified_action], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
