"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Control robot joint positions using Mello or Gello device.
Gripper control is handled through device's 7th axis.

Device options:
--device mello: Use Mello device (default)
--device gello: Use Gello device

Debug mode (--debug flag):
When debug flag is set, uses fixed joint positions instead of actual device.
The robot will move to a "home" position and stay there.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface
from diffusion_policy.real_world.gello_teleop import GelloTeleopInterface, DummyGelloTeleopInterface

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--device', '-d', default='mello', type=click.Choice(['mello', 'gello']), help="Teleoperation device to use")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--gello_port', '-gp', default=None, help="Gello device serial port (auto-detected if not specified)")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=15, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--use_cameras', '-uc', default=True, type=bool, help="Whether to use RealSense cameras (True/False).")
@click.option('--debug', is_flag=True, help="Use dummy interface with fixed joint positions for testing.")
def main(output, robot_ip, device, mello_port, gello_port, vis_camera_idx, init_joints, frequency, command_latency, use_cameras, debug):
    dt = 1/frequency
    
    with SharedMemoryManager() as shm_manager:
        # Choose between real and dummy interfaces based on device type
        if device == 'mello':
            Interface = DummyMelloTeleopInterface if debug else MelloTeleopInterface
            interface_kwargs = {} if debug else {'port': mello_port}
        elif device == 'gello':
            Interface = DummyGelloTeleopInterface if debug else GelloTeleopInterface
            interface_kwargs = {} if debug else {'port': gello_port}
        else:
            raise ValueError(f"Unknown device type: {device}")
        
        with KeystrokeCounter() as key_counter, \
            Interface(**interface_kwargs) as teleop_device, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                use_cameras=use_cameras,
                # recording resolution
                obs_image_resolution=(640,480),
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

            if use_cameras:
                # realsense exposure
                env.realsense.set_exposure(exposure=500, gain=0)
                # realsense white balance
                env.realsense.set_white_balance(white_balance=2000)

            time.sleep(1.0)
            print(f'Ready! Using {device.upper()} device')
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
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
                stage = key_counter[Key.space]

                # visualize
                if use_cameras:
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                    episode_id = env.replay_buffer.n_episodes
                    text = f'Episode: {episode_id}, Stage: {stage}, Device: {device.upper()}'
                    if is_recording:
                        text += ', Recording!'
                    if debug:
                        text += ' (DEBUG MODE)'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(255,255,255)
                    )

                    cv2.imshow('default', vis_img)
                    cv2.pollKey()   

                precise_wait(t_sample)
                
                # Get latest device values
                device_values = teleop_device.get_latest_values()
                
                # Handle different joint configurations
                # 6 joints + 1 gripper
                joints = device_values[:6]  # First 6 values are joints
                gripper_command = device_values[6]  # 7th value is gripper (1 for open, -1 for closed)
                unified_action = np.concatenate([joints, [gripper_command]])

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
