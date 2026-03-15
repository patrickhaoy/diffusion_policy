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
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--debug', is_flag=True, help="Use dummy Mello interface with fixed joint positions for testing.")
@click.option('--osc_kp_pos', default=1000.0, type=float, help="OSC position stiffness (default 1000)")
@click.option('--osc_kp_rot', default=50.0, type=float, help="OSC rotation stiffness (default 50)")
def main(output, robot_ip, mello_port, vis_camera_idx, init_joints, frequency, command_latency, debug, osc_kp_pos, osc_kp_rot):

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
        MelloInterface = DummyMelloTeleopInterface if debug else MelloTeleopInterface
        mello_kwargs = {} if debug else {'port': mello_port}
        with KeystrokeCounter() as key_counter, \
            MelloInterface(**mello_kwargs) as mello, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip,
                obs_image_resolution=(640,480),
                camera_serial_numbers=['215122255213', '832112070487',
                        '746112060198'],
                camera_configs=configs,
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                osc_kp_pos=osc_kp_pos,
                osc_kp_rot=osc_kp_rot,
            ) as env:
            cv2.setNumThreads(1)

            time.sleep(1.0)
            kd_pos = 2 * np.sqrt(osc_kp_pos) * 1.0
            kd_rot = 2 * np.sqrt(osc_kp_rot) * 1.0
            print(f'OSC: Kp_pos={osc_kp_pos}, Kp_rot={osc_kp_rot}, Kd_pos={kd_pos:.1f}, Kd_rot={kd_rot:.1f}')
            print('Ready!')
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                if debug:
                    text += ' (DEBUG)'
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

                mello_values = mello.get_latest_values()
                mello_joints = mello_values[:6]
                gripper_command = mello_values[6]
                unified_action = np.concatenate([mello_joints, [gripper_command]])

                env.exec_actions(
                    actions=[unified_action], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
