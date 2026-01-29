"""
Script to replay end-effector actions from zarr replay buffer open loop.

Usage:
python replay_ee_actions.py -i <zarr_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start replay (hand control over to policy).
Press "Q" to exit program.

================ Replay in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop replay and gain control back.
"""

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import json
import pathlib
import zarr
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

def load_actions_from_zarr(zarr_path, episode_idx=0):
    """Load actions data from zarr replay buffer.
    
    Args:
        zarr_path: Path to the zarr dataset
        episode_idx: Episode index to load (default: 0)
        
    Returns:
        np.ndarray: Array of actions for the specified episode (cartesian delta format)
    """
    if not pathlib.Path(zarr_path).exists():
        raise FileNotFoundError(f"Zarr dataset path does not exist: {zarr_path}")
    
    # Open zarr dataset
    root = zarr.open(zarr_path, mode='r')
    
    # Validate dataset structure
    if 'data' not in root:
        raise ValueError("Dataset does not contain 'data' group")
    
    data_group = root['data']
    
    if 'actions' not in data_group:
        raise ValueError("Dataset does not contain 'actions' array")
    
    # Get episode boundaries
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends']
        episode_ends_array = np.array(episode_ends[:])
        if episode_idx >= len(episode_ends_array):
            raise ValueError(f"Episode index {episode_idx} out of range. Dataset has {len(episode_ends_array)} episodes.")
        
        # Calculate episode boundaries  
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = int(episode_ends_array[episode_idx - 1])
        end_idx = int(episode_ends_array[episode_idx])
        
        episode_length = end_idx - start_idx
        print(f"Loading episode {episode_idx}: timesteps {start_idx} to {end_idx} (length: {episode_length})")
    else:
        raise ValueError("Dataset does not contain episode metadata")
    
    # Load actions for the specified episode
    actions = np.array(data_group['actions'][start_idx:end_idx])
    
    print(f"Loaded {len(actions)} actions with shape {actions.shape}")
    print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    
    return actions

def get_camera_obs_key(vis_camera_idx):
    """Get the correct camera observation key based on index."""
    camera_keys = ['front_rgb', 'side_rgb', 'wrist_rgb']
    if vis_camera_idx < len(camera_keys):
        return camera_keys[vis_camera_idx]
    else:
        # Fallback to first camera if index is out of range
        print(f"Warning: Camera index {vis_camera_idx} out of range, using front camera")
        return camera_keys[0]

@click.command()
@click.option('--input', '-i', default='/home/iggy/research/github/diffusion_policy/diffusion_policy/peg_insertion_demos_match_sim_trial/replay_buffer_changed_test.zarr', help='Path to zarr replay buffer')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="Robot's IP address e.g. 192.168.0.204")
@click.option('--episode_idx', '-ei', default=0, type=int, help='Which episode to replay')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize (0=front, 1=side, 2=wrist).")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each replay in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between command and execution in Sec.")
@click.option('--cartesian_delta', is_flag=True, default=True, help='Use Cartesian delta control mode for ee_actions.')
@click.option('--delta_scale', default=0.1, type=float, help='Scale factor for delta poses.')
def main(input, output, robot_ip, episode_idx, vis_camera_idx, init_joints, 
         max_duration, frequency, command_latency, cartesian_delta, delta_scale):
    
    print(f"=== CARTESIAN ACTION REPLAY FROM ZARR ===")
    print(f"Zarr path: {input}")
    print(f"Episode index: {episode_idx}")
    print(f"Robot IP: {robot_ip}")
    print(f"Frequency: {frequency} Hz")
    print(f"Cartesian delta mode: {cartesian_delta}")
    print(f"Delta scale: {delta_scale}")
    
    # Load actions from zarr (in cartesian delta format)
    actions = load_actions_from_zarr(input, episode_idx)
    
    # Setup device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Load RealSense camera configurations
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "415_wrist.json"))
    ]
    
    # Setup timing
    dt = 1/frequency
    
    # Observation setup
    obs_res = (224, 224)  # Common resolution
    n_obs_steps = 1  # For replay, we only need current observation
    
    print(f"\n=== STARTING REAL ROBOT CARTESIAN ACTION REPLAY ===")
    
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            rolling_action_buffer=True,
            camera_serial_numbers=['215122255213', '832112070487',
                                  '746112060198'],
            camera_configs=configs,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            
            # Set delta scale if in cartesian delta mode
            if cartesian_delta:
                env.set_delta_scale(delta_scale)
            
            cv2.setNumThreads(1)

            # Camera setup
            env.realsense.set_exposure(exposure=500, gain=0)
            env.realsense.set_white_balance(white_balance=2000)

            print("Waiting for realsense")
            time.sleep(1.0)

            # Get current robot state for reference
            robot_state = env.get_robot_state()
            initial_tcp_pose = robot_state['TargetTCPPose']
            print(f"Current robot TCP pose: {initial_tcp_pose}")
            
            time.sleep(2.0)

            print('Ready for cartesian action replay!')
            print('Press "C" in the camera window to start replay')
            
            # Get the correct camera key for visualization
            camera_key = get_camera_obs_key(vis_camera_idx)
            print(f"Using camera: {camera_key}")

            while True:
                # ========== Human control phase ==============
                obs = env.get_obs()
                
                # Show camera view
                vis_img = obs[camera_key][-1]
                text = f'Press "C" to start cartesian action replay, "Q" to quit'
                cv2.putText(
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2,
                    color=(0, 255, 0)
                )
                cv2.putText(
                    vis_img,
                    f'Episode {episode_idx}: {len(actions)} actions',
                    (10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    thickness=2,
                    color=(255, 255, 0)
                )
                cv2.imshow('Action Replay', vis_img[...,::-1])
                
                key_stroke = cv2.waitKey(1) & 0xFF
                if key_stroke == ord('q'):
                    print("Exiting...")
                    break
                elif key_stroke == ord('c'):
                    print("Starting cartesian action replay...")
                    
                    # ========== Cartesian Action replay phase ==============
                    try:
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start)
                        
                        # Wait for precise timing
                        frame_latency = 1/30
                        precise_wait(eval_t_start - frame_latency, time_func=time.time)
                        print("Cartesian action replay started!")
                        
                        # Execute cartesian actions
                        for action_idx, action in enumerate(actions):
                            # Calculate timing for this action
                            t_cycle_end = t_start + (action_idx + 1) * dt
                            
                            # Get current observations
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            print(f'Action {action_idx+1}/{len(actions)}, Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')
                            
                            # Prepare action for execution
                            if len(action.shape) == 1:
                                this_target_poses = np.expand_dims(action, axis=0)
                            else:
                                this_target_poses = action
                            
                            # Handle timing
                            action_timestamps = np.array([obs_timestamps[-1] + dt])
                            action_exec_latency = command_latency
                            curr_time = time.time()
                            
                            # Check if we're on schedule
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # Behind schedule
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(f'Behind schedule: {action_timestamp - curr_time:.3f}s')
                                action_timestamps = np.array([action_timestamp])
                            
                            # Execute cartesian action on robot
                            try:
                                if cartesian_delta:
                                    # Use cartesian control method with delta actions
                                    env.exec_cartesian_actions(
                                        target_poses=this_target_poses[:1],
                                        timestamps=action_timestamps[:1],
                                        delta_actions=this_target_poses[:1]  # actions are in cartesian delta format
                                    )
                                else:
                                    # Use joint control method (interpret actions as joint actions)
                                    env.exec_actions(
                                        actions=this_target_poses[:1],
                                        timestamps=action_timestamps[:1]
                                    )
                                # Handle different action dimensions
                                if len(action) >= 7:
                                    print(f"Executed action {action_idx+1}: pos={action[:3]} rot={action[3:6]} gripper={action[6]:.3f}")
                                elif len(action) >= 6:
                                    print(f"Executed action {action_idx+1}: pos={action[:3]} rot={action[3:6]}")
                                else:
                                    print(f"Executed action {action_idx+1}: {action}")
                                
                            except Exception as e:
                                print(f"Error executing action {action_idx}: {e}")
                                break
                            
                            # Show progress
                            vis_img = obs[camera_key][-1]
                            progress = f'Replay: {action_idx+1}/{len(actions)} ({(action_idx+1)/len(actions)*100:.1f}%)'
                            time_remaining = (len(actions) - action_idx - 1) * dt
                            status_text = f'{progress}, {time_remaining:.1f}s remaining'
                            
                            cv2.putText(
                                vis_img,
                                status_text,
                                (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6,
                                thickness=2,
                                color=(0, 255, 255)
                            )
                            cv2.putText(
                                vis_img,
                                'Press "S" to stop',
                                (10, 60),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6,
                                thickness=2,
                                color=(0, 0, 255)
                            )
                            # Display action values based on dimension
                            if len(action) >= 7:
                                action_text = f'Cartesian: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] Gripper: {action[6]:.3f}'
                            elif len(action) >= 6:
                                action_text = f'Cartesian: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}]'
                            else:
                                action_text = f'Action: {action}'
                            
                            cv2.putText(
                                vis_img,
                                action_text,
                                (10, 90),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255, 255, 255)
                            )
                            cv2.imshow('Action Replay', vis_img[...,::-1])
                            
                            # Check for stop command
                            key_stroke = cv2.pollKey()
                            if key_stroke == ord('s'):
                                print("Replay stopped by user")
                                break
                            
                            # Check timeout
                            if time.monotonic() - t_start > max_duration:
                                print("Replay terminated by timeout")
                                break
                            
                            # Wait for next action timing
                            precise_wait(t_cycle_end - frame_latency)
                        
                        print("Cartesian action replay completed!")
                        env.end_episode()
                        
                    except KeyboardInterrupt:
                        print("Replay interrupted!")
                        env.end_episode()
                    except Exception as e:
                        print(f"Error during replay: {e}")
                        env.end_episode()
                
                time.sleep(0.01)  # Small delay in main loop

            cv2.destroyAllWindows()
            print("Cartesian action replay session ended")

if __name__ == '__main__':
    main()
