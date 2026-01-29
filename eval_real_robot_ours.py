"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
Press "R" to reset robot to initial position and start new trajectory.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import json
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution_ours, 
    get_real_obs_ours
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Add imageio import for video saving
import imageio
from scipy.spatial.transform import Rotation as R

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)



@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, 
              help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, 
              help="UR5's IP address e.g. 192.168.1.10")
@click.option('--match_dataset', '-m', default=None, 
              help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, 
              help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, 
              help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, 
              help="Whether to initialize robot joint configuration in the "
                   "beginning.")
@click.option('--steps_per_inference', '-si', default=1, type=int, 
              help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=1000, 
              help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, 
              help="Control frequency in Hz.")
@click.option('--save_video', is_flag=True, default=False,
              help='Save video of concatenated camera views.')
@click.option('--action_scale', default=1.0, type=float,
              help='Scale factor for relative joint actions.')
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints, 
         steps_per_inference, max_duration,
         frequency, save_video, action_scale):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/"
                      "415_wrist.json"))
    ]

    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    cfg['policy']['obs_encoder']['extra_randomizations'] = []
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

        policy.eval().to(device)

        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # setup experiments
    dt = 1/frequency
    obs_res = get_real_obs_resolution_ours(cfg['task']['shape_meta'])
    n_obs_steps = cfg['n_obs_steps']
    n_action_steps = cfg['n_action_steps']
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference: ", steps_per_inference)
    print("n_action_steps: ", n_action_steps)
    print("action_scale: ", action_scale)

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
            
            cv2.setNumThreads(1)

            print("Waiting for realsense")
            time.sleep(5.0)

            print("Warming up policy inference")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_ours(
                    env_obs=obs, shape_meta=cfg['shape_meta'])
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                try:
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().to('cpu').numpy()
                    del result
                except Exception as e:
                    print(e)
                    # Handle case where result might not be defined
                    if 'result' in locals():
                        del result

            print('Ready!')
            time.sleep(1.0)
            
            # Initialize video recording if enabled
            if save_video:
                frames_to_save = []
                print("Video recording enabled - will save concatenated camera views")

            actions = []
            
            while True:
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        # print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # Capture frames for video if enabled
                        if save_video:
                            # Get the latest frame from each camera and concatenate
                            camera_names = ['front_rgb', 'side_rgb', 'wrist_rgb']
                            imgs = []
                            for cam_name in camera_names:
                                if cam_name in obs:
                                    # Get the most recent frame (last in time dimension)
                                    img = obs[cam_name][-1]  # Shape: (H, W, C)
                                    # Convert from float [0,1] to uint8 [0,255] if needed
                                    if img.dtype == np.float32 or img.dtype == np.float64:
                                        img = (img * 255).clip(0, 255).astype(np.uint8)
                                    imgs.append(img)
                            
                            # Concatenate frames horizontally if we have all cameras
                            if len(imgs) == 3:
                                frame = np.concatenate(imgs, axis=1)
                                frames_to_save.append(frame)

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_ours(
                                env_obs=obs, shape_meta=cfg['shape_meta']
                            )
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0:1].detach().to('cpu').numpy() 

                            # print('Inference latency:', time.time() - s)
                        
                        # Convert relative joint actions to absolute joint positions
                        # action shape: (N, 7) where [:, :6] is joint delta, [:, 6] is gripper
                        current_joints = env.get_robot_state()['ActualQ']
                        joint_deltas = action[:, :6] * action_scale  # Scale relative joint actions
                        gripper_actions = action[:, 6:7]  # Keep gripper as-is
                        print(action)
                        
                        # Compute absolute joint targets by adding deltas to current position
                        absolute_joint_targets = current_joints + joint_deltas
                        target_actions = np.concatenate([absolute_joint_targets, gripper_actions], axis=1)

                        # deal with timing
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            target_actions = target_actions[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            target_actions = target_actions[is_new]
                            action_timestamps = action_timestamps[is_new]
                        
                        # Execute joint actions with torque control
                        actions.append(target_actions)
                        env.exec_actions(
                            actions=target_actions[:n_action_steps],
                            timestamps=action_timestamps[:n_action_steps]
                        )
                        print(f"Submitted {n_action_steps} steps of actions.")

                        # Visualize camera feed for key detection
                        episode_id = env.replay_buffer.n_episodes
                        camera_key = 'side_rgb'
                        if camera_key in obs:
                            vis_img = obs[camera_key][-1]
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('Policy Control', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break
                        elif key_stroke == ord('r'):
                            # Reset robot and start new trajectory
                            print('Resetting robot for new trajectory...')
                            env.end_episode()
                            
                            # Save video if recording and we have frames
                            if save_video and frames_to_save:
                                episode_id = getattr(env.replay_buffer, 'n_episodes', 0)
                                video_filename = f'policy_cameras_reset_{episode_id:03d}.mp4'
                                video_path = pathlib.Path(output) / video_filename
                                print(f"Saving reset video with {len(frames_to_save)} frames to {video_path}")
                                imageio.mimsave(str(video_path), frames_to_save, 
                                              fps=10, codec='libx264')
                                frames_to_save = []  # Reset for next episode
                            
                            # Reset policy state
                            policy.reset()
                            
                            # Move robot to initial position
                            env.robot.reset_to_initial_position()
                            
                            # Wait a moment for robot to settle
                            time.sleep(5.0)
                            
                            # Start new episode
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            precise_wait(eval_t_start - frame_latency, time_func=time.time)
                            
                            # Reset iteration counter
                            iter_idx = 0
                            term_area_start_timestamp = float('inf')
                            
                            print('Robot reset complete! Starting new trajectory.')
                            continue

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        # term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        # curr_pose = obs['robot_eef_pose'][-1]
                        # dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        # if dist < 0.03:
                        #     # in termination area
                        #     curr_timestamp = obs['timestamp'][-1]
                        #     if term_area_start_timestamp > curr_timestamp:
                        #         term_area_start_timestamp = curr_timestamp
                        #     else:
                        #         term_area_time = curr_timestamp - term_area_start_timestamp
                        #         if term_area_time > 0.5:
                        #             terminate = True
                        # #             print('Terminated by the policy!')
                        # else:
                        #     # out of the area
                        #     term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            
                            # Save video if recording and we have frames
                            if save_video and frames_to_save:
                                episode_id = getattr(env.replay_buffer, 'n_episodes', 0)
                                video_filename = f'policy_cameras_episode_{episode_id:03d}.mp4'
                                video_path = pathlib.Path(output) / video_filename
                                print(f"Saving video with {len(frames_to_save)} frames to {video_path}")
                                imageio.mimsave(str(video_path), frames_to_save, 
                                              fps=10, codec='libx264')
                                frames_to_save = []  # Reset for next episode
                            
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except Exception as e:
                    print(e)
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                    
                    # Save video if recording and we have frames
                    if save_video and frames_to_save:
                        episode_id = getattr(env.replay_buffer, 'n_episodes', 0)
                        video_filename = f'policy_cameras_interrupted_{episode_id:03d}.mp4'
                        video_path = pathlib.Path(output) / video_filename
                        print(f"Saving interrupted video with {len(frames_to_save)} frames to {video_path}")
                        imageio.mimsave(str(video_path), frames_to_save, 
                                      fps=10, codec='libx264')
                    
                    break
                
                print("Stopped.")



# %%
if __name__ == '__main__':

    main()
