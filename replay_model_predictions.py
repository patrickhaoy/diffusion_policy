"""
Script to run model predictions on real robot open-loop using MEAN predictions.

This script:
1. Loads a trained diffusion policy model from checkpoint
2. Runs inference on a dataset trajectory to get predicted actions (using MEAN, not sampling)
3. Executes those predicted actions on the real robot (similar to replay_ee_actions.py)

Usage:
python replay_model_predictions.py \
    --checkpoint /path/to/model.ckpt \
    --dataset /path/to/dataset.zarr \
    --episode 0 \
    --output /path/to/output \
    --robot_ip 192.168.1.10

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start model prediction replay.
Press "Q" to exit program.

================ Model Prediction Replay ==============
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
import dill
import hydra
import pathlib
import zarr
from typing import Dict, List
from omegaconf import OmegaConf

# Diffusion policy imports
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_ours
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Robomimic imports
import robomimic.utils.torch_utils as TorchUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    if not pathlib.Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # Create workspace and load model
    cls = hydra.utils.get_class(cfg._target_)
    cfg['policy']['obs_encoder']['extra_randomizations'] = []
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy: BaseImagePolicy = getattr(workspace, 'model')
    if cfg.training.use_ema:
        policy = getattr(workspace, 'ema_model')
    
    policy.eval().to(device)
    
    # Set inference parameters
    if hasattr(policy, 'num_inference_steps'):
        setattr(policy, 'num_inference_steps', 16)
    
    n_obs_steps = getattr(policy, 'n_obs_steps', cfg.get('n_obs_steps', 2))
    n_action_steps = getattr(policy, 'n_action_steps', cfg.get('n_action_steps', 1))
    
    print(f"Model loaded successfully")
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"n_action_steps: {n_action_steps}")
    print(f"Policy type: {type(policy).__name__}")
    print(f"Using MEAN predictions (not sampling) for stable actions")
    
    return policy, cfg


def load_dataset_trajectory(zarr_path: str, episode_idx: int = 0):
    """Load observations from zarr dataset for specific episode."""
    if not pathlib.Path(zarr_path).exists():
        raise FileNotFoundError(f"Dataset not found: {zarr_path}")
    
    print(f"Loading dataset trajectory from {zarr_path}")
    
    # Open zarr dataset
    root = zarr.open(zarr_path, mode='r')
    
    # Validate structure
    if 'data' not in root or 'meta' not in root or 'episode_ends' not in root['meta']:
        raise ValueError("Invalid dataset structure")
    
    data_group = root['data']
    
    # Get episode boundaries
    episode_ends = np.array(root['meta']['episode_ends'][:])
    if episode_idx >= len(episode_ends):
        raise ValueError(f"Episode index {episode_idx} out of range. Dataset has {len(episode_ends)} episodes.")
    
    # Calculate episode boundaries
    if episode_idx == 0:
        start_idx = 0
    else:
        start_idx = int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    episode_length = end_idx - start_idx
    
    print(f"Loading episode {episode_idx}: timesteps {start_idx} to {end_idx} (length: {episode_length})")
    
    # Load observations
    observations = {}
    obs_group = data_group['obs']
    
    # Get observation keys
    try:
        if hasattr(obs_group, 'keys'):
            obs_keys = list(obs_group.keys())
        else:
            obs_keys = [str(k) for k in obs_group]
    except Exception:
        obs_keys = ['front_rgb', 'side_rgb', 'wrist_rgb', 'arm_joint_pos', 'end_effector_pose']
        obs_keys = [k for k in obs_keys if k in obs_group]
    
    for key in obs_keys:
        obs_data = obs_group[key]
        if len(obs_data.shape) == 4:  # Image data (T, H, W, C)
            observations[key] = np.array(obs_data[start_idx:end_idx])
        else:  # Low-dim data
            observations[key] = np.array(obs_data[start_idx:end_idx])
    
    # Also load ground truth actions for comparison
    if 'actions' in data_group:
        gt_actions = np.array(data_group['actions'][start_idx:end_idx])
    elif 'action' in data_group:
        gt_actions = np.array(data_group['action'][start_idx:end_idx])
    else:
        gt_actions = None
        print("Warning: No ground truth actions found in dataset")
    
    episode_info = {
        'episode_idx': episode_idx,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'length': episode_length,
        'obs_keys': list(observations.keys())
    }
    
    print(f"Loaded observations for {episode_length} timesteps")
    print(f"Observation keys: {episode_info['obs_keys']}")
    
    return observations, gt_actions, episode_info


def predict_actions_from_observations(policy: BaseImagePolicy, observations: Dict[str, np.ndarray], 
                                    cfg: dict, device: str = 'cuda') -> np.ndarray:
    """Run model inference on observations to get predicted actions using MEAN."""
    print("Running model inference on dataset trajectory...")
    print("NOTE: Using MEAN of action distribution for stable predictions")
    
    n_obs_steps = getattr(policy, 'n_obs_steps', cfg.get('n_obs_steps', 2))
    episode_length = len(list(observations.values())[0])
    predicted_actions = []
    
    # Reset policy state
    if hasattr(policy, 'reset'):
        policy.reset()
    
    with torch.no_grad():
        for t in range(episode_length - n_obs_steps + 1):
            # Extract observation window
            obs_window = {}
            for key, obs_data in observations.items():
                obs_window[key] = obs_data[t:t + n_obs_steps]
            
            # Process observations (format conversion)
            obs_dict_np = get_real_obs_ours(
                env_obs=obs_window, 
                shape_meta=cfg['shape_meta']
            )
            
            # Convert to tensors
            obs_dict = {}
            for key, value in obs_dict_np.items():
                obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            
            # Run model inference - the policy now uses MEAN internally
            result = policy.predict_action(obs_dict)
            action_tensor = result['action'][0]  # Remove batch dimension
            
            # Handle different action tensor shapes
            if len(action_tensor.shape) == 2:
                # Multi-step prediction (e.g., diffusion policy)
                action = action_tensor[0].detach().to('cpu').numpy()  # Take first action
            elif len(action_tensor.shape) == 1:
                # Single-step prediction (e.g., MLP policy)
                action = action_tensor.detach().to('cpu').numpy()
            else:
                raise ValueError(f"Unexpected action tensor shape: {action_tensor.shape}")
            
            predicted_actions.append(action)
            
            if t % 20 == 0:
                print(f"Processed {t+1}/{episode_length - n_obs_steps + 1} timesteps")
    
    predicted_actions = np.array(predicted_actions)
    print(f"Generated {len(predicted_actions)} MEAN-based action predictions with shape {predicted_actions.shape}")
    
    return predicted_actions


def get_camera_obs_key(vis_camera_idx):
    """Get the correct camera observation key based on index."""
    camera_keys = ['front_rgb', 'side_rgb', 'wrist_rgb']
    if vis_camera_idx < len(camera_keys):
        return camera_keys[vis_camera_idx]
    else:
        print(f"Warning: Camera index {vis_camera_idx} out of range, using front camera")
        return camera_keys[0]


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to model checkpoint')
@click.option('--dataset', '-d', required=True, help='Path to zarr dataset')
@click.option('--episode', '-e', default=0, type=int, help='Episode index to use for prediction')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="Robot's IP address e.g. 192.168.1.10")
@click.option('--vis_camera_idx', default=0, type=int, help="Which camera to visualize (0=front, 1=side, 2=wrist)")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration")
@click.option('--max_duration', '-md', default=60, help='Max duration for replay in seconds')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Command execution latency in seconds")
@click.option('--cartesian_delta', is_flag=True, default=True, help='Use Cartesian delta control mode')
@click.option('--delta_scale', default=0.1, type=float, help='Scale factor for delta poses')
@click.option('--device', default='cuda', help='Device to run inference on (cuda/cpu)')
@click.option('--use_ground_truth', is_flag=True, default=False, help='Use ground truth actions instead of model predictions')
def main(checkpoint, dataset, episode, output, robot_ip, vis_camera_idx, init_joints,
         max_duration, frequency, command_latency, cartesian_delta, delta_scale, device, use_ground_truth):
    """Run model predictions on real robot open-loop using MEAN predictions."""
    
    print("="*60)
    if use_ground_truth:
        print("GROUND TRUTH ACTION REPLAY ON REAL ROBOT")
    else:
        print("MODEL PREDICTION REPLAY ON REAL ROBOT (MEAN-BASED)")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset: {dataset}")
    print(f"Episode: {episode}")
    print(f"Robot IP: {robot_ip}")
    print(f"Frequency: {frequency} Hz")
    print(f"Cartesian delta mode: {cartesian_delta}")
    print(f"Delta scale: {delta_scale}")
    print(f"Device: {device}")
    if use_ground_truth:
        print(f"Action mode: GROUND TRUTH (demonstration replay)")
    else:
        print(f"Prediction mode: MEAN (stable, deterministic)")
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    try:
        # 2. Load dataset trajectory
        observations, gt_actions, episode_info = load_dataset_trajectory(dataset, episode)
        
        if use_ground_truth:
            # Use ground truth actions directly
            actions_to_execute = gt_actions
            print(f"\nUsing ground truth actions!")
            print(f"Ground truth has {len(gt_actions)} actions")
        else:
            # 1. Load model and run inference
            policy, cfg = load_model(checkpoint, device)
            
            # 3. Run model inference to get predicted actions (using MEAN)
            actions_to_execute = predict_actions_from_observations(policy, observations, cfg, device)
        
        print(f"\nAction preparation completed!")
        if use_ground_truth:
            print(f"Using {len(actions_to_execute)} ground truth actions")
        else:
            print(f"Generated {len(actions_to_execute)} MEAN-based predicted actions")
            if gt_actions is not None:
                print(f"Ground truth has {len(gt_actions)} actions")
                # Quick comparison
                if len(actions_to_execute) > 0 and len(gt_actions) > 0:
                    min_len = min(len(actions_to_execute), len(gt_actions))
                    mse = np.mean((actions_to_execute[:min_len] - gt_actions[:min_len])**2)
                    print(f"MSE vs ground truth: {mse:.6f}")
        
        # 4. Execute actions on real robot
        print(f"\n{'='*60}")
        if use_ground_truth:
            print("STARTING REAL ROBOT EXECUTION WITH GROUND TRUTH ACTIONS")
        else:
            print("STARTING REAL ROBOT EXECUTION WITH MEAN PREDICTIONS")
        print(f"{'='*60}")
        
        # Setup device and camera configs
        configs = [
            json.load(open("diffusion_policy/real_world/realsense_config/455_front.json")),
            json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
            json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json"))
        ]
        
        # Setup timing
        dt = 1/frequency
        obs_res = (224, 224)
        n_obs_steps = 1  # For replay, we only need current observation
        
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
                camera_serial_numbers=['215122255213', '832112070487', '746112060198'],
                camera_configs=configs,
                thread_per_video=3,
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
                
                # Get current robot state
                robot_state = env.get_robot_state()
                initial_tcp_pose = robot_state['TargetTCPPose']
                print(f"Current robot TCP pose: {initial_tcp_pose}")
                
                time.sleep(2.0)
                
                if use_ground_truth:
                    print('Ready for GROUND TRUTH action replay!')
                    print('Press "C" in the camera window to start replay')
                else:
                    print('Ready for MEAN-based model prediction replay!')
                    print('Press "C" in the camera window to start replay')
                
                # Get camera key for visualization
                camera_key = get_camera_obs_key(vis_camera_idx)
                print(f"Using camera: {camera_key}")
                
                while True:
                    # ========== Human control phase ==============
                    obs = env.get_obs()
                    
                    # Show camera view
                    vis_img = obs[camera_key][-1]
                    if use_ground_truth:
                        text = f'Press "C" to start GROUND TRUTH action replay, "Q" to quit'
                        info_text = f'Episode {episode}: {len(actions_to_execute)} ground truth actions'
                    else:
                        text = f'Press "C" to start MEAN MODEL prediction replay, "Q" to quit'
                        info_text = f'Episode {episode}: {len(actions_to_execute)} MEAN predictions'
                    
                    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(vis_img, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    if not use_ground_truth:
                        cv2.putText(vis_img, f'Stable deterministic actions (no sampling)', 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    if gt_actions is not None:
                        cv2.putText(vis_img, f'Ground truth: {len(gt_actions)} actions', 
                                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    cv2.imshow('MEAN Model Prediction Replay', vis_img[...,::-1])
                    
                    key_stroke = cv2.waitKey(1) & 0xFF
                    if key_stroke == ord('q'):
                        print("Exiting...")
                        break
                    elif key_stroke == ord('c'):
                        print("Starting MEAN-based model prediction replay...")
                        
                        # ========== Model prediction replay phase ==============
                        try:
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            
                            # Wait for precise timing
                            frame_latency = 1/30
                            precise_wait(eval_t_start - frame_latency, time_func=time.time)
                            if use_ground_truth:
                                print("Ground truth action replay started!")
                            else:
                                print("MEAN-based model prediction replay started!")
                            
                            # Execute actions
                            for action_idx, action in enumerate(actions_to_execute):
                                # Calculate timing for this action
                                t_cycle_end = t_start + (action_idx + 1) * dt
                                
                                # Get current observations
                                obs = env.get_obs()
                                obs_timestamps = obs['timestamp']
                                action_type = "GROUND TRUTH" if use_ground_truth else "MEAN"
                                print(f'{action_type} Action {action_idx+1}/{len(actions_to_execute)}, Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')
                                
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
                                
                                # Execute action on robot
                                try:
                                    if cartesian_delta:
                                        # Use cartesian control method with delta actions
                                        env.exec_cartesian_actions(
                                            target_poses=this_target_poses[:1],
                                            timestamps=action_timestamps[:1],
                                            delta_actions=this_target_poses[:1]  # Model outputs are delta actions
                                        )
                                    else:
                                        # Use joint control method
                                        env.exec_actions(
                                            actions=this_target_poses[:1],
                                            timestamps=action_timestamps[:1]
                                        )
                                    
                                    # Log executed action
                                    action_type = "GROUND TRUTH" if use_ground_truth else "MEAN"
                                    if len(action) >= 7:
                                        print(f"Executed {action_type} action {action_idx+1}: pos={action[:3]} rot={action[3:6]} gripper={action[6]:.3f}")
                                    elif len(action) >= 6:
                                        print(f"Executed {action_type} action {action_idx+1}: pos={action[:3]} rot={action[3:6]}")
                                    else:
                                        print(f"Executed {action_type} action {action_idx+1}: {action}")
                                    
                                except Exception as e:
                                    print(f"Error executing action {action_idx}: {e}")
                                    break
                                
                                # Show progress
                                vis_img = obs[camera_key][-1]
                                if use_ground_truth:
                                    progress = f'GROUND TRUTH: {action_idx+1}/{len(actions_to_execute)} ({(action_idx+1)/len(actions_to_execute)*100:.1f}%)'
                                else:
                                    progress = f'MEAN MODEL: {action_idx+1}/{len(actions_to_execute)} ({(action_idx+1)/len(actions_to_execute)*100:.1f}%)'
                                time_remaining = (len(actions_to_execute) - action_idx - 1) * dt
                                status_text = f'{progress}, {time_remaining:.1f}s remaining'
                                
                                cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                cv2.putText(vis_img, 'Press "S" to stop', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                if use_ground_truth:
                                    cv2.putText(vis_img, 'GROUND TRUTH ACTIONS', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                else:
                                    cv2.putText(vis_img, 'STABLE MEAN PREDICTIONS', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                
                                # Display action values
                                action_prefix = "GT" if use_ground_truth else "MEAN"
                                if len(action) >= 7:
                                    action_text = f'{action_prefix}: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] G: {action[6]:.3f}'
                                elif len(action) >= 6:
                                    action_text = f'{action_prefix}: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}]'
                                else:
                                    action_text = f'{action_prefix} Action: {action}'
                                
                                cv2.putText(vis_img, action_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                
                                # Show comparison with ground truth if available
                                if gt_actions is not None and action_idx < len(gt_actions):
                                    gt_action = gt_actions[action_idx]
                                    if len(gt_action) >= 7:
                                        gt_text = f'GT: [{gt_action[0]:.3f}, {gt_action[1]:.3f}, {gt_action[2]:.3f}] G: {gt_action[6]:.3f}'
                                    elif len(gt_action) >= 6:
                                        gt_text = f'GT: [{gt_action[0]:.3f}, {gt_action[1]:.3f}, {gt_action[2]:.3f}, {gt_action[3]:.3f}, {gt_action[4]:.3f}, {gt_action[5]:.3f}]'
                                    else:
                                        gt_text = f'Ground Truth: {gt_action}'
                                    cv2.putText(vis_img, gt_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 255, 128), 1)
                                
                                cv2.imshow('MEAN Model Prediction Replay', vis_img[...,::-1])
                                
                                # Check for stop command
                                key_stroke = cv2.pollKey()
                                if key_stroke == ord('s'):
                                    print("MEAN model prediction replay stopped by user")
                                    break
                                
                                # Check timeout
                                if time.monotonic() - t_start > max_duration:
                                    print("MEAN model prediction replay terminated by timeout")
                                    break
                                
                                # Wait for next action timing
                                precise_wait(t_cycle_end - frame_latency)
                            
                            print("MEAN-based model prediction replay completed!")
                            env.end_episode()
                            
                        except KeyboardInterrupt:
                            print("MEAN model prediction replay interrupted!")
                            env.end_episode()
                        except Exception as e:
                            print(f"Error during MEAN model prediction replay: {e}")
                            env.end_episode()
                    
                    time.sleep(0.01)  # Small delay in main loop
                
                cv2.destroyAllWindows()
                print("MEAN-based model prediction replay session ended")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    main()
