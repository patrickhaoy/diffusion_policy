#!/usr/bin/env python3
"""
Compare sim dataset images with real robot camera images.
Overlays them side-by-side and with alpha blending to check alignment.

Usage:
    python compare_sim_real_cameras.py \
        --dataset /home/patrickhaoy/research/OctiLab/datasets/test/rgb0.zarr \
        --robot_ip 192.168.1.10 \
        --output comparison_output
"""

import click
import numpy as np
import zarr
import cv2
import time
import json
from pathlib import Path
from multiprocessing.managers import SharedMemoryManager


@click.command()
@click.option('--dataset', '-d', required=True, help='Path to zarr dataset')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address")
@click.option('--output', '-o', default='camera_comparison', help='Output directory')
@click.option('--episode', '-e', default=0, type=int, help='Episode index to compare')
@click.option('--frame', '-f', default=0, type=int, help='Frame index within episode')
@click.option('--live', '-l', is_flag=True, default=False, help='Live comparison mode')
def main(dataset, robot_ip, output, episode, frame, live):
    """Compare sim dataset images with real robot camera images."""
    
    # Load zarr dataset
    print(f"Loading dataset: {dataset}")
    root = zarr.open(dataset, 'r')
    
    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    if episode >= len(episode_ends):
        print(f"Episode {episode} not found. Dataset has {len(episode_ends)} episodes.")
        return
    
    start_idx = episode_starts[episode]
    end_idx = episode_ends[episode]
    frame_idx = start_idx + frame
    
    if frame_idx >= end_idx:
        print(f"Frame {frame} not found in episode {episode} (has {end_idx - start_idx} frames)")
        return
    
    # Load sim images
    print(f"Loading sim images from episode {episode}, frame {frame} (global idx {frame_idx})")
    sim_front = root['data/obs/front_rgb'][frame_idx]
    sim_wrist = root['data/obs/wrist_rgb'][frame_idx]
    
    # Check if side_rgb exists
    has_side = 'data/obs/side_rgb' in root
    if has_side:
        sim_side = root['data/obs/side_rgb'][frame_idx]
    else:
        sim_side = None
        print("Note: side_rgb not found in dataset")
    
    # Convert from float [0,1] to uint8 if needed
    def to_uint8(img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img.astype(np.uint8)
    
    sim_front = to_uint8(sim_front)
    sim_wrist = to_uint8(sim_wrist)
    if sim_side is not None:
        sim_side = to_uint8(sim_side)
    
    # Handle channel format (CHW vs HWC)
    def ensure_hwc(img):
        if img.shape[0] == 3 or img.shape[0] == 4:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 4:  # RGBA -> RGB
            img = img[:, :, :3]
        return img
    
    sim_front = ensure_hwc(sim_front)
    sim_wrist = ensure_hwc(sim_wrist)
    if sim_side is not None:
        sim_side = ensure_hwc(sim_side)
    
    print(f"Sim image shapes: front={sim_front.shape}, wrist={sim_wrist.shape}", 
          f"side={sim_side.shape if sim_side is not None else 'N/A'}")
    
    # Load joint positions from dataset for robot initialization
    joint_pos = root['data/obs/arm_joint_pos'][frame_idx]
    print(f"Dataset joint positions (deg): {np.rad2deg(joint_pos)}")
    print(f"Dataset joint positions (rad): {joint_pos}")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize real robot and cameras
    from diffusion_policy.real_world.real_env import RealEnv
    
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json"))
    ]
    
    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=str(output_path),
            robot_ip=robot_ip,
            frequency=10,
            n_obs_steps=2,
            obs_image_resolution=(224, 224),
            obs_float32=True,
            init_joints=True,  # Initialize to custom position
            custom_init_joints=joint_pos.tolist(),  # Use dataset joint position
            enable_multi_cam_vis=False,
            record_raw_video=False,
            camera_serial_numbers=['215122255213', '832112070487', '746112060198'],
            camera_configs=configs,
            shm_manager=shm_manager
        ) as env:
            
            print("Waiting for cameras to initialize and robot to move...")
            time.sleep(5.0)
            
            if live:
                # Live comparison mode
                print("\n=== LIVE COMPARISON MODE ===")
                print("Press 'q' to quit, 's' to save current frame")
                print("Press 'b' to toggle alpha blend, 'd' to toggle diff view")
                
                blend_mode = False
                diff_mode = False
                save_count = 0
                
                while True:
                    obs = env.get_obs()
                    
                    # Get real camera images
                    real_front = obs['front_rgb'][-1]
                    real_wrist = obs['wrist_rgb'][-1]
                    real_side = obs.get('side_rgb', [None])[-1]
                    
                    # Convert to uint8
                    real_front = to_uint8(real_front)
                    real_wrist = to_uint8(real_wrist)
                    if real_side is not None:
                        real_side = to_uint8(real_side)
                    
                    # Resize sim to match real if needed
                    def resize_to_match(sim_img, real_img):
                        if sim_img.shape[:2] != real_img.shape[:2]:
                            return cv2.resize(sim_img, (real_img.shape[1], real_img.shape[0]))
                        return sim_img
                    
                    sim_front_r = resize_to_match(sim_front, real_front)
                    sim_wrist_r = resize_to_match(sim_wrist, real_wrist)
                    if sim_side is not None and real_side is not None:
                        sim_side_r = resize_to_match(sim_side, real_side)
                    else:
                        sim_side_r = None
                    
                    # Create comparison images
                    if blend_mode:
                        # Alpha blend (50% sim, 50% real)
                        alpha = 0.5
                        front_cmp = cv2.addWeighted(sim_front_r, alpha, real_front, 1-alpha, 0)
                        wrist_cmp = cv2.addWeighted(sim_wrist_r, alpha, real_wrist, 1-alpha, 0)
                        if sim_side_r is not None and real_side is not None:
                            side_cmp = cv2.addWeighted(sim_side_r, alpha, real_side, 1-alpha, 0)
                        else:
                            side_cmp = real_side if real_side is not None else np.zeros_like(real_front)
                    elif diff_mode:
                        # Absolute difference
                        front_cmp = cv2.absdiff(sim_front_r, real_front)
                        wrist_cmp = cv2.absdiff(sim_wrist_r, real_wrist)
                        if sim_side_r is not None and real_side is not None:
                            side_cmp = cv2.absdiff(sim_side_r, real_side)
                        else:
                            side_cmp = np.zeros_like(real_front)
                    else:
                        # Side by side (sim on top, real on bottom)
                        front_cmp = np.vstack([sim_front_r, real_front])
                        wrist_cmp = np.vstack([sim_wrist_r, real_wrist])
                        if sim_side_r is not None and real_side is not None:
                            side_cmp = np.vstack([sim_side_r, real_side])
                        else:
                            side_cmp = np.vstack([np.zeros_like(real_front), real_side]) if real_side is not None else None
                    
                    # Add labels
                    def add_labels(img, label1, label2):
                        h = img.shape[0] // 2
                        cv2.putText(img, label1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(img, label2, (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        return img
                    
                    if not blend_mode and not diff_mode:
                        front_cmp = add_labels(front_cmp, "SIM", "REAL")
                        wrist_cmp = add_labels(wrist_cmp, "SIM", "REAL")
                        if side_cmp is not None:
                            side_cmp = add_labels(side_cmp, "SIM", "REAL")
                    
                    # Display
                    cv2.imshow('Front Camera', front_cmp[..., ::-1])
                    cv2.imshow('Wrist Camera', wrist_cmp[..., ::-1])
                    if side_cmp is not None:
                        cv2.imshow('Side Camera', side_cmp[..., ::-1])
                    
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('b'):
                        blend_mode = not blend_mode
                        diff_mode = False
                        print(f"Blend mode: {blend_mode}")
                    elif key == ord('d'):
                        diff_mode = not diff_mode
                        blend_mode = False
                        print(f"Diff mode: {diff_mode}")
                    elif key == ord('s'):
                        # Save current comparison
                        cv2.imwrite(str(output_path / f'front_comparison_{save_count}.png'), front_cmp[..., ::-1])
                        cv2.imwrite(str(output_path / f'wrist_comparison_{save_count}.png'), wrist_cmp[..., ::-1])
                        if side_cmp is not None:
                            cv2.imwrite(str(output_path / f'side_comparison_{save_count}.png'), side_cmp[..., ::-1])
                        print(f"Saved comparison {save_count}")
                        save_count += 1
                
                cv2.destroyAllWindows()
            
            else:
                # Single frame comparison
                print("Capturing real camera images...")
                obs = env.get_obs()
                
                real_front = to_uint8(obs['front_rgb'][-1])
                real_wrist = to_uint8(obs['wrist_rgb'][-1])
                real_side = obs.get('side_rgb', [None])[-1]
                if real_side is not None:
                    real_side = to_uint8(real_side)
                
                print(f"Real image shapes: front={real_front.shape}, wrist={real_wrist.shape}",
                      f"side={real_side.shape if real_side is not None else 'N/A'}")
                
                # Create comparison figure
                import matplotlib.pyplot as plt
                
                def resize_to_match(sim_img, real_img):
                    if sim_img.shape[:2] != real_img.shape[:2]:
                        return cv2.resize(sim_img, (real_img.shape[1], real_img.shape[0]))
                    return sim_img
                
                n_cameras = 3 if (sim_side is not None and real_side is not None) else 2
                fig, axes = plt.subplots(3, n_cameras, figsize=(5*n_cameras, 12))
                
                cameras = [('Front', sim_front, real_front)]
                cameras.append(('Wrist', sim_wrist, real_wrist))
                if sim_side is not None and real_side is not None:
                    cameras.append(('Side', sim_side, real_side))
                
                for col, (name, sim_img, real_img) in enumerate(cameras):
                    sim_img_r = resize_to_match(sim_img, real_img)
                    
                    # Row 0: Sim
                    axes[0, col].imshow(sim_img_r)
                    axes[0, col].set_title(f'{name} - SIM')
                    axes[0, col].axis('off')
                    
                    # Row 1: Real
                    axes[1, col].imshow(real_img)
                    axes[1, col].set_title(f'{name} - REAL')
                    axes[1, col].axis('off')
                    
                    # Row 2: Overlay (alpha blend)
                    blend = cv2.addWeighted(sim_img_r, 0.5, real_img, 0.5, 0)
                    axes[2, col].imshow(blend)
                    axes[2, col].set_title(f'{name} - OVERLAY (50/50)')
                    axes[2, col].axis('off')
                
                plt.tight_layout()
                save_path = output_path / 'sim_real_comparison.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved comparison: {save_path}")
                
                # Also save difference images
                fig2, axes2 = plt.subplots(1, n_cameras, figsize=(5*n_cameras, 4))
                for col, (name, sim_img, real_img) in enumerate(cameras):
                    sim_img_r = resize_to_match(sim_img, real_img)
                    diff = cv2.absdiff(sim_img_r, real_img)
                    axes2[col].imshow(diff)
                    axes2[col].set_title(f'{name} - DIFF (|sim-real|)')
                    axes2[col].axis('off')
                
                plt.tight_layout()
                diff_path = output_path / 'sim_real_diff.png'
                plt.savefig(diff_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved diff image: {diff_path}")


if __name__ == '__main__':
    main()
