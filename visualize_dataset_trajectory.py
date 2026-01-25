"""
Visualize images from a zarr dataset trajectory.
"""
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import click


@click.command()
@click.option('--dataset', '-d', required=True, 
              help='Path to zarr dataset directory')
@click.option('--episode', '-e', default=0, type=int,
              help='Episode index to visualize')
@click.option('--step_interval', '-s', default=10, type=int,
              help='Show every N-th frame')
@click.option('--save', is_flag=True, default=False,
              help='Save images to disk')
def main(dataset, episode, step_interval, save):
    # Load zarr dataset
    zarr_path = os.path.join(dataset, 'rgb0.zarr')
    if not os.path.exists(zarr_path):
        zarr_path = dataset
    
    print(f"Loading dataset from: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    print(f"Dataset has {n_episodes} episodes")
    
    if episode >= n_episodes:
        print(f"Error: Episode {episode} does not exist")
        return
    
    # Get episode slice
    start_idx = 0 if episode == 0 else episode_ends[episode - 1]
    end_idx = episode_ends[episode]
    episode_length = end_idx - start_idx
    print(f"Episode {episode}: {episode_length} steps")
    
    # Load images
    front_rgb = root['data/obs/front_rgb'][start_idx:end_idx]
    side_rgb = root['data/obs/side_rgb'][start_idx:end_idx]
    wrist_rgb = root['data/obs/wrist_rgb'][start_idx:end_idx]
    
    print(f"Front RGB shape: {front_rgb.shape}")
    print(f"Side RGB shape: {side_rgb.shape}")
    print(f"Wrist RGB shape: {wrist_rgb.shape}")
    
    # Select frames to display
    frame_indices = list(range(0, episode_length, step_interval))
    n_frames = len(frame_indices)
    
    print(f"\nDisplaying {n_frames} frames (every {step_interval} steps)")
    
    # Create figure
    fig, axes = plt.subplots(n_frames, 3, figsize=(12, 4 * n_frames))
    if n_frames == 1:
        axes = axes[None, :]
    
    for i, idx in enumerate(frame_indices):
        # Front camera
        axes[i, 0].imshow(front_rgb[idx])
        axes[i, 0].set_title(f'Front RGB - Step {idx}')
        axes[i, 0].axis('off')
        
        # Side camera
        axes[i, 1].imshow(side_rgb[idx])
        axes[i, 1].set_title(f'Side RGB - Step {idx}')
        axes[i, 1].axis('off')
        
        # Wrist camera
        axes[i, 2].imshow(wrist_rgb[idx])
        axes[i, 2].set_title(f'Wrist RGB - Step {idx}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save:
        save_path = f'trajectory_ep{episode}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
