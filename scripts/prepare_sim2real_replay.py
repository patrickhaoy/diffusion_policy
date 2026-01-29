#!/usr/bin/env python3\

#Notes about the script:
    # make sure the output_dir is not the actual replay_buffer.zarr, it will be overwritten / deleted ;-;
    # dataset_dir should be the directory containing the videos and replay_buffer.zarr
    # output_zarr should be the directory to save the new replay_buffer.zarr
    # width and height should be the width and height of the images
    # camera_map should be the names of the cameras to use
    # repeat_first_action should be True if you want to use the first action as the last action for the first step of each episode

    #WHAT IT DOES: 
    #converts the mp4s to zarr with an existing function: real_data_to_replay_buffer
    #renames the cameras to the names in the camera_map
    #calculates the last actions for each episode and adds them to the replay_buffer.zarr
    
import argparse
import os
import sys
from typing import Dict

import numpy as np
import zarr
import cv2
from threadpoolctl import threadpool_limits

from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer

import math
import numcodecs

BLOSC = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
TARGET_CHUNK_BYTES = 64 * 1024 * 1024  # ~64MB

def compute_time_chunk_len(shape, dtype, target_bytes=TARGET_CHUNK_BYTES):
    # shape: (T, ...)  dtype: np.dtype
    if len(shape) == 1:
        per_t_bytes = dtype.itemsize
    else:
        per_t_bytes = dtype.itemsize * int(np.prod(shape[1:]))
    t_len = max(1, int(target_bytes // max(1, per_t_bytes)))
    return t_len


def compute_last_actions(actions: np.ndarray, episode_ends: np.ndarray,
                         repeat_first: bool = False) -> np.ndarray:
    """Compute per-timestep previous action within each episode.

    actions: (T, A)
    episode_ends: (E,), end indices (exclusive) in [1..T]
    repeat_first: if True, last_action[start] = actions[start], else zeros
    Returns last_action with same shape as actions.
    """
    T = actions.shape[0]
    last_action = np.zeros_like(actions)
    
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    print(episode_starts)
    for s, e in zip(episode_starts, episode_ends):
        if e - s <= 0:
            continue
        if repeat_first:
            last_action[s] = actions[s]
        if e - s > 1:
            last_action[s + 1:e] = actions[s:e - 1]
    return last_action


def copy_array(dst_group: zarr.Group, name: str, data: np.ndarray):
    if name in dst_group:
        del dst_group[name]
    t_chunk = compute_time_chunk_len(data.shape, data.dtype)
    if data.ndim == 1:
        chunks = (min(t_chunk, data.shape[0]),)
    elif data.ndim == 2:
        chunks = (min(t_chunk, data.shape[0]), data.shape[1])
    elif data.ndim == 3:
        # e.g., low-dim sequences with extra dims
        chunks = (min(t_chunk, data.shape[0]),) + data.shape[1:]
    elif data.ndim == 4:
        # images: (T, H, W, C)
        chunks = (min(t_chunk, data.shape[0]), data.shape[1], data.shape[2], data.shape[3])
    else:
        # fallback: time-chunk only
        chunks = (min(t_chunk, data.shape[0]),) + data.shape[1:]
    dst_group.create_dataset(name, data=data, chunks=chunks, dtype=data.dtype, compressor=BLOSC)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sim2real replay buffer: decode MP4s, rename cameras, add last_* actions.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path containing videos/ and replay_buffer.zarr/")
    parser.add_argument("--output_zarr", required=True,
                        help="Output zarr path (directory). Will be created/overwritten.")
    parser.add_argument("--width", type=int, default=224, help="Output image width")
    parser.add_argument("--height", type=int, default=224, help="Output image height")
    parser.add_argument("--camera_map", nargs=3, default=["front_rgb", "side_rgb", "wrist_rgb"],
                        help="Names for cameras 0,1,2 respectively.")
    parser.add_argument("--repeat_first_action", action="store_true",
                        help="Use first action as last_action for the first step of each episode (default zeros).")

    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_zarr = os.path.abspath(args.output_zarr)

    in_rb_path = os.path.join(dataset_dir, "replay_buffer.zarr")
    videos_dir = os.path.join(dataset_dir, "videos")
    if not os.path.isdir(in_rb_path):
        print(f"ERROR: {in_rb_path} not found.")
        sys.exit(1)
    if not os.path.isdir(videos_dir):
        print(f"ERROR: {videos_dir} not found.")
        sys.exit(1)

    # Prepare output store
    if os.path.exists(output_zarr):
        # Avoid stale content
        import shutil
        shutil.rmtree(output_zarr)
    out_store = zarr.DirectoryStore(output_zarr)

    # Limit threading to avoid oversubscription
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        # Decode MP4s and copy low-dim into a fresh replay buffer
        rb = real_data_to_replay_buffer(
            dataset_path=dataset_dir,
            out_store=out_store,
            out_resolutions=(args.width, args.height),
            lowdim_keys=["action", "arm_joint_pos", "stage", "timestamp"],
            image_keys=None  # infer cameras present
        )

    root = zarr.open(output_zarr, mode='a')
    data_grp: zarr.Group = root["data"]
    meta_grp: zarr.Group = root["meta"]

    # Rename camera_0/1/2 to desired keys by copying
    cam_names: Dict[int, str] = {0: args.camera_map[0], 1: args.camera_map[1], 2: args.camera_map[2]}
    for idx, out_name in cam_names.items():
        cam_key = f"camera_{idx}"
        if cam_key in data_grp:
            arr = data_grp[cam_key][:]
            copy_array(data_grp, out_name, arr)

    # Compute last_* actions
    actions = data_grp["action"][:]
    episode_ends = meta_grp["episode_ends"][:]
    last_action = compute_last_actions(actions, episode_ends, repeat_first=args.repeat_first_action)
    last_arm_action = last_action[:, :6].astype(np.float32)
    last_gripper_action = last_action[:, 6:7].astype(np.float32)
    copy_array(data_grp, "last_arm_action", last_arm_action)
    copy_array(data_grp, "last_gripper_action", last_gripper_action)

    # Move observations under data/obs to match Isaac Lab handler style
    obs_group = data_grp.require_group("obs")
    for key in [
        "front_rgb",
        "side_rgb",
        "wrist_rgb",
        "arm_joint_pos",
        "last_arm_action",
        "last_gripper_action",
    ]:
        if key in data_grp:
            copy_array(obs_group, key, data_grp[key][:])
            # remove top-level copy to avoid duplication
            del data_grp[key]

    # Optionally include stage/timestamp as observations too
    for key in ["stage", "timestamp"]:
        if key in data_grp:
            copy_array(obs_group, key, data_grp[key][:])
            del data_grp[key]

    # Rename action -> actions at top-level data group
    if "action" in data_grp:
        copy_array(data_grp, "actions", data_grp["action"][:])
        del data_grp["action"]

    # Synthesize rewards (zeros) and dones (1 at episode ends)
    episode_ends = meta_grp["episode_ends"][:]
    T = int(episode_ends[-1]) if episode_ends.size > 0 else 0
    rewards = np.zeros((T,), dtype=np.float32)
    dones = np.zeros((T,), dtype=np.uint8)
    for e in episode_ends:
        if e > 0:
            dones[e - 1] = 1
    copy_array(data_grp, "rewards", rewards)
    copy_array(data_grp, "dones", dones)

    # Clean up camera_* now that obs/* exist
    for i in [0, 1, 2]:
        cam_key = f"camera_{i}"
        if cam_key in data_grp:
            del data_grp[cam_key]

    # Summary
    print("Prepared replay buffer at:", output_zarr)
    print("data arrays:", sorted(list(data_grp.array_keys())))
    if "obs" in data_grp:
        print("obs arrays:", sorted(list(data_grp["obs"].array_keys())))
    print("meta entries:", sorted(list(meta_grp.keys())))


if __name__ == "__main__":
    main()


