from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import argparse
import os
from typing import Tuple, Any, TYPE_CHECKING, cast

import numpy as np
from scipy.spatial.transform import Rotation as R


def _compute_ee_actions(target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """Compute relative end-effector actions (delta pose) from current to target pose."""
    # Position delta: target_pos - current_pos
    delta_pos = target_pose[:3] - current_pose[:3]

    # Rotation delta: compute rotation that transforms current to target
    current_rot = R.from_rotvec(current_pose[3:])
    target_rot = R.from_rotvec(target_pose[3:])

    # delta_rot such that delta_rot * current_rot  = target_rot
    delta_rot = target_rot * current_rot.inv()
    delta_rotvec = delta_rot.as_rotvec()

    # Transform delta from policy frame to robot base frame
    # Policy operates in (1,0,0,0) frame, robot base is (0,0,0,1) - 180 deg rotation
    policy_to_robot_rot = R.from_quat([0, 0, 1, 0])  # 180 deg around Z
    transformed_delta_pos = policy_to_robot_rot.apply(delta_pos)
    transformed_delta_rot = policy_to_robot_rot.apply(delta_rotvec)

    return np.concatenate([transformed_delta_pos, transformed_delta_rot]).astype(float)


def _open_zarr(root_path: str):
    try:
        import zarr  # imported here so the module is only required when running this script
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: zarr. Install with `pip install zarr numcodecs` and rerun."
        ) from e
    return zarr.open(root_path, mode="a")


def _ensure_array(group: Any, name: str, shape: Tuple[int, int], chunks: Tuple[int, int], dtype=np.float64):
    if name in group:
        arr = group[name]
        # Basic compatibility check
        if tuple(arr.shape) != tuple(shape) or tuple(arr.chunks) != tuple(chunks):
            raise ValueError(
                f"Existing array {name} has shape {arr.shape}, chunks {arr.chunks},"
                f" expected shape {shape}, chunks {chunks}."
            )
        return arr
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def relabel_dataset(
    zarr_path: str,
    robot_ip: str = "192.168.1.10",
    tcp_offset: float = 0.1495,
    actions_key: str = "actions",
    joints_key: str = "obs/arm_joint_pos",
    batch_size: int = 2048,
):
    """Loop over the zarr dataset and add data/ee_action and data/end_effector_pose.

    - Uses existing end-effector pose and action calculations as provided above.
    - Reads joint positions from data/{joints_key} and actions from data/{actions_key}.
    - Writes arrays to data/ee_action and data/end_effector_pose matching store chunking.
    """

    root: Any = _open_zarr(zarr_path)

    if "data" not in root:
        raise FileNotFoundError(f"No 'data' group found in {zarr_path}")
    data_grp = root["data"]

    # Resolve input arrays
    if actions_key not in data_grp:
        raise FileNotFoundError(f"Missing data/{actions_key} in {zarr_path}")
    # Hint types for static checkers; at runtime we treat them as zarr arrays
    actions_arr: Any = data_grp[actions_key]

    if joints_key not in data_grp:
        raise FileNotFoundError(f"Missing data/{joints_key} in {zarr_path}")
    joints_arr: Any = data_grp[joints_key]
    
    # Check for last_arm_action array
    last_arm_action_key = "obs/last_arm_action"
    if last_arm_action_key not in data_grp:
        raise FileNotFoundError(f"Missing data/{last_arm_action_key} in {zarr_path}")
    last_arm_action_arr: Any = data_grp[last_arm_action_key]

    n = int(actions_arr.shape[0])
    if int(joints_arr.shape[0]) != n:
        raise ValueError(
            f"Length mismatch: actions {actions_arr.shape[0]} vs joints {joints_arr.shape[0]}"
        )
    if int(last_arm_action_arr.shape[0]) != n:
        raise ValueError(
            f"Length mismatch: actions {actions_arr.shape[0]} vs last_arm_action {last_arm_action_arr.shape[0]}"
        )
    
    # Get episode ends for checking episode boundaries
    if 'meta' not in root or 'episode_ends' not in root['meta']:
        raise FileNotFoundError(f"Missing meta/episode_ends in {zarr_path}")
    episode_ends = np.array(root['meta']['episode_ends'][:])
    
    # Create episode ends mask for efficient lookup
    episode_ends_set = set(episode_ends)

    # Prepare outputs - ee_action will be 7D (6D end-effector delta + 1D gripper)
    ee_action_shape = (n, 7)  # 6D EE delta + 1D gripper
    ee_pose_shape = (n, 6)    # 6D pose only
    # Use the input chunk length (axis 0) for compatibility; fall back to a reasonable batch size
    in_chunks = getattr(actions_arr, "chunks", None)
    chunk_len = int(in_chunks[0]) if in_chunks is not None else min(batch_size, n)
    ee_action_chunks = (chunk_len, 7)
    ee_pose_chunks = (chunk_len, 6)

    ee_action_arr = _ensure_array(data_grp, "actions", ee_action_shape, ee_action_chunks, dtype=np.float64)
    ee_pose_arr = _ensure_array(data_grp, "end_effector_pose", ee_pose_shape, ee_pose_chunks, dtype=np.float64)

    # RTDE interfaces (keep calculation path unchanged)
    rtde_c = RTDEControlInterface(hostname=robot_ip)
    rtde_r = RTDEReceiveInterface(hostname=robot_ip)
    try:
        # Store previous ee_delta for last_arm_action computation
        prev_ee_delta = None
        
        # Loop in batches to avoid excessive memory usage
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            # Read slices
            joints_batch = joints_arr[start:stop]  # shape [B, 6]
            actions_batch = actions_arr[start:stop]  # shape [B, ...]

            # Compute end-effector pose for current joint positions
            ee_pose_batch = np.empty((stop - start, 6), dtype=np.float64)
            target_pose_batch = np.empty((stop - start, 6), dtype=np.float64)
            gripper_actions_batch = np.empty((stop - start, 1), dtype=np.float64)

            for i in range(stop - start):
                q = np.asarray(joints_batch[i], dtype=float)
                # Note: action vector includes gripper in last dimension
                a = np.asarray(actions_batch[i], dtype=float)
                assert len(a.shape) == 1 and a.shape[0] == 7, "Action vector must have 7 dimensions"
                a_fk = a[:6]  # First 6 for forward kinematics
                gripper_actions_batch[i, 0] = a[-1]

                # Calculations exactly as sketched above
                current_pose = rtde_c.getForwardKinematics(q.tolist(), [0, 0, tcp_offset, 0, 0, 0])
                target_pose = rtde_c.getForwardKinematics(a_fk.tolist(), [0, 0, tcp_offset, 0, 0, 0])

                ee_pose_batch[i] = np.asarray(current_pose, dtype=float)
                target_pose_batch[i] = np.asarray(target_pose, dtype=float)

            # Compute ee_action via provided helper (per-sample to preserve exact math path)
            ee_delta_batch = np.empty((stop - start, 6), dtype=np.float64)
            for i in range(stop - start):
                ee_delta_batch[i] = _compute_ee_actions(target_pose_batch[i], ee_pose_batch[i])
            
            # Concatenate ee_delta (6D) with gripper actions (1D) to create 7D ee_action
            ee_action_batch = np.concatenate([ee_delta_batch, gripper_actions_batch], axis=1)
            
            # Update last_arm_action with first 6 dimensions of ee_action from previous timestep
            # But set to zeros if at episode boundary (beginning of episode)
            last_arm_action_batch = np.empty((stop - start, last_arm_action_arr.shape[1]), dtype=np.float64)
            for i in range(stop - start):
                global_idx = start + i
                prev_global_idx = global_idx - 1
                
                # Check if current timestep is at the beginning of an episode
                is_episode_start = (global_idx == 0) or (prev_global_idx in episode_ends_set)
                
                if is_episode_start:
                    # At episode start, set last_arm_action to zeros (no previous action from this episode)
                    last_arm_action_batch[i] = np.zeros(last_arm_action_arr.shape[1])
                else:
                    # Use first 6 dimensions of ee_action from previous timestep (i-1)
                    if i == 0:
                        # Use prev_ee_delta from previous batch (stored from last iteration)
                        current_prev_ee_delta = prev_ee_delta
                    else:
                        # Previous timestep is in current batch
                        current_prev_ee_delta = ee_delta_batch[i-1]
                    
                    # Set last_arm_action to first 6 dimensions of previous ee_action
                    if last_arm_action_arr.shape[1] >= 6:
                        last_arm_action_batch[i, :6] = current_prev_ee_delta[:6]
                        # Fill remaining dimensions with zeros if last_arm_action is longer than 6
                        if last_arm_action_arr.shape[1] > 6:
                            last_arm_action_batch[i, 6:] = 0.0
                    else:
                        # If last_arm_action is shorter than 6, truncate
                        last_arm_action_batch[i] = current_prev_ee_delta[:last_arm_action_arr.shape[1]]

            # Write results
            ee_pose_arr[start:stop] = ee_pose_batch
            #TODO: hack
            ee_action_arr[start:stop] = ee_action_batch
            ee_action_arr[start:stop, :6] = ee_action_arr[start:stop, :6] * 10.0
            last_arm_action_arr[start:stop] = last_arm_action_batch 
            last_arm_action_arr[start:stop, :6] = last_arm_action_arr[start:stop, :6] * 10.0
            
            # Store the last ee_delta from this batch for the next batch
            if len(ee_delta_batch) > 0:
                prev_ee_delta = ee_delta_batch[-1].copy()

            print(f"Wrote {start}:{stop} / {n}")
    finally:
        # Ensure clean shutdown
        try:
            rtde_c.disconnect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Relabel dataset with ee_pose and ee_action")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "demo",
            "replay_buffer_converted_2.zarr",
        ),
        help="Path to the source zarr store",
    )
    parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="UR robot IP")
    parser.add_argument("--tcp_offset", type=float, default=0.1495, help="TCP offset (m)")
    parser.add_argument(
        "--actions_key",
        type=str,
        default="actions",
        help="Key under data/ for actions (e.g., 'actions' or 'action')",
    )
    parser.add_argument(
        "--joints_key",
        type=str,
        default="obs/arm_joint_pos",
        help="Key under data/ for joint positions",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing")

    args = parser.parse_args()
    relabel_dataset(
        zarr_path=args.zarr_path,
        robot_ip=args.robot_ip,
        tcp_offset=args.tcp_offset,
        actions_key=args.actions_key,
        joints_key=args.joints_key,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
