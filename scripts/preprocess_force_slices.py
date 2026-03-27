"""
Preprocess a zarr dataset to create sliced force arrays for ablation conditions:
  - end_effector_force: end_effector_wrench[..., :3]  (force only, no torque)
  - right_finger_force_y: right_finger_force[..., 1:2] (squeeze axis only)
  - left_finger_force_y: left_finger_force[..., 1:2]   (squeeze axis only)

Usage:
    python scripts/preprocess_force_slices.py /path/to/dataset.zarr
"""

import argparse
import zarr
import numpy as np


SLICES = [
    ("end_effector_force", "end_effector_wrench", (slice(None), slice(0, 3))),
    ("right_finger_force_y", "right_finger_force", (slice(None), slice(1, 2))),
    ("left_finger_force_y", "left_finger_force", (slice(None), slice(1, 2))),
]


def preprocess(zarr_path: str):
    root = zarr.open(zarr_path, mode="r+")
    obs = root["data"]["obs"]

    for dst_key, src_key, idx in SLICES:
        if dst_key in obs:
            print(f"  {dst_key} already exists, skipping")
            continue
        if src_key not in obs:
            print(f"  WARNING: source {src_key} not found, skipping {dst_key}")
            continue

        src = obs[src_key][:]
        sliced = src[idx]
        obs.create_array(dst_key, data=sliced, overwrite=False)
        print(f"  Created {dst_key} {sliced.shape} from {src_key}{list(idx)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", help="Path to zarr dataset root")
    args = parser.parse_args()
    preprocess(args.zarr_path)
