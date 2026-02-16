#!/usr/bin/env python3
"""Capture a reference RGB image from a RealSense camera for sim2real alignment.

Usage:
    # Capture from all cameras:
    python capture_real_image.py

    # Capture from a specific camera:
    python capture_real_image.py --camera front
    python capture_real_image.py --camera side
    python capture_real_image.py --camera wrist

    # Custom output directory:
    python capture_real_image.py --output /path/to/output/

The images are saved as real_front.png, real_side.png, real_wrist.png.
"""

import os
import sys
import time
import json
import click
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.real_world.single_realsense import SingleRealsense

# Camera serial numbers (from demo_real_robot.py)
CAMERAS = {
    "front": {
        "serial": "215122255213",
        "config": "diffusion_policy/real_world/realsense_config/455_front.json",
    },
    "side": {
        "serial": "832112070487",
        "config": "diffusion_policy/real_world/realsense_config/435_side.json",
    },
    "wrist": {
        "serial": "746112060198",
        "config": "diffusion_policy/real_world/realsense_config/415_wrist.json",
    },
}


def capture_camera(shm_manager, name, serial, config_path, output_dir, settle_time=2.0):
    """Capture a single frame from one RealSense camera."""
    config = json.load(open(os.path.join(ROOT_DIR, config_path), "r"))

    print(f"[{name}] Opening camera {serial}...")
    with SingleRealsense(
        shm_manager=shm_manager,
        serial_number=serial,
        resolution=(640, 480),
        capture_fps=30,
        enable_color=True,
        advanced_mode_config=config,
    ) as rs:
        # Let auto-exposure / white-balance settle
        print(f"[{name}] Waiting {settle_time:.1f}s for auto-exposure to settle...")
        time.sleep(settle_time)

        # Grab frame
        data = rs.get(out=None)
        rgb = data["color"]  # (H, W, 3) uint8, BGR from OpenCV

        out_path = os.path.join(output_dir, f"real_{name}.png")
        cv2.imwrite(out_path, rgb)  # cv2 expects BGR
        print(f"[{name}] Saved {out_path}  shape={rgb.shape}")

        return rgb


@click.command()
@click.option("--camera", "-c", type=click.Choice(["front", "side", "wrist", "all"]),
              default="all", help="Which camera to capture (default: all)")
@click.option("--output", "-o", default=".", help="Output directory for saved images")
@click.option("--settle_time", "-t", default=2.0, type=float,
              help="Seconds to wait for auto-exposure before capturing")
@click.option("--preview", "-p", is_flag=True, default=False,
              help="Show a preview window before saving")
def main(camera, output, settle_time, preview):
    """Capture reference RGB images from RealSense cameras."""
    os.makedirs(output, exist_ok=True)

    cameras_to_capture = list(CAMERAS.keys()) if camera == "all" else [camera]

    with SharedMemoryManager() as shm_manager:
        for cam_name in cameras_to_capture:
            info = CAMERAS[cam_name]
            rgb = capture_camera(
                shm_manager, cam_name, info["serial"], info["config"],
                output, settle_time,
            )

            if preview:
                cv2.imshow(f"real_{cam_name}", rgb)
                print(f"[{cam_name}] Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print(f"\nDone! Images saved to {os.path.abspath(output)}/")
    print("Use with align_cameras.py:")
    for cam_name in cameras_to_capture:
        sim_cam = f"{cam_name}_camera"
        img_path = os.path.join(os.path.abspath(output), f"real_{cam_name}.png")
        print(f"  python scripts/sim2real/align_cameras.py --enable_cameras "
              f"--camera {sim_cam} --real_image {img_path}")


if __name__ == "__main__":
    main()
