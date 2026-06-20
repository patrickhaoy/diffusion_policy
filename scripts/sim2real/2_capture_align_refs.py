"""Capture per-camera real reference images at a known joint pose for sim2real
camera alignment.

Step 2 of the sim2real workflow (after 1_camera_get_rgb.py). Produces one RGB
still per camera (front / side / wrist) tied to the exact arm joint angles, so
the images can be overlaid against the simulated camera render in:

    UWLab/scripts/sim2real/align_cameras.py

The wrist camera is mounted on the gripper, so its view depends entirely on the
arm pose -- the reference photo and the sim render MUST be at the same joint
configuration. This script reads (and optionally commands) that pose and prints
the angles in degrees, ready to paste into align_cameras.py's --joint_angles.

Run from scripts/sim2real/ (so the `perception` package is importable):

    cd scripts/sim2real
    # default: move arm to the eval init pose, open the gripper, then capture:
    python 2_capture_align_refs.py
    # capture at the CURRENT arm pose without moving:
    python 2_capture_align_refs.py --no_move
"""

import argparse
import os

import numpy as np
from PIL import Image

from perception.multi_camera_wrapper import MultiCameraWrapper

# Serial -> camera view. Must match real_env.py's positional mapping. Note that
# MultiCameraWrapper sorts cameras by serial string, so we label by serial here
# rather than by index to avoid mixing up the views.
SERIAL_TO_VIEW = {
    "215122255213": "front",
    "832112070487": "side",
    "746112060198": "wrist",
}

# Eval default init pose (degrees), from real_env.py:176.
INIT_POSE_DEG = [16.85, -79.74, 99.80, -114.68, -91.09, 20.43]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot_ip", default="192.168.1.10",
                        help="UR5e IP address.")
    parser.add_argument("--move", dest="move", action="store_true", default=True,
                        help="Command the arm to the eval init pose before capturing "
                             "(prompts for confirmation). ON by default.")
    parser.add_argument("--no_move", dest="move", action="store_false",
                        help="Skip the moveJ and capture at the CURRENT arm pose.")
    parser.add_argument("--out_dir", default="align_refs",
                        help="Directory to save reference PNGs.")
    parser.add_argument("--high_res", action="store_true",
                        help="Capture at 1280x720 instead of 640x480.")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="moveJ joint speed (rad/s) when --move.")
    parser.add_argument("--accel", type=float, default=0.5,
                        help="moveJ joint acceleration (rad/s^2) when --move.")
    parser.add_argument("--gripper", choices=["open", "close", "none"], default="open",
                        help="Set the Robotiq gripper before capturing, to match the sim "
                             "overlay's --gripper. 'none' leaves it untouched. Default: open.")
    parser.add_argument("--gripper_speed", type=int, default=128, help="Robotiq move speed (0-255).")
    parser.add_argument("--gripper_force", type=int, default=128, help="Robotiq move force (0-255).")
    args = parser.parse_args()

    import rtde_receive
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    if args.move:
        import rtde_control
        init_q = (np.array(INIT_POSE_DEG) / 180.0 * np.pi).tolist()
        print(f"About to moveJ the arm to the init pose (deg): {INIT_POSE_DEG}")
        resp = input("Make sure the workspace is clear. Proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return
        rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        rtde_c.moveJ(init_q, args.speed, args.accel)
        rtde_c.stopScript()

    # Set the gripper state so the wrist-camera finger pose matches the sim overlay's
    # --gripper. Same path the eval uses: RobotiqGripper socket on robot_ip:63352,
    # move(open/closed_position, 128, 128).
    if args.gripper != "none":
        import sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper
        gripper = RobotiqGripper()
        gripper.connect(args.robot_ip, 63352)
        if not gripper.is_active():
            print("Activating gripper (one-time calibration)...")
            gripper.activate()
        target = gripper.get_closed_position() if args.gripper == "close" else gripper.get_open_position()
        print(f"Moving gripper -> {args.gripper} (pos={target})...")
        gripper.move_and_wait_for_pos(target, args.gripper_speed, args.gripper_force)
        gripper.disconnect()

    # Read the ACTUAL joint angles at capture time (radians -> degrees).
    actual_q = np.array(rtde_r.getActualQ())
    actual_deg = actual_q * 180.0 / np.pi

    # Capture one frame per camera.
    cams = MultiCameraWrapper(rgb=True, depth=False, ir=False,
                              high_res_rgb=args.high_res, align="rgb", type="realsense")
    os.makedirs(args.out_dir, exist_ok=True)

    print()
    for camera in cams._all_cameras:
        serial = camera._serial_number
        view = SERIAL_TO_VIEW.get(serial)
        rgb = camera.read_camera()["rgb"]  # already RGB
        if view is None:
            print(f"  WARNING: unknown camera serial {serial}; saving as {serial}.png")
            view = serial
        out_path = os.path.join(args.out_dir, f"{view}_real.png")
        Image.fromarray(rgb).save(out_path)
        print(f"  saved {view:5s} ({serial})  {rgb.shape}  -> {out_path}")
    cams.disable_cameras()

    abs_out = os.path.abspath(args.out_dir)
    deg_str = " ".join(f"{d:.2f}" for d in actual_deg)
    print("\n" + "=" * 70)
    print(f"Reference images saved to: {abs_out}")
    print(f"Captured at joint angles (deg): {deg_str}")
    sim_gripper = "" if args.gripper == "none" else f" --gripper {args.gripper}"
    print("\nRun the sim overlay (env_isaaclab, from ~/research/IsaacLab_factory):")
    print(
        f"  python scripts/sim2real/align_cameras_factory.py --enable_cameras --headless --device cuda:1{sim_gripper} \\\n"
        f"      --joint_angles {deg_str} \\\n"
        f"      --side_real  {abs_out}/side_real.png \\\n"
        f"      --wrist_real {abs_out}/wrist_real.png \\\n"
        f"      --out_dir /tmp/align_factory --blend 0.5"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
