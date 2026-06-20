"""Move the UR5e to the eval init pose and set the Robotiq gripper (no cameras).

For resetting the robot before/between sim2real alignment or live-overlay runs —
unlike 2_capture_align_refs.py this opens NO cameras, so it won't conflict with a
running live_board_overlay.py.

Usage (robodiff env):
    python scripts/sim2real/move_to_init.py                 # move to init pose, open gripper
    python scripts/sim2real/move_to_init.py --gripper close  # or none (leave gripper)
"""
import argparse
import os
import sys

import numpy as np

# Eval default init pose (degrees) — matches real_env.py / eval_real_robot_student.py.
INIT_POSE_DEG = [16.85, -79.74, 99.80, -114.68, -91.09, 20.43]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="192.168.1.10")
    ap.add_argument("--gripper", choices=["open", "close", "none"], default="open")
    ap.add_argument("--speed", type=float, default=0.5, help="moveJ joint speed (rad/s).")
    ap.add_argument("--accel", type=float, default=0.5, help="moveJ joint accel (rad/s^2).")
    args = ap.parse_args()

    import rtde_control

    q = (np.array(INIT_POSE_DEG) / 180.0 * np.pi).tolist()
    print(f"[move] moveJ {args.robot_ip} -> init pose (deg) {INIT_POSE_DEG}")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_c.moveJ(q, args.speed, args.accel)
    rtde_c.stopScript()

    if args.gripper != "none":
        repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper

        g = RobotiqGripper()
        g.connect(args.robot_ip, 63352)
        if not g.is_active():
            print("[move] activating gripper (one-time)...")
            g.activate()
        target = g.get_closed_position() if args.gripper == "close" else g.get_open_position()
        g.move_and_wait_for_pos(target, 128, 128)
        g.disconnect()
        print(f"[move] gripper -> {args.gripper}")
    print("[move] done.")


if __name__ == "__main__":
    main()
