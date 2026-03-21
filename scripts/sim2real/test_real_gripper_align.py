"""
Test script for Robotiq 2F-85 gripper sim2real alignment.
Moves gripper through open/close cycles and logs position trajectory.

Companion to UWLab/scripts/test_sim_gripper_align.py for comparing dynamics.
Both scripts output JSON with the same schema so trajectories can be overlaid.

Position mapping:
  Real: 0 (open) to 255 (closed) integer
  Sim:  0.0 (open) to 0.785398 rad (closed) finger_joint angle
  Normalized: 0.0 (open) to 1.0 (closed)

Usage (from diffusion_policy repo root):
    python scripts/sim2real/test_real_gripper_align.py --robot_ip 192.168.1.10 -o gripper_real.json
    python scripts/sim2real/test_real_gripper_align.py --robot_ip 192.168.1.10 --speed 64 --num_cycles 5 -o gripper_real_slow.json
"""

import numpy as np
import time
import json
import click

from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


# Gripper position mapping
GRIPPER_SIM_CLOSED_RAD = 0.785398  # finger_joint closed angle in sim


def real_to_normalized(pos, min_pos, max_pos):
    """Real position [min, max] -> normalized [0, 1] (0=open, 1=closed)."""
    if max_pos == min_pos:
        return 0.0
    return float(pos - min_pos) / float(max_pos - min_pos)


def real_to_sim_rad(pos, min_pos, max_pos):
    """Real position -> sim finger_joint angle (radians)."""
    return real_to_normalized(pos, min_pos, max_pos) * GRIPPER_SIM_CLOSED_RAD


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="Robot IP address")
@click.option('--gripper_port', default=63352, type=int, help="Robotiq gripper port")
@click.option('--output_json', '-o', default=None, type=str, help="Output JSON file")
@click.option('--speed', '-s', default=128, type=int, help="Gripper speed 0-255 (default: max)")
@click.option('--force', '-f', default=128, type=int, help="Gripper force 0-255 (default: max)")
@click.option('--poll_rate', default=100, type=int, help="Position polling rate Hz")
@click.option('--hold_time', '-ht', default=2.0, type=float, help="Hold time per command (s)")
@click.option('--num_cycles', '-n', default=3, type=int, help="Number of open/close cycles")
@click.option('--init_arm', is_flag=True, default=False, help="Move arm to init position first")
@click.option('--joints_init_deg', default='0,-90,90,-90,-90,0', type=str,
              help="Initial arm joints in degrees (comma-separated)")
def main(robot_ip, gripper_port, output_json, speed, force, poll_rate, hold_time,
         num_cycles, init_arm, joints_init_deg):
    """Test gripper open/close dynamics for sim2real alignment."""

    print("\n" + "="*60)
    print("Gripper Alignment Test - REAL ROBOT")
    print("="*60)

    # Optionally move arm to safe position
    if init_arm:
        from rtde_control import RTDEControlInterface
        j_init = np.radians([float(x) for x in joints_init_deg.split(',')])
        print(f"Moving arm to init: {joints_init_deg} deg ...")
        rtde_c = RTDEControlInterface(robot_ip)
        ok = rtde_c.moveJ(j_init.tolist(), 1.05, 1.4)
        rtde_c.disconnect()
        if not ok:
            raise RuntimeError("moveJ to init position failed")
        print("Arm at init position.")

    # Connect gripper
    print(f"Connecting to gripper at {robot_ip}:{gripper_port} ...")
    gripper = RobotiqGripper()
    gripper.connect(robot_ip, gripper_port)
    gripper.activate(auto_calibrate=True)

    min_pos = gripper.get_open_position()
    max_pos = gripper.get_closed_position()
    print(f"Calibrated range: [{min_pos}, {max_pos}]")
    print(f"Speed: {speed}, Force: {force}")
    print(f"Poll rate: {poll_rate} Hz, Hold time: {hold_time}s")
    print(f"Cycles: {num_cycles}")

    # Build command sequence: start open, then (close -> open) × num_cycles
    commands = []
    commands.append((min_pos, "open_init"))
    for i in range(num_cycles):
        commands.append((max_pos, f"close_{i+1}"))
        commands.append((min_pos, f"open_{i+1}"))

    print(f"\nSequence ({len(commands)} commands):")
    for i, (pos, name) in enumerate(commands):
        norm = real_to_normalized(pos, min_pos, max_pos)
        print(f"  {i+1}. {name:15s}  target={pos:3d}  (norm={norm:.2f})")
    print("="*60 + "\n")

    # Logging structure (matches sim script output)
    trajectory = {
        "source": "real",
        "gripper_range_real": [int(min_pos), int(max_pos)],
        "gripper_range_sim_rad": [0.0, GRIPPER_SIM_CLOSED_RAD],
        "speed": speed,
        "force": force,
        "poll_rate_hz": poll_rate,
        "hold_time": hold_time,
        "num_cycles": num_cycles,
        "commands": [],
        "timestamps": [],
        "positions_raw": [],           # 0-255
        "positions_normalized": [],    # 0-1
        "positions_sim_rad": [],       # radians (for direct sim overlay)
    }

    dt = 1.0 / poll_rate
    global_start = time.time()

    # Ensure fully open before starting
    print("Ensuring gripper is open ...")
    gripper.move_and_wait_for_pos(min_pos, speed, force)
    time.sleep(0.5)

    print("Starting test sequence ...\n")

    for cmd_idx, (target_pos, name) in enumerate(commands):
        cmd_time = time.time() - global_start
        norm_target = real_to_normalized(target_pos, min_pos, max_pos)
        sim_target = real_to_sim_rad(target_pos, min_pos, max_pos)

        trajectory["commands"].append({
            "name": name,
            "target_raw": int(target_pos),
            "target_normalized": norm_target,
            "target_sim_rad": sim_target,
            "timestamp": cmd_time,
            "sample_index": len(trajectory["timestamps"]),
        })

        print(f"[{cmd_time:6.2f}s] Cmd {cmd_idx+1}/{len(commands)}: {name}  -> target={target_pos}")

        # Send non-blocking move command
        gripper.move(target_pos, speed, force)

        # Poll position for hold_time seconds
        hold_start = time.time()
        while time.time() - hold_start < hold_time:
            t = time.time() - global_start
            pos = gripper.get_current_position()

            trajectory["timestamps"].append(t)
            trajectory["positions_raw"].append(int(pos))
            trajectory["positions_normalized"].append(
                real_to_normalized(pos, min_pos, max_pos))
            trajectory["positions_sim_rad"].append(
                real_to_sim_rad(pos, min_pos, max_pos))

            time.sleep(dt)

        final_pos = gripper.get_current_position()
        error = abs(final_pos - target_pos)
        norm_final = real_to_normalized(final_pos, min_pos, max_pos)
        print(f"         Final: {final_pos}  (norm={norm_final:.3f}, error={error})")

    gripper.disconnect()

    n_samples = len(trajectory["timestamps"])
    duration = trajectory["timestamps"][-1] if n_samples > 0 else 0
    print(f"\n{'='*60}")
    print(f"Test complete. {n_samples} samples over {duration:.1f}s")
    print(f"{'='*60}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(trajectory, f, indent=2)
        print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()
