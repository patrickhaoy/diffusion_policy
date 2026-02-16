"""
Replay Cartesian delta actions from a zarr dataset on real UR5e using OSC torque control.
Uses calibrated FK + 180° Z rotation to match simulation frame.
Same controller as test_real_ur5e_osc_cube.py.

Usage:
    python replay_dataset_actions.py -d /path/to/state.zarr -e 0 --robot_ip 192.168.1.10
    python replay_dataset_actions.py -d /path/to/state.zarr -e 0 --dry_run
"""
import os
import time
import click
import numpy as np
import zarr
import torch
from pathlib import Path

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Import OSC controller and kinematics (calibrated FK with 180° Z rotation)
from diffusion_policy.real_world.ur5e_kinematics import (
    get_ee_pose, compute_jacobian_calibrated, OperationalSpaceController,
    quat_to_axis_angle, axis_angle_to_quat, PAYLOAD_MASS, PAYLOAD_COG,
)
from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


# ============================================================================
# Plotting
# ============================================================================

def plot_ee_comparison(sim_ee, real_ee, times, output_path, episode):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, i]
        ax.plot(times, sim_ee[:, i] * 1000, 'b-', label='Sim', linewidth=2)
        ax.plot(times, real_ee[:, i] * 1000, 'r-', label='Real', linewidth=2)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (mm)')
        ax.set_title(f'EE Position - {label}'); ax.legend(); ax.grid(True, alpha=0.3)

    for i, label in enumerate(['RX', 'RY', 'RZ']):
        ax = axes[1, i]
        ax.plot(times, np.degrees(sim_ee[:, i+3]), 'b-', label='Sim', linewidth=2)
        ax.plot(times, np.degrees(real_ee[:, i+3]), 'r-', label='Real', linewidth=2)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Rotation (deg)')
        ax.set_title(f'EE Rotation - {label}'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'End-Effector Pose Comparison (Episode {episode})', fontsize=16)
    plt.tight_layout()
    path = output_path / f'ee_comparison_ep{episode}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


def plot_joint_comparison(sim_joints, real_joints, times, output_path, episode):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    fig, axes = plt.subplots(6, 2, figsize=(14, 18))

    for j in range(6):
        ax = axes[j, 0]
        ax.plot(times, np.rad2deg(sim_joints[:, j]), 'b-', label='Sim', linewidth=2)
        ax.plot(times, np.rad2deg(real_joints[:, j]), 'r-', label='Real', linewidth=2)
        ax.set_ylabel('Angle (deg)'); ax.set_title(f'{names[j]}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[j, 1]
        err = np.rad2deg(real_joints[:, j] - sim_joints[:, j])
        ax.plot(times, err, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(times, err, alpha=0.3, color='green')
        ax.set_ylabel('Error (deg)'); ax.set_title(f'{names[j]} - Error')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    plt.suptitle(f'Joint Position Comparison (Episode {episode})', fontsize=16)
    plt.tight_layout()
    path = output_path / f'joint_comparison_ep{episode}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

@click.command()
@click.option('--dataset', '-d', required=True, help='Path to zarr dataset')
@click.option('--robot_ip', default='192.168.1.10', help='Robot IP address')
@click.option('--episode', '-e', default=0, type=int, help='Episode index')
@click.option('--frequency', '-f', default=10.0, type=float, help='Policy frequency (Hz)')
@click.option('--action_scale', default='0.02,0.02,0.02,0.02,0.02,0.2', type=str,
              help='Per-dim action scale, comma-separated (6 values)')
@click.option('--kp_pos', default=1000.0, type=float, help='Position stiffness')
@click.option('--kp_rot', default=50.0, type=float, help='Rotation stiffness')
@click.option('--damping_ratio', default=1.0, type=float, help='Damping ratio')
@click.option('--absolute', is_flag=True, default=False,
              help='Use absolute sim EE poses instead of delta actions (debug mode)')
@click.option('--dry_run', is_flag=True, default=False, help='Print stats only')
@click.option('--output', '-o', default='replay_output', help='Output directory')
def main(dataset, robot_ip, episode, frequency, action_scale, kp_pos, kp_rot,
         damping_ratio, absolute, dry_run, output):

    # Parse action scales
    action_scale = np.array([float(x) for x in action_scale.split(',')])
    assert len(action_scale) == 6
    print(f"Action scales: {action_scale}")

    # ---- Load zarr ----
    zarr_path = dataset
    if not os.path.exists(zarr_path):
        print(f"Error: dataset not found at {zarr_path}"); return
    root = zarr.open(zarr_path, mode='r')

    # Print zarr structure for debugging
    print(f"Zarr keys: {list(root.keys())}")
    if 'data' in root:
        print(f"  data/: {list(root['data'].keys())}")
        if 'obs' in root['data']:
            print(f"  data/obs/: {list(root['data/obs'].keys())}")

    episode_ends = root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    print(f"Dataset: {n_episodes} episodes, ends={episode_ends}")

    if episode >= n_episodes:
        print(f"Error: episode {episode} >= {n_episodes}"); return

    s = 0 if episode == 0 else int(episode_ends[episode - 1])
    e = int(episode_ends[episode])
    ep_len = e - s
    print(f"Episode {episode}: {ep_len} steps [{s}:{e})")

    actions = np.array(root['data/actions'][s:e])
    assert actions.shape[1] >= 7, (
        f"Expected actions dim >= 7 (arm=6 + gripper=1), got {actions.shape[1]}")
    sim_joints = np.array(root['data/obs/arm_joint_pos'][s:e])
    assert sim_joints.shape[1] == 6, f"Expected 6 joints, got {sim_joints.shape[1]}"
    sim_ee = np.array(root['data/obs/end_effector_pose'][s:e]) \
        if 'end_effector_pose' in root['data/obs'] else None

    init_joints = sim_joints[0]
    print(f"Init joints (deg): {np.rad2deg(init_joints)}")
    print(f"Actions: shape={actions.shape}, range=[{actions.min():.3f}, {actions.max():.3f}]")

    raw_arm = actions[:, :6]
    n_exceed = np.sum(np.any(np.abs(raw_arm) > 1.0, axis=1))
    print(f"Raw arm actions: {n_exceed}/{len(raw_arm)} steps exceed [-1,1] "
          f"(abs_max={np.abs(raw_arm).max():.2f}) -- NO clipping applied (matches sim)")
    scaled = raw_arm * action_scale
    print(f"Scaled pos deltas (mm): mean_abs={np.mean(np.abs(scaled[:,:3]))*1000:.2f}, "
          f"max={np.max(np.abs(scaled[:,:3]))*1000:.2f}")
    print(f"Scaled rot deltas (deg): mean_abs={np.mean(np.abs(np.degrees(scaled[:,3:]))):.2f}, "
          f"max={np.max(np.abs(np.degrees(scaled[:,3:]))):.2f}")
    if actions.shape[1] > 6:
        print(f"Gripper actions: min={actions[:,6].min():.3f}, max={actions[:,6].max():.3f}")

    if sim_ee is not None:
        ee0 = get_ee_pose(init_joints)
        print(f"Calibrated FK EE t=0: [{ee0[0][0]*1000:.1f}, {ee0[0][1]*1000:.1f}, {ee0[0][2]*1000:.1f}] mm")
        print(f"Sim EE t=0:           [{sim_ee[0,0]*1000:.1f}, {sim_ee[0,1]*1000:.1f}, {sim_ee[0,2]*1000:.1f}] mm")

    if absolute and sim_ee is None:
        print("Error: --absolute requires end_effector_pose in zarr"); return

    if absolute:
        print(f"\n*** ABSOLUTE MODE: tracking sim EE poses directly (ignoring actions) ***")

    if dry_run:
        print("\n[DRY RUN] Exiting."); return

    # ---- Setup ----
    control_freq = 500
    steps_per_action = int(control_freq / frequency)
    dt = 1.0 / frequency
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    osc = OperationalSpaceController(
        motion_stiffness=(kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot),
        motion_damping_ratio=(damping_ratio,) * 6,
    )
    print(f"\nOSC: Kp_pos={kp_pos}, Kp_rot={kp_rot}, damping={damping_ratio}")
    print(f"Policy: {frequency} Hz | Control: {control_freq} Hz "
          f"({steps_per_action} inner steps per action)")

    # ---- Connect ----
    rtde_c = RTDEControlInterface(
        robot_ip, control_freq,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
    rtde_r = RTDEReceiveInterface(robot_ip, control_freq)
    rtde_c.setPayload(PAYLOAD_MASS, PAYLOAD_COG)

    # Gripper (Robotiq 2F-85 via socket)
    gripper = RobotiqGripper()
    gripper.connect(robot_ip, 63352)
    gripper.activate()
    # Sim always closes gripper on first step after reset; start closed to match
    gripper.move(gripper.get_closed_position(), 128, 128)
    gripper_state = 'closed'
    print("Gripper connected, activated, and closed (matching sim initial state).")

    real_joint_log = []
    real_ee_log = []

    try:
        # Move to initial joints
        print("Moving to initial joint position...")
        ok = rtde_c.moveJ(init_joints.tolist(), 1.05, 1.4)
        if not ok:
            raise RuntimeError("moveJ failed")
        time.sleep(1.0)

        curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
        ee_pos, ee_quat = get_ee_pose(curr_joints)
        print(f"Real EE after init: [{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, {ee_pos[2]*1000:.1f}] mm")

        input("\nPress Enter to start replay...")

        # Hold current pose until first action
        osc.set_command(np.zeros(6), ee_pos, ee_quat)

        print(f"\nReplaying {ep_len} actions at {frequency} Hz...")

        for i in range(len(actions)):
            # ---- Record state at policy tick ----
            curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
            ee_pos, ee_quat = get_ee_pose(curr_joints)
            real_joint_log.append(curr_joints.copy())
            real_ee_log.append(np.concatenate([ee_pos, quat_to_axis_angle(ee_quat)]))

            # ---- Set OSC target ----
            if absolute:
                # Directly track sim EE pose (pos + axis-angle -> pos + quat)
                target_pos = sim_ee[i, :3].copy()
                target_quat = axis_angle_to_quat(sim_ee[i, 3:6])
                osc.set_target(target_pos, target_quat)
                delta = target_pos - ee_pos  # for logging only
            else:
                # Delta mode: scale raw actions (NO clip -- sim has input_clip=None)
                raw_arm = actions[i, :6]
                if np.any(np.abs(raw_arm) > 50):
                    print(f"  WARNING step {i}: extreme raw action {raw_arm}, max={np.abs(raw_arm).max():.1f}")
                delta = raw_arm * action_scale
                osc.set_command(delta, ee_pos, ee_quat)

            # ---- Gripper: <=0 = close, >0 = open (matches sim BinaryJointPositionAction) ----
            if actions.shape[1] > 6:
                gripper_cmd = actions[i, 6]
                if gripper_cmd <= 0 and gripper_state == 'open':
                    gripper.move(gripper.get_closed_position(), 128, 128)
                    gripper_state = 'closed'
                elif gripper_cmd > 0 and gripper_state == 'closed':
                    gripper.move(gripper.get_open_position(), 128, 128)
                    gripper_state = 'open'

            # ---- Run OSC at 500 Hz for one policy period ----
            for _ in range(steps_per_action):
                t0 = rtde_c.initPeriod()
                cj = np.array(rtde_r.getActualQ(), dtype=float)
                jv = np.array(rtde_r.getActualQd(), dtype=float)
                ep, eq = get_ee_pose(cj)
                J = compute_jacobian_calibrated(cj)
                ev = J @ jv
                tau = osc.compute(ep, eq, ev, J)
                rtde_c.directTorque(tau.tolist(), friction_comp=False)
                rtde_c.waitPeriod(t0)

            if i % 5 == 0:
                mode_str = "ABS" if absolute else "DEL"
                print(f"  [{i:3d}/{ep_len}] [{mode_str}] ee=[{ee_pos[0]*1000:.1f}, {ee_pos[1]*1000:.1f}, "
                      f"{ee_pos[2]*1000:.1f}] mm  delta=[{delta[0]*1000:.2f}, "
                      f"{delta[1]*1000:.2f}, {delta[2]*1000:.2f}] mm")

        print("\nReplay complete!")

    finally:
        # Clean shutdown
        try:
            rtde_c.directTorque([0.0]*6, friction_comp=False)
            time.sleep(0.1)
            cj = rtde_r.getActualQ()
            rtde_c.servoJ(cj, 0.5, 0.5, 0.1, 0.1, 300)
            rtde_c.servoStop()
        except Exception as ex:
            print(f"Cleanup error: {ex}")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        try:
            gripper.disconnect()
        except:
            pass
        print("Disconnected.")

    # ---- Analysis ----
    if len(real_joint_log) == 0:
        print("\nNo data recorded (replay interrupted before any steps).")
        return

    real_joints_arr = np.array(real_joint_log)
    real_ee_arr = np.array(real_ee_log)
    min_len = min(len(sim_joints), len(real_joints_arr))
    sim_j = sim_joints[:min_len]
    real_j = real_joints_arr[:min_len]
    real_e = real_ee_arr[:min_len]
    times = np.arange(min_len) * dt

    print(f"\nRecorded {min_len}/{ep_len} steps.")

    jerr = np.rad2deg(real_j - sim_j)
    print(f"Joint error (deg): RMSE={np.sqrt(np.mean(jerr**2)):.2f}, "
          f"max={np.max(np.abs(jerr)):.2f}")

    sim_e = None
    if sim_ee is not None:
        sim_e = sim_ee[:min_len]
        perr = (real_e[:, :3] - sim_e[:, :3]) * 1000
        print(f"EE pos error (mm):  RMSE={np.sqrt(np.mean(perr**2)):.2f}, "
              f"max={np.max(np.abs(perr)):.2f}")
        plot_ee_comparison(sim_e, real_e, times, output_path, episode)

    plot_joint_comparison(sim_j, real_j, times, output_path, episode)

    # Save
    save_data = {
        'sim_joints': torch.from_numpy(sim_j).float(),
        'real_joints': torch.from_numpy(real_j).float(),
        'real_ee': torch.from_numpy(real_e).float(),
        'sim_ee': torch.from_numpy(sim_e).float() if sim_e is not None else None,
        'times': torch.from_numpy(times).float(),
        'actions': torch.from_numpy(actions[:min_len]).float(),
        'action_scale': torch.from_numpy(action_scale).float(),
        'episode': episode, 'frequency': frequency,
    }
    save_path = output_path / f'comparison_ep{episode}.pt'
    torch.save(save_data, save_path)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
