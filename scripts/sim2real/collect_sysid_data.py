"""Collect chirp excitation data on real UR5e for PACE system identification.

Runs a frequency-sweep (chirp) trajectory through the calibrated OSC controller
(same controller used in simulation), recording joint positions and OSC target
poses at 500 Hz.  The output .pt file is consumed directly by
UWLab/scripts/sysid/sysid_ur5e_osc.py for CMA-ES optimization.

Usage (from diffusion_policy repo root):
    python scripts/sim2real/collect_sysid_data.py --robot_ip 192.168.1.10 \
        --output data/sysid_data_real.pt

    python scripts/sim2real/collect_sysid_data.py --robot_ip 192.168.1.10 \
        --output data/sysid_data_real.pt \
        --duration 12 --f0 0.1 --f1 3.0 --pos_amp 0.10 --rot_amp 0.25
"""

import numpy as np
import time
import torch
import click
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, KeyCode,
)
from diffusion_policy.real_world.ur5e_kinematics import (
    PAYLOAD_MASS, PAYLOAD_COG,
    axis_angle_to_quat,
    compute_jacobian_calibrated,
    get_ee_pose,
    OperationalSpaceController,
)

CONTROL_FREQUENCY = 500


def generate_chirp_trajectory(duration, dt, f0, f1, pos_amp, rot_amp):
    """Linear chirp (frequency sweep) in Cartesian space.

    Each axis gets a sinusoid whose frequency sweeps from f0 to f1.
    Phase offsets decouple axes; amplitude envelope ramps up/down smoothly.

    Returns:
        offsets: (T, 6) [dx,dy,dz,drx,dry,drz] from center pose
        t: (T,) time array
    """
    T = int(duration / dt)
    t = np.linspace(0, duration, T)

    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)

    envelope = np.ones(T)
    ramp_up_n = int(2.0 / dt)
    ramp_down_n = int(3.0 / dt)
    envelope[:ramp_up_n] = np.linspace(0, 1, ramp_up_n)
    envelope[-ramp_down_n:] = np.linspace(1, 0, ramp_down_n)

    phase_offsets = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    amps = [pos_amp, pos_amp, pos_amp * 1.5,
            rot_amp * 2.0, rot_amp, rot_amp * 2.0]

    offsets = np.zeros((T, 6))
    for i in range(6):
        offsets[:, i] = amps[i] * envelope * np.sin(phase + phase_offsets[i])

    return offsets, t


def generate_wrist3_chirp(duration, dt, f0, f1, rot_amp):
    """1-DOF chirp on tool-local roll (rotation about EE z-axis).

    The wrist_3 joint axis IS the tool-flange z-axis, so a pure roll about EE-z is
    achieved by wrist_3 alone while the other 5 joints hold their current position.

    Returns:
        roll: (T,) tool-local roll angle [rad] about EE z over time
        t: (T,) time array
    """
    T = int(duration / dt)
    t = np.linspace(0, duration, T)

    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)

    envelope = np.ones(T)
    ramp_up_n = int(2.0 / dt)
    ramp_down_n = int(3.0 / dt)
    envelope[:ramp_up_n] = np.linspace(0, 1, ramp_up_n)
    envelope[-ramp_down_n:] = np.linspace(1, 0, ramp_down_n)

    roll = rot_amp * envelope * np.sin(phase)
    return roll, t


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


@click.command()
@click.option('--robot_ip', '-ri', default='192.168.1.10', help="UR5e IP address")
@click.option('--output', '-o', required=True, help="Output .pt file for sysid")
@click.option('--joints_init_deg', '-j', default='0,-90,90,-90,-90,0',
              help="Initial joint angles in degrees, comma-separated")
@click.option('--duration', '-d', default=8.0, type=float, help="Chirp duration (seconds)")
@click.option('--f0', default=0.1, type=float, help="Start frequency (Hz)")
@click.option('--f1', default=3.0, type=float, help="End frequency (Hz)")
@click.option('--pos_amp', default=0.10, type=float, help="Position amplitude (meters)")
@click.option('--rot_amp', default=0.25, type=float, help="Rotation amplitude (radians)")
@click.option('--wrist3_only', is_flag=True, default=False,
              help="Isolate wrist_3: 1-DOF chirp as tool-local roll about EE z; other joints hold position. "
                   "Pair with sysid_ur5e_osc.py --opt_joints wrist_3_joint.")
@click.option('--kp_pos', default=1000.0, type=float, help="Position stiffness")
@click.option('--kp_rot', default=50.0, type=float, help="Rotation stiffness")
@click.option('--damping_ratio', default=1.0, type=float, help="Damping ratio")
@click.option('--payload_mass', default=None, type=float, help="Override payload mass (kg)")
@click.option('--payload_cog', default=None, type=str, help="Override payload CoG 'x,y,z' (m)")
def main(robot_ip, output, joints_init_deg, duration, f0, f1, pos_amp, rot_amp,
         wrist3_only, kp_pos, kp_rot, damping_ratio, payload_mass, payload_cog):
    """Collect chirp excitation data for PACE system identification."""
    j_init = np.deg2rad([float(x) for x in joints_init_deg.split(',')])
    assert len(j_init) == 6

    pl_mass = payload_mass if payload_mass is not None else PAYLOAD_MASS
    if payload_cog is not None:
        pl_cog = [float(x) for x in payload_cog.split(',')]
    else:
        pl_cog = list(PAYLOAD_COG)

    dt = 1.0 / CONTROL_FREQUENCY
    motion_stiffness = (kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot)
    motion_damping_ratio = (damping_ratio,) * 6
    torque_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    osc = OperationalSpaceController(
        motion_stiffness=motion_stiffness,
        motion_damping_ratio=motion_damping_ratio,
        torque_max=torque_max,
    )

    if wrist3_only:
        wrist3_roll, chirp_t = generate_wrist3_chirp(
            duration=duration, dt=dt, f0=f0, f1=f1, rot_amp=rot_amp,
        )
        chirp_offsets = None
    else:
        chirp_offsets, chirp_t = generate_chirp_trajectory(
            duration=duration, dt=dt, f0=f0, f1=f1,
            pos_amp=pos_amp, rot_amp=rot_amp,
        )
        wrist3_roll = None
    n_steps = len(chirp_t)

    kd_diag = 2 * np.sqrt(np.array(motion_stiffness)) * np.array(motion_damping_ratio)
    print("=" * 60)
    print("Collect Sysid Data — Chirp Excitation" + ("  [wrist_3 only]" if wrist3_only else ""))
    print("=" * 60)
    print(f"Duration:  {duration:.1f}s  ({n_steps} steps at {CONTROL_FREQUENCY}Hz)")
    print(f"Frequency: {f0:.2f} -> {f1:.1f} Hz")
    if wrist3_only:
        print(f"Amplitude: tool-roll(wrist_3)={np.degrees(rot_amp):.1f}deg  (other joints hold position)")
    else:
        print(f"Amplitude: pos={pos_amp*1000:.0f}mm  rot={np.degrees(rot_amp):.1f}deg")
    print(f"Kp:        {list(motion_stiffness)}")
    print(f"Kd:        [{', '.join(f'{x:.1f}' for x in kd_diag)}]")
    print(f"Output:    {output}")
    print("=" * 60)
    print("Press 'q' to abort.\n")

    rtde_c = RTDEControlInterface(
        robot_ip, CONTROL_FREQUENCY,
        RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT,
    )
    rtde_r = RTDEReceiveInterface(robot_ip, CONTROL_FREQUENCY)
    rtde_c.setPayload(pl_mass, pl_cog)

    try:
        with KeystrokeCounter() as key_counter:
            print("Moving to initial joint position...")
            ok = rtde_c.moveJ(j_init.tolist(), 1.05, 1.4)
            if not ok:
                raise RuntimeError("moveJ to initial joints failed")
            print("Initial position reached.")

            current_joints = np.array(rtde_r.getActualQ(), dtype=float)
            center_pos, center_quat = get_ee_pose(current_joints)
            print(f"Center EE: [{center_pos[0]*1000:.1f}, {center_pos[1]*1000:.1f}, "
                  f"{center_pos[2]*1000:.1f}] mm\n")

            joint_positions = []
            joint_torques = []
            tcp_forces = []
            waypoints = []
            initial_joint_pos = current_joints.copy()

            osc.set_target(center_pos, center_quat)

            stop = False
            for step in range(n_steps):
                t_start = rtde_c.initPeriod()

                for ks in key_counter.get_press_events():
                    if ks == KeyCode(char='q'):
                        print("\nAborted by user.")
                        stop = True
                        break
                if stop:
                    break

                if wrist3_only:
                    # Tool-local roll about EE z (right-multiply) -> driven by wrist_3 alone.
                    target_pos = center_pos.copy()
                    target_quat = quat_multiply(center_quat, axis_angle_to_quat([0.0, 0.0, wrist3_roll[step]]))
                else:
                    offset = chirp_offsets[step]
                    target_pos = center_pos + offset[:3]
                    target_quat = quat_multiply(axis_angle_to_quat(offset[3:6]), center_quat)
                osc.set_target(target_pos, target_quat)

                curr_joints = np.array(rtde_r.getActualQ(), dtype=float)
                ee_pos, ee_quat = get_ee_pose(curr_joints)
                jacobian = compute_jacobian_calibrated(curr_joints)
                ee_vel = jacobian @ np.array(rtde_r.getActualQd(), dtype=float)

                torque_cmd = osc.compute(ee_pos, ee_quat, ee_vel, jacobian)
                rtde_c.directTorque(torque_cmd.tolist(), friction_comp=False)

                joint_positions.append(curr_joints.copy())
                joint_torques.append(torque_cmd.copy())
                tcp_forces.append(np.array(rtde_r.getActualTCPForce(), dtype=float))
                waypoints.append({
                    "step_idx": step,
                    "target_pos": target_pos.copy(),
                    "target_quat": target_quat.copy(),
                })

                if step % (CONTROL_FREQUENCY * 2) == 0:
                    elapsed = step * dt
                    inst_freq = f0 + (f1 - f0) * elapsed / duration
                    pos_err = np.linalg.norm(ee_pos - target_pos)
                    print(f"  [{elapsed:.1f}s] freq={inst_freq:.2f}Hz  "
                          f"pos_err={pos_err*1000:.1f}mm  "
                          f"|tau|={np.linalg.norm(torque_cmd):.1f}Nm")

                rtde_c.waitPeriod(t_start)

            print(f"\nChirp completed ({len(joint_positions)} steps).")

            if joint_positions:
                sysid_data = {
                    "joint_positions": torch.tensor(np.array(joint_positions), dtype=torch.float32),
                    "joint_torques": torch.tensor(np.array(joint_torques), dtype=torch.float32),
                    "tcp_forces": torch.tensor(np.array(tcp_forces), dtype=torch.float32),
                    "initial_joint_pos": torch.tensor(initial_joint_pos, dtype=torch.float32),
                    "dt": dt,
                    "control_freq": CONTROL_FREQUENCY,
                    "osc_params": {
                        "motion_stiffness": list(motion_stiffness),
                        "motion_damping_ratio": list(motion_damping_ratio),
                        "torque_max": torque_max.tolist(),
                    },
                    "chirp_params": {
                        "duration": duration, "f0": f0, "f1": f1,
                        "pos_amp": pos_amp, "rot_amp": rot_amp,
                        "wrist3_only": wrist3_only,
                    },
                    "num_waypoints": len(waypoints),
                    "waypoint_step_indices": torch.tensor(
                        [w["step_idx"] for w in waypoints], dtype=torch.long),
                    "waypoint_target_pos": torch.tensor(
                        np.array([w["target_pos"] for w in waypoints]), dtype=torch.float32),
                    "waypoint_target_quat": torch.tensor(
                        np.array([w["target_quat"] for w in waypoints]), dtype=torch.float32),
                }
                torch.save(sysid_data, output)
                print(f"Saved {len(joint_positions)} steps to: {output}")

    finally:
        try:
            rtde_c.directTorque([0.0] * 6, friction_comp=False)
            time.sleep(0.1)
            current_joints = rtde_r.getActualQ()
            rtde_c.servoJ(current_joints, 0.5, 0.5, 0.1, 0.1, 300)
            rtde_c.servoStop()
        except Exception as e:
            print(f"Cleanup error: {e}")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Disconnected from robot.")


if __name__ == "__main__":
    main()
