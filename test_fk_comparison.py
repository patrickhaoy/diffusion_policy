"""Verify FK alignment: compare calibrated FK against simulation EE poses.

Loads (joint_pos, ee_pose) pairs collected from the Isaac Lab simulator
(via OctiLab/scripts/sim2real/collect_fk_pairs.py) and compares with our
calibrated forward kinematics.  Both should match within 0.01 mm per
dimension, verifying sim2real kinematic alignment.

Usage:
    python test_fk_comparison.py --pairs fk_pairs.npz
    python test_fk_comparison.py --pairs fk_pairs.npz --threshold 0.05
"""
import click
import numpy as np

from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose


def quat_angle_distance(q1, q2):
    """Geodesic distance between two unit quaternions (w,x,y,z), in radians."""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


@click.command()
@click.option('--pairs', '-p', required=True,
              help='Path to npz file from collect_fk_pairs.py')
@click.option('--threshold', '-t', default=0.01, type=float,
              help='Pass/fail threshold in mm per dimension (default: 0.01)')
def main(pairs, threshold):
    data = np.load(pairs)
    joint_pos = data['joint_pos']       # (N, 6)
    sim_ee_pos = data['ee_pos']         # (N, 3) meters
    sim_ee_quat = data['ee_quat']       # (N, 4) w,x,y,z

    n = len(joint_pos)
    print(f"Loaded {n} FK pairs from {pairs}")
    print(f"Threshold: {threshold:.3f} mm per dimension")
    print("=" * 80)

    header = f"{'#':>4} | {'dX mm':>8} | {'dY mm':>8} | {'dZ mm':>8} | {'3D mm':>7} | {'rot deg':>8}"
    print(header)
    print("-" * 80)

    pos_errors_mm = np.zeros((n, 3))
    rot_errors_deg = np.zeros(n)

    for i in range(n):
        fk_pos, fk_quat = get_ee_pose(joint_pos[i])

        pos_err = (fk_pos - sim_ee_pos[i]) * 1000.0
        pos_err_3d = np.linalg.norm(pos_err)
        rot_err = np.degrees(quat_angle_distance(fk_quat, sim_ee_quat[i]))

        pos_errors_mm[i] = pos_err
        rot_errors_deg[i] = rot_err

        flag = " !" if np.any(np.abs(pos_err) > threshold) else ""
        print(f"{i:4d} | {pos_err[0]:8.4f} | {pos_err[1]:8.4f} | {pos_err[2]:8.4f} "
              f"| {pos_err_3d:7.4f} | {rot_err:8.4f}{flag}")

    # Summary -------------------------------------------------------------------
    abs_err = np.abs(pos_errors_mm)
    norms = np.linalg.norm(pos_errors_mm, axis=1)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Samples:             {n}")
    print(f"  Mean |err| per dim:  X={np.mean(abs_err[:,0]):.4f}  "
          f"Y={np.mean(abs_err[:,1]):.4f}  Z={np.mean(abs_err[:,2]):.4f} mm")
    print(f"  Max  |err| per dim:  X={np.max(abs_err[:,0]):.4f}  "
          f"Y={np.max(abs_err[:,1]):.4f}  Z={np.max(abs_err[:,2]):.4f} mm")
    print(f"  Mean 3D error:       {np.mean(norms):.4f} mm")
    print(f"  Max  3D error:       {np.max(norms):.4f} mm")
    print(f"  Mean rotation error: {np.mean(rot_errors_deg):.4f} deg")
    print(f"  Max  rotation error: {np.max(rot_errors_deg):.4f} deg")

    max_per_dim = np.max(abs_err, axis=0)
    passed = np.all(max_per_dim <= threshold)

    print(f"\n  Threshold:  {threshold:.3f} mm per dimension")
    if passed:
        print(f"  Result:     PASS  (max per-dim error: {np.max(max_per_dim):.4f} mm)")
    else:
        print(f"  Result:     FAIL  (max per-dim error: {np.max(max_per_dim):.4f} mm)")
        for d, name in enumerate(["X", "Y", "Z"]):
            if max_per_dim[d] > threshold:
                print(f"              {name}: {max_per_dim[d]:.4f} mm > {threshold:.3f} mm")


if __name__ == '__main__':
    main()
