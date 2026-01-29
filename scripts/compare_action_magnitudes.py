import argparse
import os
from typing import Optional

import numpy as np


def summarize(name: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        print(f"{name}: empty")
        return
    q = np.quantile(x, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(
        f"{name}: n={x.size}, mean={x.mean():.6g}, std={x.std():.6g}, min={q[0]:.6g}, "
        f"p1={q[1]:.6g}, p5={q[2]:.6g}, median={q[3]:.6g}, p95={q[4]:.6g}, p99={q[5]:.6g}, max={q[6]:.6g}"
    )


def main():
    ap = argparse.ArgumentParser(description="Compare magnitudes between ee_action zarr and actions.npy")
    ap.add_argument("--ee_path", required=True, help="Path to zarr array directory, e.g. .../replay_buffer.zarr/data/ee_action")
    ap.add_argument("--npy", required=True, help="Path to actions.npy")
    ap.add_argument("--batch", type=int, default=100000, help="Batch size for reading zarr")
    ap.add_argument("--actions_scale", type=float, default=1.0, help="Multiply actions.npy by this factor before comparing (e.g., 0.1 to divide by 10)")
    args = ap.parse_args()

    # Load actions.npy and make it 2D (N, D), using last dim as features
    actions = np.load(args.npy)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    else:
        actions = actions.reshape(-1, actions.shape[-1])
    if args.actions_scale != 1.0:
        actions = actions * args.actions_scale
    print(f"Loaded npy: shape={actions.shape}, dtype={actions.dtype}, scale={args.actions_scale}")

    # Try to load zarr
    try:
        import zarr  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: zarr. Install with `pip install zarr numcodecs` and rerun."
        ) from e

    ee = zarr.open(args.ee_path, mode="r")
    if getattr(ee, "shape", None) is None or len(ee.shape) != 2:
        raise SystemExit(f"Unexpected zarr array shape at {args.ee_path}: {getattr(ee,'shape',None)}")
    n = int(ee.shape[0])
    d = int(ee.shape[1])
    print(f"Loaded zarr: shape=({n},{d}), chunks={getattr(ee,'chunks',None)}, dtype={getattr(ee,'dtype',None)}")

    # Compute norms
    ee_norms = np.empty((n,), dtype=np.float64)
    for start in range(0, n, args.batch):
        stop = min(start + args.batch, n)
        chunk = np.asarray(ee[start:stop], dtype=float)
        ee_norms[start:stop] = np.linalg.norm(chunk, axis=1)

    # Prepare actions norms (use first d dims to match)
    if actions.shape[0] != n:
        m = min(actions.shape[0], n)
        print(f"Length mismatch: zarr N={n} vs npy N={actions.shape[0]} -> comparing first {m}")
        ee_norms = ee_norms[:m]
        actions = actions[:m]

    d_use = min(actions.shape[1], d)
    if actions.shape[1] != d_use:
        print(f"Using first {d_use} dims of actions to match ee_action dims {d}")
    act_norms = np.linalg.norm(actions[:, :d_use].astype(float), axis=1)

    # Summaries
    summarize("ee_action |norm|", ee_norms)
    summarize("actions.npy |norm|", act_norms)

    # Simple comparisons
    eps = 1e-12
    ratio = ee_norms / np.maximum(act_norms, eps)
    summarize("ratio (ee / npy)", ratio)
    # Correlation
    corr = np.corrcoef(ee_norms, act_norms)[0, 1]
    print(f"pearson corr = {corr:.6g}")


if __name__ == "__main__":
    main()
