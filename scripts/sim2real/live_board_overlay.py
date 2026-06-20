"""Live sim↔real overlay for board alignment.

Streams the real SIDE RealSense and blends it over a FIXED sim render
(align_cameras_factory.py's side_camera_sim.png), so you can physically move the
real board and watch it converge onto the sim board in real time.

Sim render is the reference (board pinned at e.g. x=0.5 y=0.25 yaw=180). Generate it
first with:
    align_cameras_factory.py ... --board_x 0.5 --board_y 0.25 --board_yaw 180 \
        --out_dir /tmp/align_factory
then point this at /tmp/align_factory/side_camera_sim.png.

Run (robodiff env, needs the display):
    python scripts/sim2real/live_board_overlay.py
    # or: DISPLAY=:1 python scripts/sim2real/live_board_overlay.py --sim_image /tmp/align_factory/side_camera_sim.png

Keys (window focused):
    [ / ]   blend less/more real (alpha)
    e       toggle EDGE mode (sim board edges drawn green over live real) — best for alignment
    s       save current overlay to /tmp/align_factory/live_overlay.png
    q / ESC quit
"""
import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs

SIDE_SERIAL = "832112070487"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_image", default="/tmp/align_factory/side_camera_sim.png",
                    help="Fixed sim render to overlay (the reference).")
    ap.add_argument("--serial", default=SIDE_SERIAL, help="RealSense serial (default = side cam).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--blend", type=float, default=0.5, help="Initial real alpha (0=all sim, 1=all real).")
    args = ap.parse_args()

    sim = cv2.imread(args.sim_image)  # BGR
    if sim is None:
        raise FileNotFoundError(f"sim image not found: {args.sim_image}")
    sim = cv2.resize(sim, (args.width, args.height))
    sim_edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 60, 160)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    pipe.start(cfg)
    print(f"[live] streaming side cam {args.serial}; sim ref = {args.sim_image}")
    print("[live] keys: [ ] blend, e edge mode, s save, q quit")

    win = "sim<->real board overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    alpha = float(args.blend)
    edge_mode = False
    try:
        while True:
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            real = np.asanyarray(cf.get_data())  # BGR HxWx3

            if edge_mode:
                out = real.copy()
                out[sim_edges > 0] = (0, 255, 0)  # sim board/edges in green over live real
                label = "EDGE (green = sim edges)"
            else:
                out = cv2.addWeighted(sim, 1.0 - alpha, real, alpha, 0.0)
                label = f"blend alpha={alpha:.2f} (0=sim,1=real)"

            cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("["):
                alpha = max(0.0, alpha - 0.05)
            elif k == ord("]"):
                alpha = min(1.0, alpha + 0.05)
            elif k == ord("e"):
                edge_mode = not edge_mode
            elif k == ord("s"):
                p = os.path.join(os.path.dirname(args.sim_image), "live_overlay.png")
                cv2.imwrite(p, out)
                print(f"[live] saved {p}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
