# factory_v2 sim2real Cookbook

Practical recipes for (A) deploying a trained RGB-DAgger student on the real UR5e
(JIT export → eval) and (B) checking sim↔real camera/scene alignment via overlays.

## Repos & environments

| Repo | Path | Conda env | Used for |
|---|---|---|---|
| IsaacLab_factory | `~/research/IsaacLab_factory` | `env_isaaclab` | sim: JIT export, sim camera render |
| diffusion_policy | `~/research/diffusion_policy` | `robodiff` | real: eval, camera capture, live overlay |
| UWLab-private | `~/research/UWLab-private/students/` | — | where student `.pt` / `_jit.pt` live |

- Robot IP: **192.168.1.10**. RealSense serials: **side `832112070487`** (D435), **wrist `746112060198`** (D415).
- Isaac runs need: `OMNI_KIT_ACCEPT_EULA=YES`, `ISAACLAB_USD_CACHE_DIR=$HOME/research/IsaacLab_factory/data_storage/usd_cache`, `TMPDIR=$ISAACLAB_USD_CACHE_DIR`, and `--device cuda:1` (GPU 1; do NOT set `CUDA_VISIBLE_DEVICES` — it hangs Kit).
- **Flaky-crash note:** Isaac config construction intermittently dies with `SystemError: unknown opcode` / `DictConfig not iterable` / segfault. It's nondeterministic — just **re-run** (the loops below retry up to 3×).

Task id (RGB student): `Isaac-FactoryV2-Ur5eRobotiq2f85-RelOSC-Sim2Real-SymCritic-RealisticDR-RGBDAgger-v0`

---

## A. Deploy a real policy (JIT export → eval)

### A1. Pick the run & know its proprio contract

Current runs are **arm-only, 19-dim**: proprio = `prev_actions(7) | joint_pos(6 arm) | ee_pose(6)`
(the env's default since commit `46bbf5e`; the gripper joints were dropped because they aren't
reliably observable on the real robot). The deploy script `eval_real_robot_student.py`
expects this 19-dim layout.

> Legacy runs (e.g. `h40vkln7`) were **25-dim** (joint_pos = all 12, incl. Robotiq mimics). If you
> ever deploy one, you'd need the old all-joints env cfg + a 25-dim deploy script. Confirm a run's
> dim from its wandb `env_cfg.observations.proprio.joint_pos` if unsure.

### A2. Download the checkpoint

```bash
conda activate env_uwlab   # has wandb
python - <<'PY'
import wandb, os
run = wandb.Api().run("patyin/factory_v2_sim2real_rgb_dagger/<RUN_ID>")
dst = os.path.expanduser("~/research/UWLab-private/students"); os.makedirs(dst, exist_ok=True)
f = "model_40000.pt"   # latest available (run may still be training)
run.file(f).download(root="/tmp/_dl", replace=True)
os.replace(f"/tmp/_dl/{f}", f"{dst}/<RUN_ID>_{f}")
print("saved", f"{dst}/<RUN_ID>_{f}")
PY
```

### A3. Make the stub teacher (one-time per box; lives in /tmp)

The exporter constructs the runner with a JIT teacher but never calls it at inference.

```bash
conda activate env_isaaclab
cd ~/research/IsaacLab_factory
python scripts/debug/make_dummy_teacher_jit.py --num_obs 1177 --num_actions 7 --out /tmp/dummy_teacher_jit.pt
```

### A4. Export the JIT

```bash
cd ~/research/IsaacLab_factory
OMNI_KIT_ACCEPT_EULA=YES \
ISAACLAB_USD_CACHE_DIR=$HOME/research/IsaacLab_factory/data_storage/usd_cache \
TMPDIR=$HOME/research/IsaacLab_factory/data_storage/usd_cache \
python -u scripts/reinforcement_learning/rsl_rl/export_student_jit_factory.py \
  --task Isaac-FactoryV2-Ur5eRobotiq2f85-RelOSC-Sim2Real-SymCritic-RealisticDR-RGBDAgger-v0 \
  --num_envs 2 --headless --enable_cameras --device cuda:1 \
  --checkpoint ~/research/UWLab-private/students/<RUN_ID>_model_40000.pt \
  --out ~/research/UWLab-private/students/<RUN_ID>_model_40000_student_jit.pt \
  --wandb_run patyin/factory_v2_sim2real_rgb_dagger/runs/<RUN_ID> \
  agent.vision_policy.teacher_jit_path=/tmp/dummy_teacher_jit.pt \
  agent.vision_policy.encoder_pretrained_path=null
```
Success = `max|wrapper - act_inference| < 5e-3`. Writes `<...>_student_jit.pt` +
`.meta.json` (records proprio_dim, squash, scale, etc.).

Sanity-check the JIT standalone:
```bash
conda activate env_uwlab
python -c "import torch; m=torch.jit.load('<jit>').eval(); \
print(m(torch.zeros(1,19), torch.rand(1,1,3,224,224), torch.rand(1,1,3,224,224)).shape)"  # -> (1,7)
```

### A5. Eval on the real robot

```bash
conda activate robodiff
cd ~/research/diffusion_policy
python eval_real_robot_student.py \
  -i ~/research/UWLab-private/students/<RUN_ID>_model_40000_student_jit.pt \
  -o ./demo_student_<RUN_ID> -ri 192.168.1.10 -j -f 10
```
- `-j` moves to the eval init pose first; `-f 10` = 10 Hz control. Keys: `r`=reset, `q`=quit.
- Obs contract baked into the script: proprio 19 (arm-only), side+wrist RGB `(1,1,3,224,224)` fed
  as `[0,1]` (ResNet18 does ImageNet norm internally), action tanh-squashed × `(0.1,0.1,0.1,0.2,0.2,0.2)`,
  gripper `<0` close / `>=0` open.
- Make sure **both** RealSense are on the bus first (see Appendix).

---

## B. Sim↔real camera alignment overlay

Goal: render the sim cameras at a known joint pose with the camera pose/FOV pinned
(appearance DR left on), and overlay against a real photo, to check/tune alignment.

### B1. Capture the real reference(s)

```bash
conda activate robodiff
cd ~/research/diffusion_policy/scripts/sim2real
python 2_capture_align_refs.py            # default: move arm to eval init pose, gripper open, capture
# variants:
python 2_capture_align_refs.py --no_move          # capture at the CURRENT arm pose (no motion)
python 2_capture_align_refs.py --gripper close     # or none (don't actuate)
```
Saves `align_refs/side_real.png` + `align_refs/wrist_real.png`, auto-detects the connected
cams (2 = side+wrist is fine), and prints the actual joint angles + a ready-to-run B2 command.

### B2. Render sim + static overlay

```bash
conda activate env_isaaclab
cd ~/research/IsaacLab_factory
OMNI_KIT_ACCEPT_EULA=YES \
ISAACLAB_USD_CACHE_DIR=$HOME/research/IsaacLab_factory/data_storage/usd_cache \
TMPDIR=$HOME/research/IsaacLab_factory/data_storage/usd_cache \
python -u scripts/sim2real/align_cameras_factory.py \
  --enable_cameras --headless --device cuda:1 --gripper open \
  --joint_angles 16.85 -79.74 99.80 -114.68 -91.09 20.43 \
  --side_real  ~/research/diffusion_policy/scripts/sim2real/align_refs/side_real.png \
  --wrist_real ~/research/diffusion_policy/scripts/sim2real/align_refs/wrist_real.png \
  --out_dir /tmp/align_factory --blend 0.5
```
Writes per camera `<cam>_sim.png`, `<cam>_overlay.png`, `<cam>_panel.png` (sim | real | blend).

Useful flags:
- `--assembled` — seat the rod in the board hole (for a real scene with the rod inserted).
- `--board_x/--board_y/--board_yaw` — pin the board to a fixed pose (deterministic; default = the
  `NIST_BOARD_SIM2REAL_RANGE` jitter). e.g. `--board_x 0.5 --board_y 0.25 --board_yaw 180`.
- `--joint_angles` — paste the **actual** degrees the capture printed.

What it pins vs the training env: camera **pose** + **FOV** jitter, `reset_positioning` (so the arm
holds the commanded joints). What it keeps: texture/color/material/HDRI **appearance DR** (sim looks
varied), and `reset_arm_home` (writes the joints — do NOT disable it or the arm goes to all-zeros).
Board renders at yaw 180° (the sim2real deploy orientation).

### B3. Live overlay — move the real board to match sim

Generate a sim reference at the target board pose (B2 with `--board_*`), then stream the real side
cam blended over it so you can physically nudge the board:

```bash
conda activate robodiff
cd ~/research/diffusion_policy
python scripts/sim2real/live_board_overlay.py --sim_image /tmp/align_factory/side_camera_sim.png
```
Keys: `e` = edge mode (sim edges drawn green over live real — best for board alignment),
`[`/`]` = blend, `s` = save, `q` = quit. (If no window: prefix `DISPLAY=:1 `.)

---

## Appendix: RealSense USB drops (the wrist cam)

The wrist cam (`746112060198`) periodically falls off USB. Root cause = USB autosuspend.

**Stop it (one-time, persistent):** install the udev rule that disables autosuspend for the
RealSense + their hub:
```bash
sudo cp ~/research/diffusion_policy/scripts/sim2real/99-realsense-no-autosuspend.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --attr-match=idVendor=8086 && sudo udevadm trigger --attr-match=idVendor=05e3
# verify: cat /sys/bus/usb/devices/2-1.3/power/control  -> "on"
```

**Recover a dropped cam (software "replug"):**
```bash
sudo bash ~/research/diffusion_policy/scripts/sim2real/reset_realsense_usb_hub.sh   # re-enumerate the hubs
# or auto-watchdog: sudo bash ~/research/diffusion_policy/scripts/sim2real/reset_realsense_watchdog.sh
```
If a cam is fully dead even after that, physically replug it once. Check what's on the bus:
```bash
conda activate robodiff
python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]"
```
