"""Real-robot eval for the factory_v2 RGB-DAgger student (TorchScript export).

For the IsaacLab factory_v2 RelOSC+tanh recipe (e.g. wandb run mi3i6p1m).
The recipe-specific knobs are isolated to the CONSTANTS below + the action scale.

Obs contract — see students/<run>_student_jit.pt.meta.json:
  proprio (19) = single current frame, concatenated in env-cfg term order:
      [0:7]    prev_actions  : 7  (tanh-squashed last JIT action: 6 OSC delta + gripper)
      [7:13]   joint_pos     : 6  (UR5e arm joints only — shoulder_pan/lift, elbow,
                                   wrist_1/2/3, from RTDE getActualQ. No gripper/Robotiq
                                   joints: the student is trained arm-only.)
      [13:19]  ee_pose       : 6  (wrist_3_link in robot-root frame, axis-angle)
  side_seq, wrist_seq : (1,1,3,224,224) — single current frame/camera, float [0,1] CHW.
      NO mean subtraction here: the ResNet18 encoder applies ImageNet mean/std internally.
  action (7) : RelCartesianOSC. JIT output is TANH-SQUASHED in (-1,1) already.
      [:6] multiplied by CARTESIAN_SCALE = (0.1,0.1,0.1,0.2,0.2,0.2) for the OSC delta.
      [6]  gripper: <0 close, >=0 open (same convention as OmniReset / RealEnv).

Usage:
    python eval_real_robot_student.py \\
        -i ~/research/UWLab-private/students/mi3i6p1m_model_40000_student_jit.pt \\
        -o ./demo_student_mi3i6p1m --robot_ip 192.168.1.10 -j -f 10
Keys (terminal focused): r=reset, q=quit (Ctrl-C also stops).
"""

import time
import sys
import select
import termios
import tty
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import pathlib

from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose, quat_to_axis_angle, apply_delta_pose

# Per-axis Cartesian scale for the factory_v2 tanh recipe.
# env.actions.arm.scale_xyz_axisangle = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2). JIT output is
# already tanh-squashed in (-1, 1); multiply by this scale to get the OSC delta.
CARTESIAN_SCALE = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])

# Single current frame. proprio 19 = prev_actions(7) + joint_pos(6) + ee_pose(6).
# joint_pos = 6 UR5e arm joints only (the student is trained arm-only; no Robotiq joints).
# RGB is a single current frame per camera, shape (1,1,3,224,224).
PROPRIO_DIM = 19    # 7 + 6 + 6
IMG = 224


class KeyPoller:
    """Non-blocking single-char terminal key reader (no Enter, no cv2 focus).

    Keep the TERMINAL focused and press c/s/r/q. Ctrl-C still works (ISIG kept).
    No-op if stdin is not a tty.
    """

    def __enter__(self):
        self.fd = None
        try:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except Exception:
            self.fd = None
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        if self.fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def compute_calibrated_ee_pose(joint_positions):
    """EE pose via calibrated FK (wrist_3_link in REP-103 base frame), axis-angle.
    Matches the sim's target_asset_pose_in_root_asset_frame. (n,6) -> (n,6).

    IMPORTANT: canonicalize the quaternion to w>=0 before axis-angle, to match
    IsaacLab's axis_angle_from_quat (which negates q when w<0). Without this, when
    the EE orientation crosses the |angle|=pi singularity (w changes sign — common
    for a downward gripper), quat_to_axis_angle returns the OTHER branch
    (angle in (pi,2pi), opposite axis sign) and the student gets an orientation obs
    that diverges from its sim training convention -> erratic motion."""
    n = joint_positions.shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    for t in range(n):
        pos, quat = get_ee_pose(joint_positions[t])
        quat = np.asarray(quat, dtype=np.float64)
        if quat[0] < 0.0:           # canonicalize w>=0 (matches IsaacLab)
            quat = -quat
        out[t, :3] = pos
        out[t, 3:] = quat_to_axis_angle(quat)
    return out


def preprocess_img(img_hwc):
    """Real obs image (H,W,3) float [0,1] -> (3,224,224) CHW tensor, bilinear-antialias
    resize. Feed [0,1]: the student's ResNet18Encoder applies ImageNet mean/std
    INSIDE its forward, so do NOT normalize here."""
    if img_hwc.dtype == np.uint8:
        img_hwc = img_hwc.astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).unsqueeze(0).float()
    if t.shape[-2:] != (IMG, IMG):
        t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", antialias=True)
    return t.clamp_(0.0, 1.0)[0]  # (3,224,224)


@click.command()
@click.option('--input', '-i', required=True, help='Path to student TorchScript (.pt)')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5e IP, e.g. 192.168.1.10")
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="Move to initial joint configuration on start.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency (Hz).")
@click.option('--max_duration', '-md', default=1000, help='Max episode duration (s).')
@click.option('--const_action', default=None, type=str,
              help='DEBUG: ignore the policy and send a fixed raw 7-vec every step, '
                   'e.g. "0,0,-15,0,0,0,1". Isolates the exec path from the NN.')
@click.option('--save_obs_video/--no_obs_video', default=True,
              help='Save a side|wrist concatenated video of the exact obs frames the policy '
                   'sees (the 224x224 student input), to <output>/obs_sidewrist.mp4. Mirrors '
                   "the sim --eval_video. Default on.")
@click.option('--no_stuck', is_flag=True, default=False,
              help='Disable stuck-detection gripper auto-open macro (debugging).')
def main(input, output, robot_ip, init_joints, frequency, max_duration, const_action, save_obs_video, no_stuck):
    const_act = None
    if const_action is not None:
        const_act = np.array([float(x) for x in const_action.split(',')], dtype=np.float32)
        assert const_act.shape == (7,), "const_action must be 7 comma-separated numbers"
        print(f"[DEBUG] const_action mode: sending fixed {const_act} every step (policy ignored)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = torch.jit.load(input, map_location=device).eval()
    print(f"Loaded student JIT: {input} on {device}")
    meta_path = input + '.meta.json'
    if pathlib.Path(meta_path).exists():
        print("obs contract:", json.load(open(meta_path)).get('proprio_composition', ''))
    env_n_obs_steps = 2
    env_obs_res = (IMG, IMG)

    dt = 1.0 / frequency
    # Stuck detection: if the robot doesn't move for STUCK_WINDOW_S, open gripper to unstick.
    STUCK_WINDOW_S = 2.0
    STUCK_JOINT_THRESHOLD_RAD = 0.002  # ~0.1 deg max movement per joint over window
    STUCK_GRIPPER_OPEN_STEPS = int(frequency)  # 1s open at control freq
    # factory_v2 student only needs side + wrist (no front). Drop the front camera
    # entirely so RealEnv doesn't try to open it. RealEnv labels camera obs by index
    # (idx 0 -> 'front_rgb', idx 1 -> 'side_rgb'), so we ALSO remap keys below.
    configs = [json.load(open(f"diffusion_policy/real_world/realsense_config/{c}.json"))
               for c in ("435_side", "415_wrist")]

    def env_get_obs(env):
        """RealEnv labels camera_obs by camera_idx; we passed [side, wrist] so
        idx 0 lands in 'front_rgb' (actually side) and idx 1 in 'side_rgb'
        (actually wrist). Rename via a 2-step shuffle so downstream code can
        use side_rgb / wrist_rgb."""
        o = env.get_obs()
        if 'front_rgb' in o:
            o['side_rgb_tmp'] = o.pop('front_rgb')
        if 'side_rgb' in o:
            o['wrist_rgb'] = o.pop('side_rgb')
        if 'side_rgb_tmp' in o:
            o['side_rgb'] = o.pop('side_rgb_tmp')
        if 'side_rgb' not in o or 'wrist_rgb' not in o:
            raise RuntimeError(
                f"env_get_obs: missing camera keys after rename. obs keys = {sorted(o.keys())}. "
                "Likely one of the 2 RealSense cameras stopped delivering frames mid-run."
            )
        return o

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output, robot_ip=robot_ip, frequency=frequency,
            n_obs_steps=env_n_obs_steps, obs_image_resolution=env_obs_res, obs_float32=True,
            init_joints=init_joints, enable_multi_cam_vis=False,
            record_raw_video=True, action_mode='cartesian',
            camera_serial_numbers=['832112070487', '746112060198'],
            camera_configs=configs, thread_per_video=3, video_crf=21,
            shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)
            cv2.namedWindow('Student Eval', cv2.WINDOW_NORMAL)
            print("Waiting for realsense"); time.sleep(5.0)

            # warmup inference
            obs = env_get_obs(env)
            last_raw_action = np.zeros(7, dtype=np.float32)

            def build_inputs(obs, last_raw_action):
                # proprio joint_pos: 6 UR5e arm joints (arm-only student; no gripper/Robotiq joints)
                arm_q = obs['arm_joint_pos'][-1].astype(np.float32)            # (6,)
                ee = compute_calibrated_ee_pose(obs['arm_joint_pos'])[-1]      # (6,)
                proprio_vec = np.concatenate([last_raw_action, arm_q, ee]).astype(np.float32)
                assert proprio_vec.shape[0] == PROPRIO_DIM, f"proprio dim {proprio_vec.shape[0]} != {PROPRIO_DIM}"
                proprio = torch.from_numpy(proprio_vec).unsqueeze(0).to(device)             # (1,19)
                side = preprocess_img(obs['side_rgb'][-1])[None, None].to(device)           # (1,1,3,224,224)
                wrist = preprocess_img(obs['wrist_rgb'][-1])[None, None].to(device)
                return proprio, side, wrist

            def compute_action(obs, last_raw_action):
                p, s, w = build_inputs(obs, last_raw_action)
                return policy(p, s, w)[0].detach().cpu().numpy().astype(np.float32)

            with torch.no_grad():
                _ = compute_action(obs, last_raw_action)
            print("Ready! Policy runs immediately. Keep THIS TERMINAL focused: r=reset, q=quit (Ctrl-C also stops).")

            # side|wrist concatenated obs video (the exact frames the policy sees)
            obs_writer = None
            obs_vid_path = f"{output}/obs_sidewrist.mp4"
            if save_obs_video:
                import imageio
                obs_writer = imageio.get_writer(obs_vid_path, fps=int(round(frequency)),
                                                codec="libx264", macro_block_size=None)
                print(f"[obs video] recording side|wrist policy-obs -> {obs_vid_path}")

            def _record_obs(obs):
                if obs_writer is None:
                    return
                def _u8(im):
                    return (im * 255).clip(0, 255).astype(np.uint8) if im.dtype != np.uint8 else im
                s = _u8(obs['side_rgb'][-1]); w = _u8(obs['wrist_rgb'][-1])
                obs_writer.append_data(np.concatenate([s, w], axis=1))   # (H, 2W, 3) RGB

            def _close_obs_video():
                nonlocal obs_writer
                if obs_writer is not None:
                    obs_writer.close(); print(f"[obs video] saved -> {obs_vid_path}")
                    obs_writer = None

            pending_gripper_open_steps = 0  # set by 'r' to force gripper-open at next episode start
            with KeyPoller() as kp:
                while True:   # episode loop (re-entered on 'r' reset)
                    try:
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start)
                        precise_wait(eval_t_start, time_func=time.time)
                        print("Policy running. r=reset, q=quit (Ctrl-C also stops).")
                        last_raw_action = np.zeros(7, dtype=np.float32)
                        iter_idx = 0
                        # gripper macro: force action[6] to gripper_macro_value for N steps.
                        # On reset ('r') we seed open (+1); stuck detection toggles to the opposite
                        # of the current commanded gripper state.
                        gripper_macro_steps_remaining = pending_gripper_open_steps
                        gripper_macro_value = 1.0  # open
                        pending_gripper_open_steps = 0
                        stuck_buffer = []
                        while True:
                            t_cycle_end = t_start + (iter_idx + 1) * dt
                            obs = env_get_obs(env)
                            _record_obs(obs)

                            with torch.no_grad():
                                action = compute_action(obs, last_raw_action)
                            if const_act is not None:
                                action = const_act.copy()   # DEBUG: bypass policy

                            # Stuck detection: if arm joints barely move for STUCK_WINDOW_S,
                            # force the gripper to the OPPOSITE of its current commanded state
                            # (open -> close, closed -> open) for STUCK_GRIPPER_OPEN_STEPS to unstick.
                            if not no_stuck and gripper_macro_steps_remaining == 0:
                                t_now = time.monotonic()
                                stuck_buffer.append((t_now, obs['arm_joint_pos'][-1].copy()))
                                while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                    stuck_buffer.pop(0)
                                if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                    jps = np.array([b[1] for b in stuck_buffer])
                                    if (jps.max(0) - jps.min(0)).max() < STUCK_JOINT_THRESHOLD_RAD:
                                        # Toggle: last commanded gripper >=0 (open) -> close (-1); <0 (close) -> open (+1)
                                        gripper_macro_value = -1.0 if last_raw_action[6] >= 0 else 1.0
                                        gripper_macro_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                        stuck_buffer.clear()
                                        verb = "closing" if gripper_macro_value < 0 else "opening"
                                        print(f"[Stuck detection] No movement for 2s, {verb} gripper")

                            if gripper_macro_steps_remaining > 0:
                                action[6] = gripper_macro_value
                                gripper_macro_steps_remaining -= 1
                                if gripper_macro_steps_remaining == 0:
                                    print("[Gripper macro] done, returning to policy control")

                            last_raw_action = action.copy()  # becomes next prev_action

                            # OSC: scale delta, compute absolute target from observed pose
                            raw_arm = action[:6]
                            gripper = action[6:7]
                            scaled = raw_arm * CARTESIAN_SCALE

                            obs_jp = obs['arm_joint_pos'][-1]
                            obs_pos, obs_quat = get_ee_pose(obs_jp)
                            tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled)
                            tgt_aa = quat_to_axis_angle(tgt_quat)
                            abs_target = np.concatenate([tgt_pos, tgt_aa])[None]
                            target_actions = np.concatenate([abs_target, gripper[None]], axis=1)

                            # Schedule the target at the next FUTURE step boundary relative to
                            # eval_t_start (wall clock). Naive obs_ts+dt lands ~now (obs latency),
                            # giving the OSC interpolator ~no horizon -> arm barely moves.
                            curr_time = time.time()
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_ts = eval_t_start + next_step_idx * dt
                            if action_ts <= curr_time + 0.01:   # at least ~latency in the future
                                action_ts += dt
                            env.exec_actions(actions=target_actions,
                                             timestamps=np.array([action_ts]))

                            # visualize (window is display-only; control is via terminal keys)
                            vis_img = obs['side_rgb'][-1]
                            vis = (vis_img * 255).astype(np.uint8) if vis_img.dtype != np.uint8 else vis_img.copy()
                            cv2.putText(vis, f"t={time.monotonic()-t_start:.1f}s", (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.imshow('Student Eval', vis[..., ::-1]); cv2.waitKey(1)
                            k = kp.poll()
                            if k == 'q':
                                env.end_episode(); _close_obs_video(); return
                            elif k == 'r':
                                env.end_episode()
                                # reset_to_initial_position sends close_gripper=False, so the
                                # controller opens the gripper during the joint move.
                                env.robot.reset_to_initial_position()
                                time.sleep(5.0)
                                # Belt-and-suspenders: also force gripper open for the first
                                # second of the next episode via the existing macro path.
                                pending_gripper_open_steps = STUCK_GRIPPER_OPEN_STEPS
                                print("Reset (arm to init, gripper open); restarting episode.")
                                break

                            if time.monotonic() - t_start > max_duration:
                                env.end_episode(); print("Timeout; restarting episode."); break
                            precise_wait(t_cycle_end)
                            iter_idx += 1
                    except BaseException as e:  # incl. KeyboardInterrupt -> graceful stop
                        import traceback; traceback.print_exc()
                        env.end_episode(); _close_obs_video(); print(f"Interrupted: {e}"); break


if __name__ == '__main__':
    main()
