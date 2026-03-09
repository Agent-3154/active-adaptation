"""
Play and visualize a policy in the environment.

Recording modes:
  --record      Isaac Sim viewport (robot motion in sim), needs enable_cameras
  --record-gs   GS-rendered third-person + first-person, fully headless
"""

import math
import os
import torch
import hydra
import itertools
import datetime
import copy
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.learning.modules.vecnorm import VecNorm

FILE_PATH = Path(__file__).parent


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu()
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"policy-{time_str}.onnx"
    export_onnx(deploy_policy, fake_input, str(path))


def _save_video(frames, path, fps):
    """Encode frames to MP4 via torchvision."""
    from torchvision.io import write_video
    video_array = np.stack(frames)
    write_video(str(path), video_array=video_array, fps=fps, video_codec="h264")
    print(f"[record] Saved {len(frames)} frames -> {path}")


def _follow_camera_eye(robot_pos, offset=(3.0, 3.0, 2.0), smooth_state=None, alpha=0.1):
    """Fixed-offset camera that always looks at the robot."""
    rx, ry, rz = robot_pos
    target = (rx, ry, rz + 0.5)
    ex = rx + offset[0]
    ey = ry + offset[1]
    ez = rz + offset[2]

    if smooth_state is not None:
        sex, sey, sez = smooth_state
        ex = sex * (1 - alpha) + ex * alpha
        ey = sey * (1 - alpha) + ey * alpha
        ez = sez * (1 - alpha) + ez * alpha

    return (ex, ey, ez), target, (ex, ey, ez)


# ---- HUD overlay helpers ----

def _draw_hud(frame, traj_xy, vel_xy, cmd, step, dt, max_speed=1.0):
    """Draw trajectory minimap + velocity/command info on a video frame.

    Args:
        frame: (H, W, 3) uint8 numpy array, modified in-place and returned.
        traj_xy: list of (x, y) positions so far.
        vel_xy: (vx, vy) current velocity.
        cmd: (cmd_vx, cmd_vy, cmd_yaw, cmd_h) command.
        step: current env step.
        dt: env step dt.
    """
    from PIL import Image, ImageDraw, ImageFont

    H, W = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font

    # -- trajectory minimap (bottom-left) --
    map_size = min(180, H // 3)
    map_margin = 12
    mx0 = map_margin
    my0 = H - map_size - map_margin

    draw.rectangle([mx0, my0, mx0 + map_size, my0 + map_size], fill=(0, 0, 0, 160))

    if len(traj_xy) >= 2:
        xs = [p[0] for p in traj_xy]
        ys = [p[1] for p in traj_xy]
        cx, cy = xs[-1], ys[-1]
        view_range = max(3.0, max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6 + 1.0)

        def to_map(wx, wy):
            px = int(mx0 + (wx - cx + view_range) / (2 * view_range) * map_size)
            py = int(my0 + map_size - (wy - cy + view_range) / (2 * view_range) * map_size)
            return (px, py)

        trail_pts = [to_map(p[0], p[1]) for p in traj_xy]
        for j in range(1, len(trail_pts)):
            alpha = int(100 + 155 * j / len(trail_pts))
            draw.line([trail_pts[j-1], trail_pts[j]], fill=(0, 200, 255, alpha), width=2)

        cur = trail_pts[-1]
        draw.ellipse([cur[0]-4, cur[1]-4, cur[0]+4, cur[1]+4], fill=(255, 80, 80, 230))

        # velocity arrow
        vx, vy = vel_xy
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > 0.05:
            arrow_scale = map_size * 0.3 / max(max_speed, speed)
            ax = int(cur[0] + vx * arrow_scale)
            ay = int(cur[1] - vy * arrow_scale)
            draw.line([cur, (ax, ay)], fill=(255, 255, 0, 220), width=2)

    # -- text HUD (top-left) --
    vx, vy = vel_xy
    speed = math.sqrt(vx*vx + vy*vy)
    t_sec = step * dt

    lines = [
        f"t={t_sec:6.1f}s  step={step}",
        f"vel=({vx:+.2f}, {vy:+.2f})  |v|={speed:.2f}",
        f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f})  yaw={cmd[2]:+.2f}",
    ]

    tx, ty = 10, 8
    line_h = 20
    bg_w = 340
    bg_h = len(lines) * line_h + 12
    draw.rectangle([tx-4, ty-4, tx + bg_w, ty + bg_h], fill=(0, 0, 0, 160))
    for line in lines:
        draw.text((tx, ty), line, fill=(220, 220, 220, 255), font=font)
        ty += line_h

    result = np.array(img)
    frame[:] = result[:, :, :3]
    return frame


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=False)
    
    from helpers import make_env_policy
    env, policy = make_env_policy(cfg)
    
    if cfg.export_policy:
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_policy(env, policy, export_dir)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    carry = env.reset()
    
    assert not env.base_env.training

    # ---- Isaac sim recording config (--record) ----
    record_sim = cfg.get("record_sim", False)
    record_steps = cfg.get("record_steps", 3000)
    record_fps = cfg.get("record_fps", 30)
    record_interval = max(1, round(1.0 / (env.step_dt * record_fps)))
    record_env = cfg.get("record_env", 0)
    record_dir = Path(cfg.get("record_dir", "recordings"))
    sim_frames = []
    cam_smooth = None
    traj_xy = []

    if record_sim:
        record_dir.mkdir(parents=True, exist_ok=True)
        effective_fps = 1.0 / (env.step_dt * record_interval)
        print(f"[record-sim] Isaac viewport recording @ {effective_fps:.0f}fps, "
              f"env={record_env}, {record_steps} steps")

    # ---- GS recording config (--record-gs) ----
    record_gs = cfg.get("record_gs", False)
    record_gs_fps = cfg.get("record_gs_fps", 30)
    record_gs_interval = max(1, round(1.0 / (env.step_dt * record_gs_fps)))
    record_gs_width = cfg.get("record_gs_width", 1280)
    record_gs_height = cfg.get("record_gs_height", 720)
    record_gs_steps = cfg.get("record_gs_steps", 3000)
    record_gs_env = cfg.get("record_gs_env", 0)
    record_gs_dir = Path(cfg.get("record_gs_dir", "recordings"))
    tp_frames = []
    fp_frames = []

    if record_gs:
        record_gs_dir.mkdir(parents=True, exist_ok=True)
        effective_fps = 1.0 / (env.step_dt * record_gs_interval)
        print(f"[record-gs] GS video: {record_gs_width}x{record_gs_height} @ {effective_fps:.0f}fps, "
              f"env={record_gs_env}, {record_gs_steps} steps")

    any_record = record_sim or record_gs
    max_steps = max(
        record_steps if record_sim else 0,
        record_gs_steps if record_gs else 0,
    )

    vis_gs_rgb = cfg.get("vis_gs_rgb", False)
    vis_gs_rgb_interval = cfg.get("vis_gs_rgb_interval", 5)
    vis_gs_rgb_port = cfg.get("vis_gs_rgb_port", 8890)
    viser_img_handle = None
    if vis_gs_rgb:
        import viser
        viser_server = viser.ViserServer(port=vis_gs_rgb_port)
        print(f"[vis_gs_rgb] Open http://localhost:{vis_gs_rgb_port} to see env0 GS RGB")

    dump_gs_imgs = cfg.get("dump_gs_imgs", False)
    dump_gs_imgs_interval = cfg.get("dump_gs_imgs_interval", 5)
    dump_gs_imgs_dir = cfg.get("dump_gs_imgs_dir", "gs_dump")
    if dump_gs_imgs:
        os.makedirs(dump_gs_imgs_dir, exist_ok=True)
        print(f"[dump_gs] Will save all-env GS images every {dump_gs_imgs_interval} steps -> {dump_gs_imgs_dir}")

    use_timer = not any_record
    timer = Timer(env.step_dt)

    with set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            with torch.inference_mode():
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)

            episode_stats.add(td)

            base = env.base_env if hasattr(env, "base_env") else env
            if i % 50 == 0:
                root_vel = base.scene["robot"].data.root_com_lin_vel_w[0, :2]
                cmd = base.command_manager.command[0]
                speed = root_vel.norm().item()
                print(f"[step {i:5d}] vel=({root_vel[0].item():+.2f}, {root_vel[1].item():+.2f}) |v|={speed:.2f}  "
                      f"cmd=({cmd[0].item():+.2f}, {cmd[1].item():+.2f}, yaw={cmd[2].item():+.2f}, h={cmd[3].item():.2f})")

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            # ---- record Isaac sim viewport frames ----
            if record_sim and i % record_interval == 0 and i < record_steps:
                robot = base.scene["robot"]
                pos = robot.data.root_link_pos_w[record_env].cpu().tolist()
                vel = robot.data.root_com_lin_vel_w[record_env, :2].cpu().tolist()
                cmd_t = base.command_manager.command[record_env].cpu().tolist()

                traj_xy.append((pos[0], pos[1]))

                eye, target, cam_smooth = _follow_camera_eye(
                    pos, smooth_state=cam_smooth,
                )
                sim = base.sim if hasattr(base, "sim") else None
                if sim is not None and hasattr(sim, "set_camera_view"):
                    sim.set_camera_view(eye, target)

                frame = base.render("rgb_array")
                if frame is not None:
                    frame = frame.copy()
                    _draw_hud(frame, traj_xy, vel, cmd_t, i, env.step_dt)
                    sim_frames.append(frame)

                if i % 200 == 0:
                    print(f"[record-sim] step {i}/{record_steps}, {len(sim_frames)} frames")

            # ---- record GS frames ----
            if record_gs and i % record_gs_interval == 0 and i < record_gs_steps:
                if hasattr(base, "render_third_person_gs"):
                    tp = base.render_third_person_gs(
                        env_id=record_gs_env,
                        width=record_gs_width,
                        height=record_gs_height,
                    )
                    if tp is not None:
                        tp_frames.append(tp)

                if hasattr(base, "debug_gs_render"):
                    fp = base.debug_gs_render(
                        env_id=record_gs_env,
                        width=record_gs_width // 3,
                        height=record_gs_height // 3,
                    )
                    if fp is not None:
                        fp_frames.append(fp)

                if i % 200 == 0:
                    print(f"[record-gs] step {i}/{record_gs_steps}, {len(tp_frames)} frames")

            if any_record and i >= max_steps:
                break

            if vis_gs_rgb and i % vis_gs_rgb_interval == 0:
                if hasattr(base, "debug_gs_render"):
                    rgb_np = base.debug_gs_render(env_id=0)
                    if rgb_np is not None:
                        if viser_img_handle is None:
                            viser_img_handle = viser_server.scene.add_image(
                                "/gs_rgb_env0",
                                rgb_np,
                                render_width=2.0,
                                render_height=2.0 * rgb_np.shape[0] / rgb_np.shape[1],
                            )
                        else:
                            viser_img_handle.image = rgb_np

            if dump_gs_imgs and i % dump_gs_imgs_interval == 0:
                if hasattr(base, "debug_dump_all_gs"):
                    base.debug_dump_all_gs(out_dir=dump_gs_imgs_dir, step=i)

            if use_timer:
                timer.sleep()

    # ---- save recorded videos ----
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")

    if record_sim and sim_frames:
        effective_fps = 1.0 / (env.step_dt * record_interval)
        sim_path = record_dir / f"sim_{time_str}.mp4"
        _save_video(sim_frames, sim_path, fps=int(effective_fps))

    if record_gs and tp_frames:
        effective_fps = 1.0 / (env.step_dt * record_gs_interval)

        tp_path = record_gs_dir / f"third_person_{time_str}.mp4"
        _save_video(tp_frames, tp_path, fps=int(effective_fps))

        if fp_frames:
            fp_path = record_gs_dir / f"first_person_{time_str}.mp4"
            _save_video(fp_frames, fp_path, fps=int(effective_fps))

        if fp_frames and len(fp_frames) == len(tp_frames):
            composite_frames = []
            fh, fw = fp_frames[0].shape[:2]
            margin = 10
            for tp_f, fp_f in zip(tp_frames, fp_frames):
                comp = tp_f.copy()
                y0 = margin
                x0 = comp.shape[1] - fw - margin
                comp[y0:y0+fh, x0:x0+fw] = fp_f
                composite_frames.append(comp)
            comp_path = record_gs_dir / f"composite_{time_str}.mp4"
            _save_video(composite_frames, comp_path, fps=int(effective_fps))

    # Isaac Sim's OmniGraph plugin segfaults during atexit cleanup
    # when replicator cameras are used headless. Force exit after saving.
    if any_record:
        print("[record] Done. Force-exiting to avoid Isaac shutdown crash.")
        os._exit(0)

    env.close()


if __name__ == "__main__":
    main()

