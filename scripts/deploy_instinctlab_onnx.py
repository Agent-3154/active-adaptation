"""Deploy an InstinctLab parkour ONNX checkpoint in active-adaptation sim.

Reuses the InstinctLab observation / action pipeline exactly, loading the
depth_encoder + actor ONNX models.  The robot is driven by randomly sampled
velocity commands that change every few seconds.  A RayCasterCamera provides
proper depth observations matching the training pipeline.

Usage (from active-adaptation/ root):
    python scripts/deploy_instinctlab_onnx.py \
        --ckpt_dir <path-to-exported-onnx-dir> \
        [--num_envs 4] [--headless] [--max_steps 10000]
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _find_default_ckpt_dir() -> str:
    candidates = [
        Path(__file__).resolve().parents[2]
        / "hiking-in-the-wild_Data&Model" / "data&model"
        / "checkpoints" / "parkour_onboard_preview_stair" / "exported",
        Path("/home/elgce/gsloco")
        / "hiking-in-the-wild_Data&Model" / "data&model"
        / "checkpoints" / "parkour_onboard_preview_stair" / "exported",
    ]
    for c in candidates:
        if c.is_dir() and (c / "actor.onnx").exists():
            return str(c)
    return str(candidates[0])


parser = argparse.ArgumentParser(description="Deploy InstinctLab ONNX in sim")
parser.add_argument("--ckpt_dir", type=str, default=_find_default_ckpt_dir())
parser.add_argument("--task", type=str, default="cfg/task/G1FlatDeploy.yaml")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--vx_range", type=float, nargs=2, default=[0.0, 1.0])
parser.add_argument("--vy_range", type=float, nargs=2, default=[-0.3, 0.3])
parser.add_argument("--wz_range", type=float, nargs=2, default=[-1.0, 1.0])
parser.add_argument("--cmd_resample_steps", type=int, default=200,
                    help="Resample random command every N sim steps")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# InstinctLab observation spec (must match the trained checkpoint)
# ---------------------------------------------------------------------------
NUM_JOINTS = 29
PROPRIO_HISTORY = 8
DEPTH_HISTORY_FRAMES = 8

DEPTH_RAW_H, DEPTH_RAW_W = 36, 64
CROP_TOP, CROP_BOT, CROP_LEFT, CROP_RIGHT = 18, 0, 16, 16
DEPTH_H = DEPTH_RAW_H - CROP_TOP - CROP_BOT   # 18
DEPTH_W = DEPTH_RAW_W - CROP_LEFT - CROP_RIGHT  # 32
DEPTH_MIN, DEPTH_MAX = 0.0, 2.5

OBS_SCALES = {"base_ang_vel": 0.25, "joint_vel": 0.05}

PROPRIO_SIZE = PROPRIO_HISTORY * (3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS)
DEPTH_FLAT_SIZE = DEPTH_HISTORY_FRAMES * DEPTH_H * DEPTH_W

# Camera extrinsics (relative to torso_link) from training env.yaml
CAM_POS = (0.0487988662332928, 0.01, 0.4378029937970051)
CAM_ROT_WXYZ = (0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0)
# Camera intrinsics from training env.yaml
CAM_HORIZONTAL_APERTURE = 1.9829684971963653
CAM_VERTICAL_APERTURE = 1.1152440419261633
CAM_FOCAL_LENGTH = 1.0

INSTINCTLAB_ACTION_SCALES = {
    "left_hip_yaw_joint": 0.5475464652142303, "right_hip_yaw_joint": 0.5475464652142303,
    "left_hip_roll_joint": 0.3506614663788243, "right_hip_roll_joint": 0.3506614663788243,
    "left_hip_pitch_joint": 0.5475464652142303, "right_hip_pitch_joint": 0.5475464652142303,
    "left_knee_joint": 0.3506614663788243, "right_knee_joint": 0.3506614663788243,
    "left_ankle_pitch_joint": 0.43857731392336724, "right_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724, "right_ankle_roll_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724, "waist_pitch_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464652142303,
    "left_shoulder_pitch_joint": 0.43857731392336724, "right_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724, "right_shoulder_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724, "right_shoulder_yaw_joint": 0.43857731392336724,
    "left_elbow_joint": 0.43857731392336724, "right_elbow_joint": 0.43857731392336724,
    "left_wrist_roll_joint": 0.43857731392336724, "right_wrist_roll_joint": 0.43857731392336724,
    "left_wrist_pitch_joint": 0.07450087032950714, "right_wrist_pitch_joint": 0.07450087032950714,
    "left_wrist_yaw_joint": 0.07450087032950714, "right_wrist_yaw_joint": 0.07450087032950714,
}


# ---------------------------------------------------------------------------
# ONNX policy
# ---------------------------------------------------------------------------
class OnnxPolicy:
    def __init__(self, ckpt_dir: str, device: str = "cuda:0"):
        import onnxruntime as ort

        encoder_path = os.path.join(ckpt_dir, "0-depth_encoder.onnx")
        actor_path = os.path.join(ckpt_dir, "actor.onnx")

        if not os.path.isfile(encoder_path) or not os.path.isfile(actor_path):
            raise FileNotFoundError(
                f"ONNX files not found in '{ckpt_dir}'.\n"
                f"  Expected: {encoder_path}\n"
                f"  Expected: {actor_path}\n"
                f"  Pass --ckpt_dir pointing to the 'exported/' folder."
            )

        providers = ort.get_available_providers()
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.actor = ort.InferenceSession(actor_path, providers=providers)
        self.device = device

        enc_in = self.encoder.get_inputs()[0]
        act_in = self.actor.get_inputs()[0]
        print(f"[ONNX] depth_encoder input: {enc_in.name} {enc_in.shape}")
        print(f"[ONNX] actor input: {act_in.name} {act_in.shape}")

    def __call__(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs_np = obs_flat.detach().cpu().float().numpy()
        B = obs_np.shape[0]
        proprio = obs_np[:, :PROPRIO_SIZE]
        depth_flat = obs_np[:, PROPRIO_SIZE:]
        depth_input = depth_flat.reshape(B, DEPTH_HISTORY_FRAMES, DEPTH_H, DEPTH_W)

        enc_name = self.encoder.get_inputs()[0].name
        act_name = self.actor.get_inputs()[0].name

        all_actions = []
        for i in range(B):
            enc_out = self.encoder.run(
                None, {enc_name: depth_input[i:i+1]}
            )[0]
            actor_in = np.concatenate([proprio[i:i+1], enc_out], axis=1)
            act_out = self.actor.run(None, {act_name: actor_in})[0]
            all_actions.append(act_out)

        actions = np.concatenate(all_actions, axis=0)
        return torch.from_numpy(actions).to(self.device)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
class ObservationBuilder:
    def __init__(self, num_envs: int, device):
        self.num_envs = num_envs
        self.device = device

        self.ang_vel_hist = torch.zeros(num_envs, PROPRIO_HISTORY, 3, device=device)
        self.gravity_hist = torch.zeros(num_envs, PROPRIO_HISTORY, 3, device=device)
        self.cmd_hist = torch.zeros(num_envs, PROPRIO_HISTORY, 3, device=device)
        self.jpos_hist = torch.zeros(num_envs, PROPRIO_HISTORY, NUM_JOINTS, device=device)
        self.jvel_hist = torch.zeros(num_envs, PROPRIO_HISTORY, NUM_JOINTS, device=device)
        self.act_hist = torch.zeros(num_envs, PROPRIO_HISTORY, NUM_JOINTS, device=device)
        self.depth_hist = torch.zeros(
            num_envs, DEPTH_HISTORY_FRAMES, DEPTH_H, DEPTH_W, device=device
        )
        self.default_joint_pos = None
        self.prev_action = torch.zeros(num_envs, NUM_JOINTS, device=device)

    def set_default_joint_pos(self, default_pos: torch.Tensor):
        self.default_joint_pos = default_pos.to(self.device)

    def _shift_and_push(self, buf: torch.Tensor, new_val: torch.Tensor) -> torch.Tensor:
        buf = torch.roll(buf, -1, dims=1)
        buf[:, -1] = new_val
        return buf

    def push_proprio(self, base_ang_vel, projected_gravity, velocity_commands,
                     joint_pos, joint_vel, last_action):
        joint_pos_rel = joint_pos
        if self.default_joint_pos is not None:
            joint_pos_rel = joint_pos - self.default_joint_pos

        self.ang_vel_hist = self._shift_and_push(
            self.ang_vel_hist, base_ang_vel * OBS_SCALES["base_ang_vel"])
        self.gravity_hist = self._shift_and_push(self.gravity_hist, projected_gravity)
        self.cmd_hist = self._shift_and_push(self.cmd_hist, velocity_commands)
        self.jpos_hist = self._shift_and_push(self.jpos_hist, joint_pos_rel)
        self.jvel_hist = self._shift_and_push(
            self.jvel_hist, joint_vel * OBS_SCALES["joint_vel"])
        self.act_hist = self._shift_and_push(self.act_hist, last_action)
        self.prev_action = last_action.clone()

    def push_depth(self, depth_frame: torch.Tensor):
        self.depth_hist = torch.roll(self.depth_hist, -1, dims=1)
        self.depth_hist[:, -1] = depth_frame

    def build_obs(self) -> torch.Tensor:
        parts = [
            self.ang_vel_hist.flatten(1),
            self.gravity_hist.flatten(1),
            self.cmd_hist.flatten(1),
            self.jpos_hist.flatten(1),
            self.jvel_hist.flatten(1),
            self.act_hist.flatten(1),
            self.depth_hist.flatten(1),
        ]
        return torch.cat(parts, dim=1)

    def reset(self, env_ids: torch.Tensor):
        self.ang_vel_hist[env_ids] = 0
        self.gravity_hist[env_ids] = 0
        self.cmd_hist[env_ids] = 0
        self.jpos_hist[env_ids] = 0
        self.jvel_hist[env_ids] = 0
        self.act_hist[env_ids] = 0
        self.depth_hist[env_ids] = 0
        self.prev_action[env_ids] = 0


# ---------------------------------------------------------------------------
# Quaternion utilities (w, x, y, z)
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ], dim=-1)


def _quat_inv(q):
    inv = q.clone()
    inv[:, 1:] = -inv[:, 1:]
    return inv


def _projected_gravity(root_quat_wxyz):
    grav = torch.tensor([[0.0, 0.0, -1.0]], device=root_quat_wxyz.device)
    grav = grav.expand(root_quat_wxyz.shape[0], 3)
    return _quat_rotate(_quat_inv(root_quat_wxyz), grav)


# ---------------------------------------------------------------------------
# Depth processing (matches training noise_pipeline exactly)
# ---------------------------------------------------------------------------
def process_raycaster_depth(raw_depth: torch.Tensor) -> torch.Tensor:
    """Process raw (N, 36, 64) depth to match InstinctLab training pipeline.

    Pipeline order (from env.yaml noise_pipeline):
      1. crop_and_resize: crop_region=(18, 0, 16, 16)  -> (N, 18, 32)
      2. gaussian_blur: kernel=3, sigma=1
      3. depth_normalization: clip [0, 2.5], normalize to [0, 1]
    """
    h_end = DEPTH_RAW_H - CROP_BOT if CROP_BOT > 0 else DEPTH_RAW_H
    w_end = DEPTH_RAW_W - CROP_RIGHT if CROP_RIGHT > 0 else DEPTH_RAW_W
    cropped = raw_depth[:, CROP_TOP:h_end, CROP_LEFT:w_end]  # (N, 18, 32)

    # Gaussian blur (kernel=3, sigma=1) matching training
    cropped_4d = cropped.unsqueeze(1)  # (N, 1, 18, 32)
    kernel_size = 3
    sigma = 1.0
    x = torch.arange(kernel_size, device=raw_depth.device, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    blurred = F.conv2d(cropped_4d, kernel, padding=kernel_size // 2)
    blurred = blurred.squeeze(1)  # (N, 18, 32)

    blurred = blurred.clamp(DEPTH_MIN, DEPTH_MAX)
    normalized = (blurred - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)
    return normalized


# ---------------------------------------------------------------------------
# Random command sampler
# ---------------------------------------------------------------------------
class RandomCommandSampler:
    def __init__(self, num_envs: int, device,
                 vx_range: tuple[float, float],
                 vy_range: tuple[float, float],
                 wz_range: tuple[float, float],
                 resample_interval: int):
        self.num_envs = num_envs
        self.device = device
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.wz_range = wz_range
        self.resample_interval = resample_interval

        self.commands = torch.zeros(num_envs, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._resample(torch.arange(num_envs, device=device))

    def _resample(self, env_ids: torch.Tensor):
        n = env_ids.shape[0]
        vx = torch.empty(n, device=self.device).uniform_(*self.vx_range)
        vy = torch.empty(n, device=self.device).uniform_(*self.vy_range)
        wz = torch.empty(n, device=self.device).uniform_(*self.wz_range)
        self.commands[env_ids] = torch.stack([vx, vy, wz], dim=-1)
        self.step_counter[env_ids] = 0

    def step(self) -> torch.Tensor:
        self.step_counter += 1
        resample_ids = (self.step_counter >= self.resample_interval).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample(resample_ids)
        return self.commands

    def reset(self, env_ids: torch.Tensor):
        self._resample(env_ids)


# ---------------------------------------------------------------------------
# Deploy env: subclass SimpleEnvIsaac to inject RayCasterCamera into the
# scene config *before* InteractiveScene is constructed and sim.reset().
# ---------------------------------------------------------------------------
class DeployEnvIsaac:
    """SimpleEnvIsaac with a RayCasterCamera for depth observations."""

    def __init__(self, task_cfg, device: str, headless: bool):
        from active_adaptation.envs.locomotion import SimpleEnvIsaac

        _orig_setup = SimpleEnvIsaac.setup_scene

        def _patched_setup(env_self):
            """Monkey-patched setup_scene that injects RayCasterCameraCfg."""
            from isaaclab.sensors import RayCasterCameraCfg
            from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
            from isaaclab.scene import InteractiveSceneCfg, InteractiveScene

            # Let original build the scene_cfg up to but not including
            # InteractiveScene construction. We intercept by patching
            # InteractiveScene.__init__.
            captured = {}
            _orig_scene_init = InteractiveScene.__init__

            def _intercept_init(scene_self, cfg, *a, **kw):
                # Inject depth camera into the cfg before it's built
                cfg.depth_camera = RayCasterCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                    mesh_prim_paths=["/World/ground"],
                    update_period=0.0,
                    offset=RayCasterCameraCfg.OffsetCfg(
                        pos=CAM_POS, rot=CAM_ROT_WXYZ, convention="world",
                    ),
                    pattern_cfg=PinholeCameraPatternCfg(
                        focal_length=CAM_FOCAL_LENGTH,
                        horizontal_aperture=CAM_HORIZONTAL_APERTURE,
                        vertical_aperture=CAM_VERTICAL_APERTURE,
                        width=DEPTH_RAW_W,
                        height=DEPTH_RAW_H,
                    ),
                    data_types=["distance_to_image_plane"],
                    debug_vis=False,
                )
                _orig_scene_init(scene_self, cfg, *a, **kw)
                captured["scene"] = scene_self

            InteractiveScene.__init__ = _intercept_init
            try:
                _orig_setup(env_self)
            finally:
                InteractiveScene.__init__ = _orig_scene_init

            self._camera = captured["scene"].sensors.get("depth_camera")
            if self._camera is not None:
                print(f"[INFO] RayCasterCamera injected: {DEPTH_RAW_H}x{DEPTH_RAW_W}")
            else:
                print("[WARN] RayCasterCamera injection failed, using fallback depth.")

        SimpleEnvIsaac.setup_scene = _patched_setup
        try:
            self.env = SimpleEnvIsaac(task_cfg, device, headless=headless)
        finally:
            SimpleEnvIsaac.setup_scene = _orig_setup

    @property
    def camera(self):
        return getattr(self, "_camera", None)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_depth(self) -> torch.Tensor:
        """Read raw depth (N, 36, 64) from the raycaster camera."""
        self._camera.update(dt=self.env.physics_dt)
        raw = self._camera.data.output["distance_to_image_plane"]  # (N, 36, 64, 1)
        raw = raw.squeeze(-1)  # (N, 36, 64)
        raw = raw.clamp(0.0, 1e6)
        raw[raw >= 1e5] = DEPTH_MAX
        return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from omegaconf import OmegaConf
    import active_adaptation as aa

    cfg_path = Path(args.task)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path

    task_cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(task_cfg)
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.num_envs = args.num_envs

    full_cfg = OmegaConf.create({
        "backend": "isaac",
        "device": args.device,
        "headless": args.headless,
        "app": {"headless": args.headless, "enable_cameras": not args.headless},
        "task": task_cfg,
    })

    aa.init(full_cfg, auto_rank=True)

    deploy_env = DeployEnvIsaac(full_cfg.task, str(args.device), headless=args.headless)
    env = deploy_env.env
    robot = env.scene.articulations["robot"]
    num_envs = env.num_envs
    device = env.device

    joint_names = robot.data.joint_names
    default_joint_pos = robot.data.default_joint_pos[0, :NUM_JOINTS].clone()

    action_scale = torch.zeros(NUM_JOINTS, device=device)
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        action_scale[i] = INSTINCTLAB_ACTION_SCALES.get(jn, 0.5)

    has_camera = deploy_env.camera is not None
    print(f"[INFO] Depth source: {'RayCasterCamera' if has_camera else 'constant (fallback)'}")

    policy = OnnxPolicy(args.ckpt_dir, device=str(device))
    obs_builder = ObservationBuilder(num_envs, device)
    obs_builder.set_default_joint_pos(default_joint_pos)

    cmd_sampler = RandomCommandSampler(
        num_envs, device,
        vx_range=tuple(args.vx_range),
        vy_range=tuple(args.vy_range),
        wz_range=tuple(args.wz_range),
        resample_interval=args.cmd_resample_steps,
    )

    env.reset()
    obs_builder.reset(torch.arange(num_envs, device=device))

    print(f"[INFO] ckpt: {args.ckpt_dir}")
    print(f"[INFO] num_envs: {num_envs}, device: {device}")
    print(f"[INFO] vx_range: {args.vx_range}, vy_range: {args.vy_range}, wz_range: {args.wz_range}")
    print(f"[INFO] cmd resample every {args.cmd_resample_steps} steps")
    print(f"[INFO] Joint order ({NUM_JOINTS} joints):")
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        print(f"  {i:2d}: {jn:35s}  default={default_joint_pos[i]:.3f}  scale={action_scale[i]:.4f}")

    for step in range(args.max_steps):
        with torch.inference_mode():
            root_quat_w = robot.data.root_quat_w
            joint_pos = robot.data.joint_pos[:, :NUM_JOINTS]
            joint_vel = robot.data.joint_vel[:, :NUM_JOINTS]

            base_ang_vel_b = _quat_rotate(
                _quat_inv(root_quat_w), robot.data.root_com_ang_vel_w
            )
            proj_grav = _projected_gravity(root_quat_w)

            vel_cmd = cmd_sampler.step()

            # ---- depth from raycaster camera ----
            if has_camera:
                raw_depth = deploy_env.get_depth()
                depth_frame = process_raycaster_depth(raw_depth)
            else:
                depth_frame = torch.ones(num_envs, DEPTH_H, DEPTH_W, device=device) * 0.5
            obs_builder.push_depth(depth_frame)

            obs_builder.push_proprio(
                base_ang_vel=base_ang_vel_b,
                projected_gravity=proj_grav,
                velocity_commands=vel_cmd,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                last_action=obs_builder.prev_action,
            )

            obs_flat = obs_builder.build_obs()
            raw_actions = policy(obs_flat)
            obs_builder.prev_action = raw_actions.clone()

            target_pos = raw_actions * action_scale + default_joint_pos.unsqueeze(0)
            full_target = robot.data.default_joint_pos.clone()
            full_target[:, :NUM_JOINTS] = target_pos
            robot.set_joint_position_target(full_target)

            for _ in range(env.decimation):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)

            if not args.headless:
                env.sim.render()

            # ---- reset fallen robots ----
            root_pos = robot.data.root_pos_w
            fallen = root_pos[:, 2] < 0.3
            fallen_ids = fallen.nonzero(as_tuple=False).flatten()
            if fallen_ids.numel() > 0:
                env._reset_idx(fallen_ids)
                obs_builder.reset(fallen_ids)
                cmd_sampler.reset(fallen_ids)
                robot.reset(fallen_ids)

            if step % 100 == 0:
                root_vel = robot.data.root_com_lin_vel_w[0]
                cmd = vel_cmd[0]
                act = raw_actions[0]
                d = depth_frame[0]
                print(
                    f"Step {step:5d} | "
                    f"pos=({root_pos[0, 0]:.2f}, {root_pos[0, 1]:.2f}, {root_pos[0, 2]:.2f}) | "
                    f"vel=({root_vel[0]:.2f}, {root_vel[1]:.2f}, {root_vel[2]:.2f}) | "
                    f"cmd=({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}) | "
                    f"depth=[{d.min():.2f}, {d.mean():.2f}, {d.max():.2f}]"
                )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
