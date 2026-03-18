"""Closed-loop deploy of InstinctLab parkour ONNX policy in a GS scene.

The robot walks inside a Gaussian-Splatting mesh.  At each policy step the
script renders an RGB image from the GS scene at the robot's camera pose,
runs Depth Anything V3 (DA3) to produce a depth map, then crops / normalizes
the depth to match the InstinctLab training pipeline before feeding it (along
with proprioception) to the ONNX policy.

Usage (from active-adaptation/ root):
    python scripts/deploy_instinctlab_gs.py \
        [--ckpt_dir <path-to-exported-onnx-dir>] \
        [--task cfg/task/G1GsDeploy.yaml] \
        [--num_envs 1] [--headless] [--vx 0.5]
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_ckpt_dir() -> str:
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


parser = argparse.ArgumentParser(description="Deploy InstinctLab ckpt in GS scene")
parser.add_argument("--ckpt_dir", type=str, default=_default_ckpt_dir())
parser.add_argument("--task", type=str, default="cfg/task/G1GsDeploy.yaml")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--vx", type=float, default=0.5,
                    help="Fallback constant vx when no spawn points available")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--episode_len", type=int, default=500,
                    help="Max steps per episode before auto-reset")
parser.add_argument("--max_speed", type=float, default=0.5,
                    help="Max walking speed along waypoint path")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-SMALL")
parser.add_argument("--depth_mode", type=str, default="da3",
                    choices=["da3", "gs_render"],
                    help="Depth source: 'da3' = DA3 prediction on GS RGB, "
                         "'gs_render' = direct GS expected-depth rendering")
parser.add_argument("--render_size", type=int, default=0,
                    help="(deprecated, ignored -- render resolution is fixed to match training camera)")
parser.add_argument("--record", action="store_true",
                    help="Headless record mode: run record_steps, save composites, then exit")
parser.add_argument("--record_steps", type=int, default=2000)
parser.add_argument("--diag_video_dir", type=str, default="output/diag",
                    help="Directory for composite videos (saved every diag_interval steps)")
parser.add_argument("--diag_interval", type=int, default=500,
                    help="Save composite video every N steps")
args = parser.parse_args()

if args.record:
    args.headless = True
    args.max_steps = args.record_steps


# ---------------------------------------------------------------------------
# InstinctLab observation spec (must match trained checkpoint exactly)
# ---------------------------------------------------------------------------
NUM_JOINTS = 29
PROPRIO_HISTORY = 8
DEPTH_HISTORY_FRAMES = 8

DEPTH_RAW_H, DEPTH_RAW_W = 36, 64
CROP_TOP, CROP_BOT, CROP_LEFT, CROP_RIGHT = 18, 0, 16, 16
DEPTH_H = DEPTH_RAW_H - CROP_TOP - CROP_BOT   # 18
DEPTH_W = DEPTH_RAW_W - CROP_LEFT - CROP_RIGHT  # 32
DEPTH_MIN, DEPTH_MAX = 0.0, 2.5

# Camera extrinsics relative to torso_link (from InstinctLab training config)
CAM_OFFSET_POS = (0.0487988662332928, 0.01, 0.4378029937970051)
# Camera offset quaternion (w,x,y,z) with convention="world".
# This encodes the full rotation from body frame to the camera frame INCLUDING
# the ~48 deg downward pitch. In gsplat terms this is the cam-to-body quat.
CAM_OFFSET_QUAT_WXYZ = (
    0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0
)
# InstinctLab camera: focal_length=1.0, horizontal_aperture=2*tan(89.51°/2), width=64, height=36
# fov_h=89.51°, fov_v=58.29°
CAM_FOV_H_DEG = 89.51
CAM_FOV_V_DEG = 58.29
CAM_FOV_V_RAD = math.radians(CAM_FOV_V_DEG)

OBS_SCALES = {"base_ang_vel": 0.25, "joint_vel": 0.05}

PROPRIO_SIZE = PROPRIO_HISTORY * (3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS)
DEPTH_FLAT_SIZE = DEPTH_HISTORY_FRAMES * DEPTH_H * DEPTH_W
TOTAL_OBS_SIZE = PROPRIO_SIZE + DEPTH_FLAT_SIZE

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
                f"ONNX not found in '{ckpt_dir}'.\n"
                f"  Need: {encoder_path}\n  Need: {actor_path}"
            )

        providers = ort.get_available_providers()
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.actor = ort.InferenceSession(actor_path, providers=providers)
        self.device = device

        enc_in = self.encoder.get_inputs()[0]
        act_in = self.actor.get_inputs()[0]
        print(f"[ONNX] encoder input: {enc_in.name} {enc_in.shape}")
        print(f"[ONNX] actor   input: {act_in.name} {act_in.shape}")

    def __call__(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Run encoder + actor per env (ONNX exported with batch=1)."""
        obs_np = obs_flat.detach().cpu().float().numpy()
        B = obs_np.shape[0]
        proprio = obs_np[:, :PROPRIO_SIZE]
        depth_input = obs_np[:, PROPRIO_SIZE:].reshape(
            B, DEPTH_HISTORY_FRAMES, DEPTH_H, DEPTH_W
        )

        enc_name = self.encoder.get_inputs()[0].name
        act_name = self.actor.get_inputs()[0].name

        all_actions = []
        for i in range(B):
            enc_out = self.encoder.run(None, {enc_name: depth_input[i:i+1]})[0]
            actor_in = np.concatenate([proprio[i:i+1], enc_out], axis=1)
            act_out = self.actor.run(None, {act_name: actor_in})[0]
            all_actions.append(act_out)

        return torch.from_numpy(np.concatenate(all_actions, 0)).to(self.device)


# ---------------------------------------------------------------------------
# Observation builder (identical to the flat-deploy version)
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

    def _shift_and_push(self, buf: torch.Tensor, new_val: torch.Tensor):
        buf = torch.roll(buf, -1, dims=1)
        buf[:, -1] = new_val
        return buf

    def push_proprio(self, base_ang_vel, projected_gravity, velocity_commands,
                     joint_pos, joint_vel, last_action):
        jp_rel = joint_pos
        if self.default_joint_pos is not None:
            jp_rel = joint_pos - self.default_joint_pos

        self.ang_vel_hist = self._shift_and_push(
            self.ang_vel_hist, base_ang_vel * OBS_SCALES["base_ang_vel"])
        self.gravity_hist = self._shift_and_push(self.gravity_hist, projected_gravity)
        self.cmd_hist = self._shift_and_push(self.cmd_hist, velocity_commands)
        self.jpos_hist = self._shift_and_push(self.jpos_hist, jp_rel)
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
# Quaternion utilities (w, x, y, z) convention
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ], dim=-1)


def _quat_inv(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _projected_gravity(root_quat_wxyz):
    grav = torch.tensor([[0.0, 0.0, -1.0]], device=root_quat_wxyz.device)
    grav = grav.expand(root_quat_wxyz.shape[0], 3)
    return _quat_rotate(_quat_inv(root_quat_wxyz), grav)


# ---------------------------------------------------------------------------
# GS camera pose computation
# ---------------------------------------------------------------------------
def _build_camera_world_pose(torso_pos_w, torso_quat_wxyz, device):
    """Camera world position from InstinctLab offset relative to torso_link.

    Returns (cam_pos_w, cam_quat_w) both in (w, x, y, z).
    """
    B = torso_pos_w.shape[0]
    off_pos = torch.tensor(CAM_OFFSET_POS, device=device).expand(B, 3)
    cam_pos_w = torso_pos_w + _quat_rotate(torso_quat_wxyz, off_pos)

    off_q = torch.tensor(CAM_OFFSET_QUAT_WXYZ, device=device).expand(B, 4)
    cam_quat_w = _quat_mul(torso_quat_wxyz, off_q)
    return cam_pos_w, cam_quat_w


# ---------------------------------------------------------------------------
# GS RGB renderer (pitched camera matching InstinctLab training)
# ---------------------------------------------------------------------------
class GsRgbRenderer:
    """Render RGB from the GS scene using the InstinctLab camera's pitched pose.

    The camera is oriented with CAM_OFFSET_QUAT_WXYZ applied to the torso body
    frame. This quaternion includes the ~48 deg downward pitch exactly matching
    the InstinctLab training camera, so the rendered RGB sees the same scene
    geometry as the training depth sensor.

    Render resolution uses 16:9 aspect (matching 64:36 training camera) at a
    DA3-compatible size (both dims divisible by 14).
    """

    RENDER_W = 448   # 64/36 * 252 = 448; 448 % 14 == 0
    RENDER_H = 252   # 252 % 14 == 0

    def __init__(self, env, device: str):
        self.device = device
        self.terrain = env.scene.terrain

        self._has_gs = (
            hasattr(self.terrain, "render")
            and hasattr(self.terrain, "gs_scene")
            and self.terrain.gs_scene is not None
        )
        if not self._has_gs:
            print("[WARN] No GS scene gaussians loaded. Depth will be zeros.")
            return

        ha = 2.0 * math.tan(math.radians(CAM_FOV_H_DEG) / 2.0)
        va = 2.0 * math.tan(math.radians(CAM_FOV_V_DEG) / 2.0)
        fx = self.RENDER_W * 1.0 / ha
        fy = self.RENDER_H * 1.0 / va
        self._K = torch.tensor([
            [fx,  0.0, self.RENDER_W / 2.0],
            [0.0, fy,  self.RENDER_H / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=device)

        # gsplat cam-to-body quat: start from _get_body_to_cam_quat (forward-looking,
        # no pitch) and add InstinctLab's ~48° downward pitch in the cam frame.
        # Pitch down in gsplat (X-right,Y-down,Z-fwd) = rotate -48° around cam X.
        from gs_scene.gs_env import _get_body_to_cam_quat
        c2b_flat = _get_body_to_cam_quat(device)
        # b2c quat rotates image content CCW 90°; compensate with +90° around cam Z
        # cz, sz = math.cos(math.pi / 4), math.sin(math.pi / 4)
        # fix_roll = torch.tensor([cz, 0.0, 0.0, sz], dtype=torch.float32, device=device)
        # c2b_flat = _quat_mul(c2b_flat, fix_roll)
        # TODO: uncomment to add 48° downward pitch once forward-looking is confirmed
        pitch_rad = math.radians(-48.0)
        cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
        pitch_q = torch.tensor([cp, sp, 0.0, 0.0], dtype=torch.float32, device=device)
        c2b_flat = _quat_mul(c2b_flat, pitch_q)
        self._c2b = c2b_flat
        print(f"[INFO] GS renderer ready ({self.RENDER_W}x{self.RENDER_H}, "
              f"pitched camera, fx={fx:.1f} fy={fy:.1f})")

    @property
    def has_gs(self) -> bool:
        return self._has_gs

    @torch.no_grad()
    def render(self, torso_pos_w, torso_quat_wxyz, cam_pos_w):
        """Render RGB (B, 3, H, W) in [0,1] from the GS scene."""
        B = torso_pos_w.shape[0]
        w, h = self.RENDER_W, self.RENDER_H

        if not self._has_gs:
            return torch.zeros(B, 3, h, w, device=self.device)

        c2b = self._c2b.unsqueeze(0).expand(B, 4)
        gs_cam_quat = _quat_mul(torso_quat_wxyz, c2b)

        offsets = (
            self.terrain.sub_mesh_offsets[:B]
            if hasattr(self.terrain, "sub_mesh_offsets")
            else torch.zeros(B, 3, device=self.device)
        )
        gs_cam_pos = cam_pos_w - offsets
        Ks = self._K.unsqueeze(0).expand(B, 3, 3)

        rgb_hwc = torch.zeros(B, h, w, 3, device=self.device)
        for i in range(B):
            try:
                sl = (
                    self.terrain.slice_indices[i:i+1]
                    if hasattr(self.terrain, "slice_indices")
                    else None
                )
                rgb_hwc[i:i+1] = self.terrain.render(
                    camera_pos_w=gs_cam_pos[i:i+1],
                    camera_quat_w=gs_cam_quat[i:i+1],
                    Ks=Ks[i:i+1],
                    width=w,
                    height=h,
                    slice_indices=sl,
                )
            except Exception as e:
                print(f"[WARN] GS render failed env {i}: {e}")

        return rgb_hwc.permute(0, 3, 1, 2).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# DA3 depth estimation + InstinctLab-aligned post-processing
# ---------------------------------------------------------------------------
class DA3DepthProcessor:
    """Produce InstinctLab-aligned depth from GS-rendered RGB via DA3.

    The GS RGB is already rendered from the correct pitched camera pose
    (matching the InstinctLab training camera with ~48° downward pitch).

    Pipeline (matching InstinctLab training noise_pipeline exactly):
      1. DA3 forward pass on RGB -> relative inverse-depth at input res
      2. Resize to (36, 64) -- InstinctLab raw camera resolution
      3. Crop: top=18, bottom=0, left=16, right=16 -> (18, 32)
      4. Gaussian blur (k=3, sigma=1)
      5. Per-image normalize to [0, 1]

    DA3 outputs inverse-depth (close = large value). We invert it to get
    depth-like ordering (close = small value) before normalizing, matching
    the training's distance_to_image_plane convention.
    """

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._autocast_dtype = None
        self._blur_kernel = None
        self.last_raw_depth = None  # (B, H_da3, W_da3) before any post-processing

    def _ensure_model(self):
        if self._model is not None:
            return
        from depth_anything_3.api import DepthAnything3
        print(f"[INFO] Loading DA3 model '{self.model_name}' ...")
        self._model = DepthAnything3.from_pretrained(self.model_name)
        self._model = self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        k, sigma = 3, 1.0
        ax = torch.arange(k, device=self.device, dtype=torch.float32) - k // 2
        g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        g1d = g1d / g1d.sum()
        self._blur_kernel = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)
        print("[INFO] DA3 loaded.")

    @torch.no_grad()
    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1] -- rendered from pitched camera
        Returns:
            depth_processed: (B, DEPTH_H, DEPTH_W) in [0, 1]
        """
        self._ensure_model()
        B = rgb.shape[0]

        results = []
        with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
            for i in range(B):
                frame = rgb[i:i+1].unsqueeze(1)  # (1, 1, 3, H, W) for DA3
                out = self._model.model(frame)
                results.append(out.depth.squeeze())  # (H_da3, W_da3)

        depth = torch.stack(results, 0).float()  # (B, H_da3, W_da3)
        self.last_raw_depth = depth.clone()

        # DA3 outputs inverse-depth (large = close). Invert to get depth ordering
        # (small = close) matching training's distance_to_image_plane.
        d_max = depth.flatten(1).max(dim=1).values.reshape(-1, 1, 1)
        depth = d_max - depth + 1e-6

        # Resize to InstinctLab raw camera resolution (36x64)
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(DEPTH_RAW_H, DEPTH_RAW_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, 36, 64)

        # Crop (matches training noise_pipeline: crop_region=(18, 0, 16, 16))
        h_end = DEPTH_RAW_H - CROP_BOT if CROP_BOT > 0 else DEPTH_RAW_H
        w_end = DEPTH_RAW_W - CROP_RIGHT if CROP_RIGHT > 0 else DEPTH_RAW_W
        depth = depth[:, CROP_TOP:h_end, CROP_LEFT:w_end]  # (B, 18, 32)

        # Gaussian blur (k=3, sigma=1) matching training
        depth = F.conv2d(
            depth.unsqueeze(1), self._blur_kernel, padding=1
        ).squeeze(1)

        # Per-image normalize to [0, 1] (DA3 relative depth has no metric scale)
        d_min = depth.flatten(1).min(dim=1).values.reshape(-1, 1, 1)
        d_range = depth.flatten(1).max(dim=1).values.reshape(-1, 1, 1) - d_min
        depth = (depth - d_min) / (d_range + 1e-6)

        return depth


# ---------------------------------------------------------------------------
# GS direct depth renderer (no DA3 -- uses gsplat expected-depth)
# ---------------------------------------------------------------------------
class GsDirectDepthProcessor:
    """Produce InstinctLab-aligned depth directly from GS expected-depth rendering.

    Uses gsplat render_mode="RGB+ED" to get metric expected depth, then applies
    the same resize -> crop -> blur -> normalize pipeline as DA3DepthProcessor
    so the policy receives identically shaped input.
    """

    def __init__(self, gs_renderer: "GsRgbRenderer"):
        self.gs_renderer = gs_renderer
        self.device = gs_renderer.device
        self.last_raw_depth = None

        k, sigma = 3, 1.0
        ax = torch.arange(k, device=self.device, dtype=torch.float32) - k // 2
        g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        g1d = g1d / g1d.sum()
        self._blur_kernel = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def render_rgbd(self, torso_pos_w, torso_quat_wxyz, cam_pos_w):
        """Render RGB (B,3,H,W) and depth (B,1,H,W) from GS scene."""
        B = torso_pos_w.shape[0]
        r = self.gs_renderer
        w, h = r.RENDER_W, r.RENDER_H

        if not r.has_gs:
            rgb = torch.zeros(B, 3, h, w, device=self.device)
            depth = torch.zeros(B, 1, h, w, device=self.device)
            return rgb, depth

        c2b = r._c2b.unsqueeze(0).expand(B, 4)
        gs_cam_quat = _quat_mul(torso_quat_wxyz, c2b)

        offsets = (
            r.terrain.sub_mesh_offsets[:B]
            if hasattr(r.terrain, "sub_mesh_offsets")
            else torch.zeros(B, 3, device=self.device)
        )
        gs_cam_pos = cam_pos_w - offsets
        Ks = r._K.unsqueeze(0).expand(B, 3, 3)

        rgb_hwc = torch.zeros(B, h, w, 3, device=self.device)
        depth_hw = torch.zeros(B, h, w, device=self.device)
        for i in range(B):
            try:
                rgb_d, depth_d = r.terrain.render_rgbd(
                    camera_pos_w=gs_cam_pos[i:i+1],
                    camera_quat_w=gs_cam_quat[i:i+1],
                    Ks=Ks[i:i+1],
                    width=w,
                    height=h,
                )
                rgb_hwc[i] = rgb_d[0]
                depth_hw[i] = depth_d[0, :, :, 0]
            except Exception as e:
                print(f"[WARN] GS render_rgbd failed env {i}: {e}")

        rgb_out = rgb_hwc.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        return rgb_out, depth_hw

    @torch.no_grad()
    def __call__(self, depth_raw: torch.Tensor) -> torch.Tensor:
        """Post-process raw GS depth (B, H_render, W_render) -> (B, DEPTH_H, DEPTH_W).

        Applies the same resize/crop/blur/normalize as DA3DepthProcessor.
        """
        self.last_raw_depth = depth_raw.clone()

        depth = F.interpolate(
            depth_raw.unsqueeze(1),
            size=(DEPTH_RAW_H, DEPTH_RAW_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        h_end = DEPTH_RAW_H - CROP_BOT if CROP_BOT > 0 else DEPTH_RAW_H
        w_end = DEPTH_RAW_W - CROP_RIGHT if CROP_RIGHT > 0 else DEPTH_RAW_W
        depth = depth[:, CROP_TOP:h_end, CROP_LEFT:w_end]

        depth = F.conv2d(
            depth.unsqueeze(1), self._blur_kernel, padding=1
        ).squeeze(1)

        d_min = depth.flatten(1).min(dim=1).values.reshape(-1, 1, 1)
        d_range = depth.flatten(1).max(dim=1).values.reshape(-1, 1, 1) - d_min
        depth = (depth - d_min) / (d_range + 1e-6)

        return depth


# ---------------------------------------------------------------------------
# Waypoint navigation: spawn on collision-free points, walk along A* path
# ---------------------------------------------------------------------------
class WaypointNavigator:
    """Manages spawn selection, A* path planning, and velocity command generation.

    Aligned with gs_scene.gs_commands_nav.NavigateToGoal:
    - spawn_pts are mesh-local; waypoints stored with sub_mesh_offset (world coords)
    - vel command: world-frame linvel then quat_rotate_inverse to body frame

    When ``stair_mode=True``, uses 3D-aware pathfinding and tracks per-env
    expected Z for fallen detection on stairs.
    """

    def __init__(
        self,
        spawn_pts: torch.Tensor,
        sub_mesh_offsets: torch.Tensor,
        spacing: float,
        num_envs: int,
        device,
        default_root_state: torch.Tensor,
        max_speed: float = 0.5,
        waypoint_advance_dist: float = 0.4,
        yaw_stiffness: float = 1.5,
        yaw_clamp: float = 1.0,
        max_waypoints: int = 256,
        smooth_alpha: float = 0.15,
        slowdown_dist: float = 0.6,
        stair_mode: bool = False,
    ):
        self.device = device
        self.num_envs = num_envs
        self.spawn_pts = spawn_pts
        self.sub_mesh_offsets = sub_mesh_offsets
        self.n_spawn = len(spawn_pts)
        self.max_speed = max_speed
        self.waypoint_advance_dist = waypoint_advance_dist
        self.yaw_stiffness = yaw_stiffness
        self.yaw_clamp = yaw_clamp
        self.max_waypoints = max_waypoints
        self.smooth_alpha = smooth_alpha
        self.slowdown_dist = slowdown_dist
        self.stair_mode = stair_mode
        self.default_root_state = default_root_state.clone()

        pts_np = spawn_pts.cpu().numpy()
        self.graph = _build_spawn_graph(pts_np, spacing, stair_mode=stair_mode)

        self.waypoints = torch.zeros(num_envs, max_waypoints, 3, device=device)
        self.num_wp = torch.ones(num_envs, dtype=torch.long, device=device)
        self.wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._cmd_linvel_w = torch.zeros(num_envs, 3, device=device)
        self._cmd_yawvel = torch.zeros(num_envs, 1, device=device)
        self.expected_floor_z = torch.zeros(num_envs, device=device)

    def reset_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        root_state = self.default_root_state[env_ids].clone()
        robot_standing_height = self.default_root_state[0, 2].item()

        for local_i, eid in enumerate(env_ids.tolist()):
            sub_off = self.sub_mesh_offsets[eid]

            if self.stair_mode:
                z_vals = self.spawn_pts[:, 2]
                z_order = torch.argsort(z_vals)
                n10 = max(1, self.n_spawn // 5)
                low_pool = z_order[:n10]
                high_pool = z_order[-n10:]
                start_idx = int(low_pool[torch.randint(0, len(low_pool), (1,))].item())
                goal_idx = int(high_pool[torch.randint(0, len(high_pool), (1,))].item())
                sz = float(self.spawn_pts[start_idx, 2])
                gz = float(self.spawn_pts[goal_idx, 2])
                print(f"[Stair] UP: start Z={sz:.2f}, goal Z={gz:.2f}, dZ={gz-sz:+.2f}")
            else:
                start_idx = int(torch.randint(0, self.n_spawn, (1,)).item())
                goal_idx = start_idx
                attempts = 0
                while goal_idx == start_idx and attempts < 50:
                    goal_idx = int(torch.randint(0, self.n_spawn, (1,)).item())
                    attempts += 1

            path = self.graph.plan(start_idx, goal_idx)
            if len(path) == 0:
                path = [start_idx]

            n_wp = min(len(path), self.max_waypoints)
            pts = self.spawn_pts[path[:n_wp]]
            self.waypoints[eid, :n_wp] = pts + sub_off.unsqueeze(0)
            self.waypoints[eid, n_wp:] = pts[-1] + sub_off
            self.num_wp[eid] = n_wp
            self.wp_idx[eid] = min(1, n_wp - 1)
            self.ep_steps[eid] = 0
            self._cmd_linvel_w[eid] = 0.0
            self._cmd_yawvel[eid] = 0.0

            sp = self.spawn_pts[start_idx]
            root_state[local_i, 0] = sp[0] + sub_off[0]
            root_state[local_i, 1] = sp[1] + sub_off[1]
            root_state[local_i, 2] = sp[2] + robot_standing_height + sub_off[2]

            self.expected_floor_z[eid] = sp[2] + sub_off[2]

            # Face toward the first waypoint
            if n_wp >= 2:
                first_wp = pts[1]
            else:
                first_wp = self.spawn_pts[goal_idx]
            dx = float(first_wp[0] - sp[0])
            dy = float(first_wp[1] - sp[1])
            yaw = math.atan2(dy, dx)
            # quat wxyz from yaw
            root_state[local_i, 3] = math.cos(yaw / 2)
            root_state[local_i, 4] = 0.0
            root_state[local_i, 5] = 0.0
            root_state[local_i, 6] = math.sin(yaw / 2)

        root_state[:, 7:] = 0.0
        return root_state

    def compute_vel_cmd(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute (vx_b, vy_b, wz) body-frame command. Matches NavigateToGoal.update."""
        self.ep_steps += 1

        cur_wp = self.waypoints[
            torch.arange(self.num_envs, device=self.device), self.wp_idx
        ]
        delta = cur_wp - root_pos_w
        delta[:, 2] = 0.0
        dist = delta[:, :2].norm(dim=-1)

        advance = dist < self.waypoint_advance_dist
        not_last = self.wp_idx < (self.num_wp - 1)
        self.wp_idx = torch.where(
            advance & not_last, self.wp_idx + 1, self.wp_idx
        )

        cur_wp = self.waypoints[
            torch.arange(self.num_envs, device=self.device), self.wp_idx
        ]
        delta = cur_wp - root_pos_w
        delta[:, 2] = 0.0
        dist = delta[:, :2].norm(dim=-1)

        if self.stair_mode:
            self.expected_floor_z = cur_wp[:, 2]

        desired_yaw = torch.atan2(delta[:, 1], delta[:, 0])
        robot_yaw = _yaw_from_quat(root_quat_w)
        yaw_err = _wrap_to_pi(desired_yaw - robot_yaw)

        target_yawvel = (self.yaw_stiffness * yaw_err).clamp(
            -self.yaw_clamp, self.yaw_clamp
        ).unsqueeze(1)
        a = self.smooth_alpha
        self._cmd_yawvel = self._cmd_yawvel.lerp(target_yawvel, a)

        speed = dist.clamp(max=self.max_speed)
        speed = torch.where(
            dist < self.slowdown_dist,
            speed * (dist / self.slowdown_dist).clamp(max=1.0),
            speed,
        )

        target_w = torch.zeros(self.num_envs, 3, device=self.device)
        target_w[:, 0] = desired_yaw.cos() * speed
        target_w[:, 1] = desired_yaw.sin() * speed
        self._cmd_linvel_w = self._cmd_linvel_w.lerp(target_w, a)

        yaw_q = torch.stack([
            torch.cos(robot_yaw / 2),
            torch.zeros_like(robot_yaw),
            torch.zeros_like(robot_yaw),
            torch.sin(robot_yaw / 2),
        ], dim=-1)
        linvel_b = _quat_rotate(_quat_inv(yaw_q), self._cmd_linvel_w)

        return torch.cat([linvel_b[:, :2], self._cmd_yawvel], dim=-1)


def _build_spawn_graph(pts_np: np.ndarray, spacing: float, stair_mode: bool = False):
    """Build a SpawnGraph; import from gs_scene if available, else inline.

    When stair_mode is True, uses 3D-aware StairSpawnGraph that connects
    neighbours only when their Z difference is within stair step height.
    """
    if stair_mode:
        try:
            from gs_scene.utils.pathfinding import StairSpawnGraph
            return StairSpawnGraph(pts_np, spacing=spacing)
        except ImportError:
            pass

    try:
        from gs_scene.utils.pathfinding import SpawnGraph
        return SpawnGraph(pts_np, spacing=spacing)
    except ImportError:
        pass

    import heapq
    connect_radius = spacing * math.sqrt(2) * 1.5
    tree = cKDTree(pts_np[:, :2])
    n = len(pts_np)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    pairs = tree.query_pairs(r=connect_radius, output_type="ndarray")
    for i, j in pairs:
        if stair_mode and abs(float(pts_np[i, 2] - pts_np[j, 2])) > 0.5:
            continue
        d = float(np.linalg.norm(pts_np[i] - pts_np[j]) if stair_mode
                  else np.linalg.norm(pts_np[i, :2] - pts_np[j, :2]))
        adj[i].append((j, d))
        adj[j].append((i, d))

    class _Graph:
        def __init__(self):
            self.points = pts_np
            self.n = n
            self._adj = adj

        def plan(self, s, g):
            if s == g:
                return [s]
            goal = self.points[g, :3] if stair_mode else self.points[g, :2]
            open_set = [(0.0, s)]
            came_from = {}
            g_score = np.full(self.n, np.inf)
            g_score[s] = 0.0
            closed = np.zeros(self.n, dtype=bool)
            while open_set:
                _, cur = heapq.heappop(open_set)
                if cur == g:
                    path = [cur]
                    while cur in came_from:
                        cur = came_from[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                if closed[cur]:
                    continue
                closed[cur] = True
                for nb, ec in self._adj[cur]:
                    if closed[nb]:
                        continue
                    t = g_score[cur] + ec
                    if t < g_score[nb]:
                        g_score[nb] = t
                        came_from[nb] = cur
                        p = self.points[nb, :3] if stair_mode else self.points[nb, :2]
                        h = float(np.linalg.norm(p - goal))
                        heapq.heappush(open_set, (t + h, nb))
            return []

    return _Graph()


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion (w, x, y, z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Isaac Sim viewport third-person capture
# ---------------------------------------------------------------------------
class ThirdPersonCapture:
    """Capture Isaac Sim viewport as overhead follow camera.

    Camera is directly above the robot looking straight down.
    """

    def __init__(self, env, height: float = 3.0, smooth: float = 0.9):
        self.env = env
        self.height = height
        self.smooth = smooth
        self._eye_smooth: np.ndarray | None = None

    def capture(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor,
                env_id: int = 0) -> np.ndarray | None:
        pos = root_pos_w[env_id].cpu().numpy()

        eye = np.array([pos[0], pos[1], pos[2] + self.height])

        if self._eye_smooth is None:
            self._eye_smooth = eye.copy()
        else:
            a = self.smooth
            self._eye_smooth = a * self._eye_smooth + (1.0 - a) * eye

        # Look at the robot with a tiny Y offset so the camera up-vector is well defined
        target = [pos[0], pos[1], pos[2]]
        try:
            self.env.sim.set_camera_view(self._eye_smooth.tolist(), target)
            self.env.sim.render()
            return self.env.render("rgb_array")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from omegaconf import OmegaConf
    import active_adaptation as aa

    # Resolve task config path (relative to gs-scene/ or active-adaptation/)
    cfg_path = Path(args.task)
    if not cfg_path.is_absolute():
        for base in [
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve().parents[2] / "gs-scene",
        ]:
            candidate = base / cfg_path
            if candidate.exists():
                cfg_path = candidate
                break

    if not cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {cfg_path}")

    task_cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(task_cfg)
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.num_envs = args.num_envs

    if not task_cfg.scene.get("load_gaussians", False):
        print("[INFO] Forcing load_gaussians=true for GS rendering")
        task_cfg.scene.load_gaussians = True

    full_cfg = OmegaConf.create({
        "backend": "isaac",
        "device": args.device,
        "headless": args.headless,
        "app": {
            "headless": args.headless,
            "enable_cameras": True,
        },
        "task": task_cfg,
    })

    aa.init(full_cfg, auto_rank=True)

    from gs_scene.gs_env import GSEnvIsaac

    env = GSEnvIsaac(full_cfg.task, str(args.device), headless=args.headless)
    robot = env.scene.articulations["robot"]
    num_envs = env.num_envs
    device = env.device

    # Joint setup
    joint_names = robot.data.joint_names
    default_joint_pos = robot.data.default_joint_pos[0, :NUM_JOINTS].clone()
    action_scale = torch.zeros(NUM_JOINTS, device=device)
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        action_scale[i] = INSTINCTLAB_ACTION_SCALES.get(jn, 0.5)

    # Torso body index for camera attachment
    body_idx = robot.find_bodies("torso_link")[0][0]

    # Components
    policy = OnnxPolicy(args.ckpt_dir, device=str(device))
    obs_builder = ObservationBuilder(num_envs, device)
    obs_builder.set_default_joint_pos(default_joint_pos)
    gs_renderer = GsRgbRenderer(env, str(device))
    use_gs_depth = args.depth_mode == "gs_render"
    if use_gs_depth:
        gs_depth_proc = GsDirectDepthProcessor(gs_renderer)
        da3_depth = None
        print("[INFO] Depth mode: gs_render (direct GS expected-depth)")
    else:
        gs_depth_proc = None
        da3_depth = DA3DepthProcessor(args.da3_model, str(device))
        print(f"[INFO] Depth mode: da3 (model={args.da3_model})")
    tp_capture = ThirdPersonCapture(env)

    env.reset()

    # ---- Waypoint navigator (uses valid_spawn_points from terrain) ----
    terrain = env.scene.terrain
    spawn_pts = getattr(terrain, "valid_spawn_points", None)
    sub_mesh_offsets = (
        terrain.sub_mesh_offsets[:num_envs]
        if hasattr(terrain, "sub_mesh_offsets")
        else torch.zeros(num_envs, 3, device=device)
    )

    scene_cfg = getattr(env, "cfg", {})
    if hasattr(scene_cfg, "get"):
        spawn_spacing = scene_cfg.get("scene", {}).get("spawn_spacing", 0.3)
        stair_mode = scene_cfg.get("scene", {}).get("spawn_mode", "flat") == "stair"
    else:
        spawn_spacing = 0.3
        stair_mode = False

    navigator = None
    if spawn_pts is not None and len(spawn_pts) > 1:
        navigator = WaypointNavigator(
            spawn_pts=spawn_pts,
            sub_mesh_offsets=sub_mesh_offsets,
            spacing=spawn_spacing,
            num_envs=num_envs,
            device=device,
            default_root_state=robot.data.default_root_state,
            max_speed=args.max_speed,
            stair_mode=stair_mode,
        )
        print(f"[INFO] Navigator: {len(spawn_pts)} spawn points, "
              f"spacing={spawn_spacing}, stair_mode={stair_mode}")
    else:
        print("[INFO] No spawn points found; using constant vel command as fallback")

    # Initial reset: teleport all envs to random spawn points
    all_ids = torch.arange(num_envs, device=device)
    if navigator is not None:
        init_root_state = navigator.reset_envs(all_ids)
        robot.write_root_state_to_sim(init_root_state, all_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[all_ids],
            torch.zeros_like(robot.data.default_joint_vel[all_ids]),
            env_ids=all_ids,
        )
        for _ in range(env.decimation):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    init_z = robot.data.root_pos_w[:, 2].mean().item()
    robot_standing_height = robot.data.default_root_state[0, 2].item()
    if stair_mode:
        fallen_drop = 0.5
        print(f"[INFO] Stair mode: init_z={init_z:.3f}, "
              f"standing_height={robot_standing_height:.3f}, "
              f"fallen if Z < floor_z + standing_height - {fallen_drop}")
    else:
        fallen_threshold = init_z - 0.5
        print(f"[INFO] init z={init_z:.3f}, fallen threshold={fallen_threshold:.3f}")

    obs_builder.reset(all_ids)

    # Per-segment composite video buffers (env 0)
    seg_tp_frames: list[np.ndarray] = []    # Isaac Sim viewport
    seg_rgb_frames: list[np.ndarray] = []   # first-person RGB
    seg_depth_frames: list[np.ndarray] = [] # DA3 raw depth
    seg_id = 0

    # Debug draw for path visualization
    debug_draw = None
    if not args.headless:
        try:
            from active_adaptation.utils.debug import DebugDraw
            debug_draw = DebugDraw()
            print("[INFO] Debug draw enabled for path visualization")
        except Exception as e:
            print(f"[WARN] Debug draw unavailable: {e}")

    def _draw_paths():
        """Visualize current waypoint paths for all envs."""
        if debug_draw is None or navigator is None:
            return
        debug_draw.clear()
        for eid in range(num_envs):
            n_wp = navigator.num_wp[eid].item()
            wp_i = navigator.wp_idx[eid].item()
            if n_wp < 2:
                continue
            wp = navigator.waypoints[eid, :n_wp].clone()
            wp[:, 2] += 0.15  # lift slightly above ground
            debug_draw.plot(wp, size=2.0, color=(0.2, 0.8, 0.2, 1.0))
            debug_draw.point(
                wp[wp_i:wp_i+1], color=(1.0, 0.2, 0.0, 1.0), size=15.0
            )

    print(f"[INFO] ckpt: {args.ckpt_dir}")
    print(f"[INFO] num_envs={num_envs}, device={device}")
    print(f"[INFO] episode_len={args.episode_len}, max_speed={args.max_speed}")
    print(f"[INFO] GS available: {gs_renderer.has_gs}")
    print(f"[INFO] Joint order ({NUM_JOINTS}):")
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        print(f"  {i:2d}: {jn:35s} default={default_joint_pos[i]:+.3f} scale={action_scale[i]:.4f}")

    _draw_paths()

    depth_label = "GS Depth (raw)" if use_gs_depth else "DA3 Depth (raw)"

    def _save_composite_video(tp_list, rgb_list, depth_list, out_dir, seg_idx):
        """Save a composite video: top = third-person GS, bottom = [RGB | depth]."""
        import cv2
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"seg_{seg_idx:04d}.mp4")

        n = min(len(tp_list), len(rgb_list), len(depth_list))
        if n == 0:
            return

        TOP_W, TOP_H = 960, 540
        BOT_W, BOT_H = TOP_W // 2, TOP_H // 2
        TOTAL_H = TOP_H + BOT_H

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (TOP_W, TOTAL_H),
        )

        for i in range(n):
            canvas = np.zeros((TOTAL_H, TOP_W, 3), dtype=np.uint8)

            tp = cv2.resize(tp_list[i], (TOP_W, TOP_H))
            canvas[:TOP_H, :] = cv2.cvtColor(tp, cv2.COLOR_RGB2BGR)

            r = cv2.resize(rgb_list[i], (BOT_W, BOT_H))
            canvas[TOP_H:, :BOT_W] = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)

            d = cv2.resize(depth_list[i], (BOT_W, BOT_H))
            canvas[TOP_H:, BOT_W:] = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)

            cv2.putText(canvas, "Isaac Sim Viewport", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "RGB (pitched cam)", (10, TOP_H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, depth_label, (BOT_W + 10, TOP_H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            writer.write(canvas)

        writer.release()
        print(f"[INFO] Composite video seg {seg_idx}: {out_path} ({n} frames)")

    # Fallback constant command when no navigator
    const_vel_cmd = torch.tensor(
        [[args.vx, args.vy, args.wz]], device=device
    ).expand(num_envs, 3)

    def _do_reset(reset_ids: torch.Tensor):
        """Reset envs to random spawn points with default joint state."""
        if navigator is not None:
            rst_state = navigator.reset_envs(reset_ids)
        else:
            rst_state = robot.data.default_root_state[reset_ids].clone()
            rst_state[:, :3] = robot.data.root_pos_w[reset_ids]
            rst_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(rst_state, reset_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[reset_ids],
            torch.zeros_like(robot.data.default_joint_vel[reset_ids]),
            env_ids=reset_ids,
        )
        obs_builder.reset(reset_ids)
        _draw_paths()

    for step in range(args.max_steps):
        with torch.inference_mode():
            # ---- read robot state ----
            root_quat_w = robot.data.root_quat_w
            root_pos_w = robot.data.root_pos_w
            joint_pos = robot.data.joint_pos[:, :NUM_JOINTS]
            joint_vel = robot.data.joint_vel[:, :NUM_JOINTS]
            torso_pos_w = robot.data.body_pos_w[:, body_idx]
            torso_quat_w = robot.data.body_quat_w[:, body_idx]

            base_ang_vel_b = _quat_rotate(
                _quat_inv(root_quat_w), robot.data.root_com_ang_vel_w
            )
            proj_grav = _projected_gravity(root_quat_w)

            # ---- velocity command from navigator or fallback ----
            if navigator is not None:
                vel_cmd = navigator.compute_vel_cmd(root_pos_w, root_quat_w)
            else:
                vel_cmd = const_vel_cmd

            # ---- GS render -> depth ----
            cam_pos_w, _ = _build_camera_world_pose(
                torso_pos_w, torso_quat_w, device
            )
            if use_gs_depth:
                rgb, depth_raw = gs_depth_proc.render_rgbd(
                    torso_pos_w, torso_quat_w, cam_pos_w
                )
                depth_frame = gs_depth_proc(depth_raw)
            else:
                rgb = gs_renderer.render(torso_pos_w, torso_quat_w, cam_pos_w)
                depth_frame = da3_depth(rgb)
            obs_builder.push_depth(depth_frame)

            # ---- proprioception ----
            obs_builder.push_proprio(
                base_ang_vel=base_ang_vel_b,
                projected_gravity=proj_grav,
                velocity_commands=vel_cmd,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                last_action=obs_builder.prev_action,
            )

            # ---- policy inference ----
            obs_flat = obs_builder.build_obs()
            raw_actions = policy(obs_flat)
            obs_builder.prev_action = raw_actions.clone()

            # ---- apply actions manually (bypass action manager) ----
            target_pos = raw_actions * action_scale + default_joint_pos.unsqueeze(0)
            full_target = robot.data.default_joint_pos.clone()
            full_target[:, :NUM_JOINTS] = target_pos
            robot.set_joint_position_target(full_target)

            for _ in range(env.decimation):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)

            # ---- collect frames for composite video (env 0) ----
            tp_frame = tp_capture.capture(root_pos_w, root_quat_w, env_id=0)
            if tp_frame is not None:
                seg_tp_frames.append(tp_frame)

            # first-person RGB
            rgb_np = (rgb[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            seg_rgb_frames.append(rgb_np)

            # Raw depth visualization
            _raw_d = (gs_depth_proc.last_raw_depth if use_gs_depth
                      else (da3_depth.last_raw_depth if da3_depth else None))
            if _raw_d is not None:
                rd = _raw_d[0]
                rd_min, rd_max = rd.min(), rd.max()
                rd_norm = ((rd - rd_min) / (rd_max - rd_min + 1e-6) * 255).byte()
                rd_rgb = rd_norm.unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
                seg_depth_frames.append(rd_rgb)

            if len(seg_tp_frames) >= args.diag_interval:
                _save_composite_video(
                    seg_tp_frames, seg_rgb_frames, seg_depth_frames,
                    args.diag_video_dir, seg_id,
                )
                seg_id += 1
                seg_tp_frames.clear()
                seg_rgb_frames.clear()
                seg_depth_frames.clear()

            # ---- reset: fallen OR episode timeout (max episode_len steps) ----
            if stair_mode and navigator is not None:
                expected_z = navigator.expected_floor_z + robot_standing_height
                fallen = root_pos_w[:, 2] < (expected_z - fallen_drop)
            else:
                fallen = root_pos_w[:, 2] < fallen_threshold
            if navigator is not None:
                timeout = navigator.ep_steps >= args.episode_len
            else:
                timeout = torch.zeros(num_envs, dtype=torch.bool, device=device)
            reset_mask = fallen | timeout
            reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
            if reset_ids.numel() > 0:
                _do_reset(reset_ids)

            # ---- logging ----
            if step % 100 == 0:
                rv = robot.data.root_com_lin_vel_w[0]
                d = depth_frame[0]
                ep_step = navigator.ep_steps[0].item() if navigator else "N/A"
                wp_i = navigator.wp_idx[0].item() if navigator else "N/A"
                wp_n = navigator.num_wp[0].item() if navigator else "N/A"
                print(
                    f"Step {step:5d} ep_step={ep_step} wp={wp_i}/{wp_n} | "
                    f"pos=({root_pos_w[0,0]:.2f}, {root_pos_w[0,1]:.2f}, {root_pos_w[0,2]:.2f}) | "
                    f"vel=({rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f}) | "
                    f"cmd=({vel_cmd[0,0]:.2f}, {vel_cmd[0,1]:.2f}, {vel_cmd[0,2]:.2f}) | "
                    f"depth=[{d.min():.2f}, {d.mean():.2f}, {d.max():.2f}]"
                )

    # Save remaining frames
    if seg_tp_frames:
        _save_composite_video(
            seg_tp_frames, seg_rgb_frames, seg_depth_frames,
            args.diag_video_dir, seg_id,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
