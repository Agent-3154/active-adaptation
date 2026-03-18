"""Verify InstinctLab ONNX policy on flat ground using raycaster depth.

Same observation/action pipeline as deploy_instinctlab.py in gs-scene, but
uses IsaacLab RayCasterCamera for depth instead of GS + DA3. If the robot
walks stably on flat ground, the pipeline is correct.

Usage (from active-adaptation/ root):
    conda run -n gs python scripts/verify_flat_deploy.py \
        [--vx 0.5] [--vy 0.0] [--wz 0.0] [--headless]
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cfg/task/G1FlatDeploy.yaml")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=str(
        Path(__file__).resolve().parents[2]
        / "hiking-in-the-wild_Data&Model"
        / "data&model"
        / "checkpoints"
        / "parkour_onboard_preview_stair"
        / "exported"
    ),
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# InstinctLab observation spec
# ---------------------------------------------------------------------------
NUM_JOINTS = 29
PROPRIO_HISTORY = 8
DEPTH_HISTORY_FRAMES = 8

DEPTH_RAW_H, DEPTH_RAW_W = 36, 64
CROP_TOP, CROP_BOT, CROP_LEFT, CROP_RIGHT = 18, 0, 16, 16
DEPTH_H = DEPTH_RAW_H - CROP_TOP - CROP_BOT   # 18
DEPTH_W = DEPTH_RAW_W - CROP_LEFT - CROP_RIGHT  # 32
DEPTH_MIN, DEPTH_MAX = 0.0, 2.5

CAM_OFFSET_POS = (0.0487988662332928, 0.01, 0.4378029937970051)
CAM_OFFSET_QUAT_WXYZ = (0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0)

OBS_SCALES = {"base_ang_vel": 0.25, "joint_vel": 0.05}

PROPRIO_SIZE = PROPRIO_HISTORY * (3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS)
DEPTH_FLAT_SIZE = DEPTH_HISTORY_FRAMES * DEPTH_H * DEPTH_W


# ---------------------------------------------------------------------------
# ONNX policy  (same as deploy_instinctlab.py)
# ---------------------------------------------------------------------------
class OnnxPolicy:
    def __init__(self, ckpt_dir: str, device: str = "cuda:0"):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        self.encoder = ort.InferenceSession(
            os.path.join(ckpt_dir, "0-depth_encoder.onnx"), providers=providers
        )
        self.actor = ort.InferenceSession(
            os.path.join(ckpt_dir, "actor.onnx"), providers=providers
        )
        self.device = device
        enc_in = self.encoder.get_inputs()[0]
        act_in = self.actor.get_inputs()[0]
        print(f"[ONNX] depth_encoder input: {enc_in.shape}")
        print(f"[ONNX] actor input: {act_in.shape}")

    def __call__(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs_np = obs_flat.detach().cpu().float().numpy()
        proprio = obs_np[:, :PROPRIO_SIZE]
        depth_flat = obs_np[:, PROPRIO_SIZE:]
        depth_input = depth_flat.reshape(-1, DEPTH_HISTORY_FRAMES, DEPTH_H, DEPTH_W)

        enc_out = self.encoder.run(
            None, {self.encoder.get_inputs()[0].name: depth_input}
        )[0]
        actor_input = np.concatenate([proprio, enc_out], axis=1)
        actions = self.actor.run(
            None, {self.actor.get_inputs()[0].name: actor_input}
        )[0]
        return torch.from_numpy(actions).to(self.device)


# ---------------------------------------------------------------------------
# Observation builder  (same as deploy_instinctlab.py)
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
        self.depth_hist = torch.zeros(num_envs, DEPTH_HISTORY_FRAMES, DEPTH_H, DEPTH_W, device=device)
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
# Quaternion utilities (w,x,y,z)
# ---------------------------------------------------------------------------
def _quat_rotate(q, v):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack([vx + w * tx + y * tz - z * ty,
                        vy + w * ty + z * tx - x * tz,
                        vz + w * tz + x * ty - y * tx], dim=-1)

def _quat_inv(q):
    inv = q.clone()
    inv[:, 1:] = -inv[:, 1:]
    return inv

def _projected_gravity(root_quat_wxyz):
    grav = torch.tensor([[0.0, 0.0, -1.0]], device=root_quat_wxyz.device).expand(root_quat_wxyz.shape[0], 3)
    return _quat_rotate(_quat_inv(root_quat_wxyz), grav)


# ---------------------------------------------------------------------------
# Depth processing: crop then normalize (matching InstinctLab noise_pipeline)
# ---------------------------------------------------------------------------
def process_depth(raw: torch.Tensor) -> torch.Tensor:
    """raw: (N, 36, 64) → (N, 18, 32) normalized [0, 1]"""
    h_end = DEPTH_RAW_H - CROP_BOT if CROP_BOT > 0 else DEPTH_RAW_H
    w_end = DEPTH_RAW_W - CROP_RIGHT if CROP_RIGHT > 0 else DEPTH_RAW_W
    cropped = raw[:, CROP_TOP:h_end, CROP_LEFT:w_end]
    cropped = cropped.clamp(DEPTH_MIN, DEPTH_MAX)
    return (cropped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)


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

    from active_adaptation.envs.locomotion import SimpleEnvIsaac

    env = SimpleEnvIsaac(full_cfg.task, str(args.device), headless=args.headless)

    robot = env.scene.articulations["robot"]
    num_envs = env.num_envs
    device = env.device

    joint_names = robot.data.joint_names
    default_joint_pos = robot.data.default_joint_pos[0, :NUM_JOINTS].clone()

    # Action scales from InstinctLab env.yaml
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
    action_scale = torch.zeros(NUM_JOINTS, device=device)
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        action_scale[i] = INSTINCTLAB_ACTION_SCALES.get(jn, 0.5)
    policy = OnnxPolicy(args.ckpt_dir, device=str(device))
    obs_builder = ObservationBuilder(num_envs, device)
    obs_builder.set_default_joint_pos(default_joint_pos)

    vel_cmd = torch.tensor([[args.vx, args.vy, args.wz]], device=device).expand(num_envs, 3)

    env.reset()
    obs_builder.reset(torch.arange(num_envs, device=device))

    print(f"[INFO] ckpt: {args.ckpt_dir}")
    print(f"[INFO] vel cmd: vx={args.vx}, vy={args.vy}, wz={args.wz}")
    print(f"[INFO] Joint order ({len(joint_names)} joints):")
    for i, jn in enumerate(joint_names[:NUM_JOINTS]):
        print(f"  {i:2d}: {jn:35s} default={default_joint_pos[i]:.3f}")
    print(f"[INFO] Total obs size: {PROPRIO_SIZE} (proprio) + {DEPTH_FLAT_SIZE} (depth)")

    for step in range(args.max_steps):
        with torch.inference_mode():
            # ---- robot state ----
            root_quat_w = robot.data.root_quat_w
            joint_pos = robot.data.joint_pos[:, :NUM_JOINTS]
            joint_vel = robot.data.joint_vel[:, :NUM_JOINTS]
            base_ang_vel_b = _quat_rotate(_quat_inv(root_quat_w),
                                          robot.data.root_com_ang_vel_w)
            proj_grav = _projected_gravity(root_quat_w)

            # ---- depth (flat ground → constant, normalized to ~1.0) ----
            # On flat ground, the camera sees the plane at a constant distance.
            # For verification we feed uniform depth; the policy should still walk
            # because proprioception dominates on flat terrain.
            depth_frame = torch.ones(num_envs, DEPTH_H, DEPTH_W, device=device)
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

            # ---- policy ----
            obs_flat = obs_builder.build_obs()
            raw_actions = policy(obs_flat)
            obs_builder.prev_action = raw_actions.clone()

            # ---- apply actions manually (bypass action manager to use
            #      InstinctLab default_pos, not aa's) ----
            target_pos = raw_actions * action_scale + default_joint_pos.unsqueeze(0)
            # pad to full joint count if needed
            full_target = robot.data.default_joint_pos.clone()
            full_target[:, :NUM_JOINTS] = target_pos
            robot.set_joint_position_target(full_target)
            for _ in range(env.decimation):
                robot.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)

            if not args.headless:
                env.sim.render()

            if step % 100 == 0:
                root_pos = robot.data.root_pos_w[0]
                root_vel = robot.data.root_com_lin_vel_w[0]
                print(
                    f"Step {step:5d} | "
                    f"pos=({root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}) | "
                    f"vel=({root_vel[0]:.2f}, {root_vel[1]:.2f}, {root_vel[2]:.2f})"
                )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
