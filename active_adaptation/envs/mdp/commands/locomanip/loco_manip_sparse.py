from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import torch
import warp as wp
from typing_extensions import override
from tensordict import TensorDict, TensorDictBase

from active_adaptation.utils.math import (
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    sample_quat_yaw,
    yaw_quat,
    clamp_norm,
    quat_conjugate,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.envs.mdp.commands.locomanip.loco_manip_kernels import (
    quat_wxyz_to_xyzw,
    sample_world_goal,
    update_world_command,
)
from ..base import CommandV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


# Per-env sparse command modes (distinct from LocoManipNew's 0/1/2).
MODE_GOAL_REACHING = 0  # sparse target: persistent world-frame EEF goal
MODE_TRAJECTORY = 1  # dense targets: parametric curve / hindsight path

# Curve kinds for online trajectory following.
CURVE_CIRCLE = 0
CURVE_LINE = 1


@wp.kernel(enable_backward=False)
def sample_sparse_world_goal(
    resample: wp.array(dtype=wp.bool),
    seed: wp.int32,
    eef_z_min: wp.float32,
    eef_z_max: wp.float32,
    world_radius_min: wp.float32,
    world_radius_max: wp.float32,
    standoff_reach_min: wp.float32,
    standoff_reach_max: wp.float32,
    linvel_gain_min: wp.float32,
    linvel_gain_max: wp.float32,
    yaw_gain_min: wp.float32,
    yaw_gain_max: wp.float32,
    root_pos_w: wp.array(dtype=wp.vec3),
    root_yaw_quat: wp.array(dtype=wp.quat),
    cmd_eef_pos_w: wp.array(dtype=wp.vec3),
    cmd_eef_status: wp.array(dtype=wp.int32),
    standoff_pos_w: wp.array(dtype=wp.vec3),
    standoff_yaw_w: wp.array(dtype=wp.float32),
    world_linvel_gain: wp.array(dtype=wp.float32),
    world_yaw_gain: wp.array(dtype=wp.float32),
):
    """Sample world goals for goal-reaching envs (same math as LocoManipNew mode 0)."""
    tid = wp.tid()
    if not resample[tid]:
        return

    seed_ = wp.rand_init(seed, tid)
    (
        seed_,
        cmd_eef_w,
        standoff_xy,
        standoff_yaw,
        linvel_gain,
        yaw_gain,
    ) = sample_world_goal(
        seed_,
        root_pos_w[tid],
        root_yaw_quat[tid],
        eef_z_min,
        eef_z_max,
        world_radius_min,
        world_radius_max,
        standoff_reach_min,
        standoff_reach_max,
        linvel_gain_min,
        linvel_gain_max,
        yaw_gain_min,
        yaw_gain_max,
    )
    cmd_eef_pos_w[tid] = cmd_eef_w
    standoff_pos_w[tid] = standoff_xy
    standoff_yaw_w[tid] = standoff_yaw
    world_linvel_gain[tid] = linvel_gain
    world_yaw_gain[tid] = yaw_gain
    if wp.randf(seed_, 0.0, 1.0) < 0.5:
        cmd_eef_status[tid] = wp.int32(0)
    else:
        cmd_eef_status[tid] = wp.int32(1)


@wp.kernel(enable_backward=False)
def update_sparse_world_command(
    sparse_mode: wp.array(dtype=wp.int32),
    root_pos_w: wp.array(dtype=wp.vec3),
    root_yaw_quat: wp.array(dtype=wp.quat),
    heading_w: wp.array(dtype=wp.float32),
    cmd_eef_pos_w: wp.array(dtype=wp.vec3),
    standoff_pos_w: wp.array(dtype=wp.vec3),
    standoff_yaw_w: wp.array(dtype=wp.float32),
    world_linvel_gain: wp.array(dtype=wp.float32),
    world_yaw_gain: wp.array(dtype=wp.float32),
    linvel_x_min: wp.float32,
    linvel_x_max: wp.float32,
    linvel_y_min: wp.float32,
    linvel_y_max: wp.float32,
    yaw_rate_min: wp.float32,
    yaw_rate_max: wp.float32,
    cmd_eef_pos_b: wp.array(dtype=wp.vec3),
    base_pos_error: wp.array(dtype=wp.float32),
):
    """Refresh heading-frame EEF + base_pos_error for goal-reaching envs.

    Calls ``update_world_command`` (same as LocoManipNew mode 0). The helper also
    returns loco linvel/yaw; those are discarded here — sparse policy is EEF-only.
    ``base_pos_error`` (standoff distance) is kept for reward gates only.
    """
    tid = wp.tid()
    if sparse_mode[tid] != wp.int32(0):
        return

    cmd_eef_pos_b_tid, _cmd_linvel_b, _cmd_yawvel_b, base_err = update_world_command(
        root_pos_w[tid],
        root_yaw_quat[tid],
        heading_w[tid],
        cmd_eef_pos_w[tid],
        standoff_pos_w[tid],
        standoff_yaw_w[tid],
        world_linvel_gain[tid],
        world_yaw_gain[tid],
        linvel_x_min,
        linvel_x_max,
        linvel_y_min,
        linvel_y_max,
        yaw_rate_min,
        yaw_rate_max,
    )
    cmd_eef_pos_b[tid] = cmd_eef_pos_b_tid
    base_pos_error[tid] = base_err


class LocoManipSparseBase(CommandV2):
    """Shared EEF-only command surface for random and replay sparse variants."""

    # Subclasses must set these before calling ``_resolve_eef_and_allocate_buffers``.
    eef_body_name: str
    gripper_joint_names: str

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        """Resolve EEF/gripper handles and allocate shared online tracking buffers."""
        super()._initialize(env)
        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, got {body_ids.numel()}"
            )
        self.eef_body_idx = body_ids[0]
        self.gripper_joint_ids, _ = self.asset.find_joints(self.gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(self.gripper_joint_ids, device=self.device)
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        with torch.device(self.device):
            self.sparse_mode = torch.zeros(self.num_envs, dtype=torch.int32)

            # --- Policy / reward EEF command & tracking ---
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)

            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)

            self.forward_diff_w = torch.zeros(self.num_envs, 3)
            self.forward_diff_b = torch.zeros(self.num_envs, 3)
            self.upward_diff_w = torch.zeros(self.num_envs, 3)
            self.upward_diff_b = torch.zeros(self.num_envs, 3)

            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)
            self.cmd_eef_rot_b = torch.zeros(self.num_envs, 4)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_b = torch.zeros(self.num_envs, 3)

            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.int32)

            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached_time = torch.zeros(self.num_envs, 1, dtype=torch.float)

            # --- Unused by sparse policy (no base / payload command) ---
            # Kept so rewards / Warp world-update match LocoManipNew mode 0.
            self.has_payload = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.payload_force_w = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            # Distance to standoff (New loco gate); not a commanded base velocity.
            self.base_pos_error = torch.zeros(self.num_envs, 1)
            # Standoff pose + gains feed update_world_command; linvel/yaw outputs discarded.
            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs)
            self.world_linvel_gain = torch.ones(self.num_envs)
            self.world_yaw_gain = torch.ones(self.num_envs)
        
        self.marker = None
        self.eef_pose_marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.eef_pose_marker = self.scene.create_frame_marker(
                "/Visuals/Command/target_eef_pose",
                scale=(0.1, 0.1, 0.1),
            )

    @property
    def eef_pos_w(self) -> torch.Tensor:
        return self.asset.data.body_link_pos_w[:, self.eef_body_idx]

    @property
    def eef_quat_w(self) -> torch.Tensor:
        return self.asset.data.body_link_quat_w[:, self.eef_body_idx]

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.cmd_eef_pos_b,  # [N, 3]
                self.pos_diff_b,  # [N, 3]
                self.cmd_eef_forward_b,  # [N, 3]
                self.forward_diff_b,  # [N, 3]
                self.cmd_eef_upward_b,  # [N, 3]
                self.upward_diff_b,  # [N, 3]
                self.cmd_eef_status.float(),  # [N, 1]
                (1 - self.cmd_eef_status.float()),  # [N, 1]
            ],
            dim=-1,
        )

    @override
    def symmetry_transform(self):
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        forward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_upward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        upward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        eef_status = SymmetryTransform(perm=[0, 1], signs=[1, 1])
        return SymmetryTransform.cat(
            [
                cmd_eef_pos_b,
                pos_diff_b,
                cmd_eef_forward_b,
                forward_diff_b,
                cmd_eef_upward_b,
                upward_diff_b,
                eef_status,
            ]
        )

    @override
    def debug_draw(self) -> None:
        self.env.scene.draw_vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        if self.eef_pose_marker is not None:
            self.eef_pose_marker.visualize(
                translations=self.cmd_eef_pos_w,
                orientations=self.cmd_eef_rot_w,
            )

    def get_gripper_status(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed)."""
        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        return 1.0 - openness

    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        """Relabel ``LocoManipNew`` rollouts into sparse EEF-only commands.

        * Teacher ``mode == 0`` (world) → goal reaching: target = world EEF goal.
        * Teacher ``mode in {1, 2}`` → trajectory following: target at step ``t``
          is the achieved EEF pose at ``t+1`` (hindsight), with done-aware shift.
        """
        device = tensordict.device
        # TensorClass does not support string key indexing; convert for relabel I/O.
        cs = tensordict["command_state"]
        if hasattr(cs, "to_tensordict"):
            tensordict["command_state"] = cs.to_tensordict()
        command_state = tensordict["command_state"]
        done = tensordict["next", "done"]

        mode = command_state["mode"]
        if mode.ndim == command_state["eef_pos_w"].ndim:
            mode = mode.squeeze(-1)
        is_goal = mode == 0  # [T, N]

        root_pose_w = command_state["root_pose_w"]
        root_pos_w = root_pose_w[..., :3]
        root_quat_w = root_pose_w[..., 3:7]
        root_yaw = yaw_quat(root_quat_w)

        eef_pos_w = command_state["eef_pos_w"]
        eef_quat_w = command_state["eef_quat_w"]

        # Goal: teacher world goal. Trajectory: hindsight eef[t+1].
        goal_pos_w = command_state["cmd_eef_pos_w"]
        traj_pos_w = self._shift_next_along_time(eef_pos_w, done)
        traj_quat_w = self._shift_next_along_time(eef_quat_w, done)

        goal_rot_w = command_state["cmd_eef_rot_w"]
        is_goal_exp = is_goal.unsqueeze(-1)
        cmd_eef_pos_w = torch.where(is_goal_exp, goal_pos_w, traj_pos_w)
        cmd_eef_rot_w = torch.where(is_goal_exp, goal_rot_w, traj_quat_w)

        cmd_eef_status = command_state["cmd_eef_status"]
        command_sparse, extras = self._build_sparse_command_from_targets(
            root_pos_w=root_pos_w,
            root_yaw_quat=root_yaw,
            eef_pos_w=eef_pos_w,
            eef_quat_w=eef_quat_w,
            cmd_eef_pos_w=cmd_eef_pos_w,
            cmd_eef_rot_w=cmd_eef_rot_w,
            cmd_eef_status=cmd_eef_status,
            device=device,
        )

        tensordict["command_state", "forward_diff_w"] = extras["forward_diff_w"]
        tensordict["command_state", "upward_diff_w"] = extras["upward_diff_w"]
        tensordict["command_state", "pos_error_norm2"] = extras["pos_error_norm2"]
        tensordict["command_state", "pos_error_norm"] = extras["pos_error_norm"]
        # Effective sparse targets (world goal or hindsight eef[t+1]).
        tensordict["command_state", "cmd_eef_pos_w"] = cmd_eef_pos_w
        tensordict["command_state", "cmd_eef_rot_w"] = cmd_eef_rot_w
        # Keep teacher base_pos_error for reward gates (0 in body/nominal).
        tensordict["command"] = command_sparse
        tensordict["next", "command"] = self._shift_next_along_time(
            command_sparse, done
        )
        return tensordict
    
    @staticmethod
    def _shift_next_along_time(
        x: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """For each step t, take x[t+1] unless done[t]; last step stays x[-1]."""
        out = torch.empty_like(x)
        # done: [T, N, 1] → broadcast over feature dims
        done_b = done
        while done_b.ndim < x.ndim:
            done_b = done_b.unsqueeze(-1)
        out[:-1] = torch.where(done_b[:-1], x[:-1], x[1:])
        out[-1] = x[-1]
        return out

    def _build_sparse_command_from_targets(
        self,
        root_pos_w: torch.Tensor,
        root_yaw_quat: torch.Tensor,
        eef_pos_w: torch.Tensor,
        eef_quat_w: torch.Tensor,
        cmd_eef_pos_w: torch.Tensor,
        cmd_eef_rot_w: torch.Tensor,
        cmd_eef_status: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared math for online-shaped sparse command + tracking fields."""
        forward_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
        upward_vec = torch.tensor([0.0, 0.0, 1.0], device=device)
        # Broadcast axes to [1, 1, 3] for [T, N, 3] or [1, 3] for [N, 3]
        lead = (1,) * (cmd_eef_pos_w.ndim - 1)
        forward_vec = forward_vec.reshape(*lead, 3)
        upward_vec = upward_vec.reshape(*lead, 3)

        xy_mask = torch.tensor([1.0, 1.0, 0.0], device=device).reshape(*lead, 3)
        cmd_eef_pos_b = quat_rotate_inverse(
            root_yaw_quat, cmd_eef_pos_w - root_pos_w * xy_mask
        )
        cmd_eef_forward_w = quat_rotate(cmd_eef_rot_w, forward_vec)
        cmd_eef_forward_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_forward_w)
        cmd_eef_upward_w = quat_rotate(cmd_eef_rot_w, upward_vec)
        cmd_eef_upward_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_upward_w)

        pos_diff_w = cmd_eef_pos_w - eef_pos_w
        pos_diff_b = quat_rotate_inverse(root_yaw_quat, pos_diff_w)
        pos_error_norm2 = pos_diff_w.square().sum(dim=-1, keepdim=True)
        pos_error_norm = pos_error_norm2.sqrt()

        eef_forward_w = quat_rotate(eef_quat_w, forward_vec)
        eef_upward_w = quat_rotate(eef_quat_w, upward_vec)
        forward_diff_w = cmd_eef_forward_w - eef_forward_w
        upward_diff_w = cmd_eef_upward_w - eef_upward_w
        forward_diff_b = quat_rotate_inverse(root_yaw_quat, forward_diff_w)
        upward_diff_b = quat_rotate_inverse(root_yaw_quat, upward_diff_w)

        cmd_status = cmd_eef_status.float()
        command_sparse = torch.cat(
            [
                cmd_eef_pos_b,
                pos_diff_b,
                cmd_eef_forward_b,
                forward_diff_b,
                cmd_eef_upward_b,
                upward_diff_b,
                cmd_status,
                1.0 - cmd_status,
            ],
            dim=-1,
        )
        extras = {
            "forward_diff_w": forward_diff_w,
            "upward_diff_w": upward_diff_w,
            "pos_error_norm2": pos_error_norm2,
            "pos_error_norm": pos_error_norm,
        }
        return command_sparse, extras

class LocoManipSparse(LocoManipSparseBase):
    """EEF-only loco-manip command (no base velocity command).

    Two online modes (mix controlled by ``trajectory_prob``):

    0. **Goal reaching** (sparse target): world-goal sample / update via the same
       Warp helpers as ``LocoManipNew`` mode 0 (polar annulus, standoff, heading-
       frame EEF refresh). Policy still sees EEF-only commands; ``base_pos_error``
       is computed for reward gates.
    1. **Trajectory following** (dense targets): a parametric curve (circle or
       line segment) in world frame, advanced each step.

    Policy command layout (heading / yaw-aligned frame):

    ``[eef_xyz, pos_diff, fwd, fwd_diff, up, up_diff, closed, open]``

    Relabel from ``LocoManipNew`` maps teacher world mode → goal reaching and
    body/nominal modes → trajectory following with hindsight target
    ``eef_pos_w[t+1]`` (and quat) at step ``t``.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        eef_z_range: Tuple[float, float] = (0.2, 0.8),
        world_goal_radius_range: Tuple[float, float] = (1.5, 3.0),
        # --- New mode-0 loco plumbing (no base cmd in sparse policy) ---
        # Used only to drive update_world_command → base_pos_error for reward gates;
        # commanded linvel / yaw from that helper are discarded.
        standoff_reach_range: Tuple[float, float] = (0.5, 0.7),
        world_linvel_gain_range: Tuple[float, float] = (1.0, 2.0),
        world_yaw_gain_range: Tuple[float, float] = (1.0, 2.0),
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
        # ---
        goal_spawn_radius_range: Tuple[float, float] = (0.0, 0.3),
        trajectory_prob: float = 0.5,
        traj_spawn_radius_range: Tuple[float, float] = (0.3, 1.0),
        curve_radius_range: Tuple[float, float] = (0.15, 0.4),
        curve_omega_range: Tuple[float, float] = (0.3, 1.2),
        curve_z_amp_range: Tuple[float, float] = (0.0, 0.08),
        curve_line_length_range: Tuple[float, float] = (0.3, 0.8),
        circle_prob: float = 0.6,
        resample_interval: int = 300,
        resample_prob: float = 0.75,
    ) -> None:
        self.eef_body_name = eef_body_name
        self.gripper_joint_names = gripper_joint_names
        self.eef_z_range = eef_z_range
        self.world_goal_radius_range = world_goal_radius_range
        self.standoff_reach_range = standoff_reach_range
        self.world_linvel_gain_range = world_linvel_gain_range
        self.world_yaw_gain_range = world_yaw_gain_range
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.goal_spawn_radius_range = goal_spawn_radius_range
        self.trajectory_prob = float(trajectory_prob)
        self.traj_spawn_radius_range = traj_spawn_radius_range
        self.curve_radius_range = curve_radius_range
        self.curve_omega_range = curve_omega_range
        self.curve_z_amp_range = curve_z_amp_range
        self.curve_line_length_range = curve_line_length_range
        self.circle_prob = float(circle_prob)
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)

        with torch.device(self.device):
            # Trajectory-following curve parameters (random Sparse only).
            self.curve_kind = torch.zeros(self.num_envs, dtype=torch.int32)
            self.curve_center_w = torch.zeros(self.num_envs, 3)
            self.curve_radius = torch.zeros(self.num_envs)
            self.curve_omega = torch.zeros(self.num_envs)
            self.curve_phase = torch.zeros(self.num_envs)
            self.curve_z_amp = torch.zeros(self.num_envs)
            self.curve_line_dir = torch.zeros(self.num_envs, 2)
            self.curve_line_half_len = torch.zeros(self.num_envs)

        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.init_eef_rot_b = quat_mul(
            quat_conjugate(root_yaw_q),
            self.asset.data.body_link_quat_w[:, self.eef_body_idx],
        )
        self.cmd_eef_rot_b[:] = self.init_eef_rot_b

        self._wp_device = wp.get_device(str(self.device))
        self._warp_seed = 0

        self.sync_state()

    # @override
    # def pre_step(self, substep: int) -> None:
    #     self.asset._external_force_b[:, self.eef_body_idx] = quat_rotate_inverse(
    #         self.asset.data.body_link_quat_w[:, self.eef_body_idx],
    #         self.payload_force_w,
    #     )
    #     self.asset.has_external_wrench = True

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        return (
            torch.rand(num_samples, device=self.device)
            * (value_range[1] - value_range[0])
            + value_range[0]
        )

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Spawn near env origin (goal) or on a tighter ring (traj)."""
        origins = self.env.scene.get_spawn_origins(env_ids)
        robot_init = self.init_root_state[env_ids].clone()
        default_z_offset = robot_init[:, 2].clone()

        is_traj = torch.rand(len(env_ids), device=self.device) < self.trajectory_prob
        self.sparse_mode[env_ids] = torch.where(
            is_traj,
            torch.full((), MODE_TRAJECTORY, dtype=torch.int32, device=self.device),
            torch.full((), MODE_GOAL_REACHING, dtype=torch.int32, device=self.device),
        )

        radius = torch.where(
            is_traj,
            self._sample_uniform(len(env_ids), self.traj_spawn_radius_range),
            self._sample_uniform(len(env_ids), self.goal_spawn_radius_range),
        )

        angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        robot_init[:, 0] = origins[:, 0] + radius * torch.cos(angle)
        robot_init[:, 1] = origins[:, 1] + radius * torch.sin(angle)
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_z_offset
        )
        robot_init[:, 3:7] = quat_mul(
            robot_init[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )
        return robot_init

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample world goals (Warp) and/or trajectory curves for ``env_ids``."""
        if env_ids.numel() == 0:
            return

        is_traj = self.sparse_mode[env_ids] == MODE_TRAJECTORY
        goal_ids = env_ids[~is_traj]
        traj_ids = env_ids[is_traj]

        if goal_ids.numel() > 0:
            resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            resample[goal_ids] = True
            self._warp_seed = (self._warp_seed + 1) % (2**31 - 1)
            root_pos_w = self.asset.data.root_link_pos_w.contiguous()
            root_yaw_xyzw = quat_wxyz_to_xyzw(
                yaw_quat(self.asset.data.root_link_quat_w)
            )
            wp.launch(
                kernel=sample_sparse_world_goal,
                dim=[self.num_envs],
                inputs=[
                    wp.from_torch(resample, dtype=wp.bool, return_ctype=True),
                    self._warp_seed,
                    self.eef_z_range[0],
                    self.eef_z_range[1],
                    self.world_goal_radius_range[0],
                    self.world_goal_radius_range[1],
                    self.standoff_reach_range[0],
                    self.standoff_reach_range[1],
                    self.world_linvel_gain_range[0],
                    self.world_linvel_gain_range[1],
                    self.world_yaw_gain_range[0],
                    self.world_yaw_gain_range[1],
                    wp.from_torch(root_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(root_yaw_xyzw, dtype=wp.quat, return_ctype=True),
                ],
                outputs=[
                    wp.from_torch(self.cmd_eef_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(
                        self.cmd_eef_status[:, 0], dtype=wp.int32, return_ctype=True
                    ),
                    wp.from_torch(self.standoff_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(
                        self.standoff_yaw_w, dtype=wp.float32, return_ctype=True
                    ),
                    wp.from_torch(
                        self.world_linvel_gain, dtype=wp.float32, return_ctype=True
                    ),
                    wp.from_torch(
                        self.world_yaw_gain, dtype=wp.float32, return_ctype=True
                    ),
                ],
                device=self._wp_device,
            )
            self.world_eef_pos_w[resample] = self.cmd_eef_pos_w[resample]
            self.cmd_eef_rot_b[goal_ids] = self.init_eef_rot_b[goal_ids]
            self.base_pos_error[goal_ids] = 0.0

        if traj_ids.numel() > 0:
            n = traj_ids.numel()
            origins = self.env.scene.env_origins[traj_ids]
            center = origins.clone()
            center[:, 2] = (
                self.env.get_ground_height_at(origins)
                + self._sample_uniform(n, self.eef_z_range)
            )
            self.curve_center_w[traj_ids] = center
            is_circle = torch.rand(n, device=self.device) < self.circle_prob
            self.curve_kind[traj_ids] = torch.where(
                is_circle,
                torch.full((), CURVE_CIRCLE, dtype=torch.int32, device=self.device),
                torch.full((), CURVE_LINE, dtype=torch.int32, device=self.device),
            )
            self.curve_radius[traj_ids] = self._sample_uniform(
                n, self.curve_radius_range
            )
            self.curve_omega[traj_ids] = self._sample_uniform(n, self.curve_omega_range)
            self.curve_phase[traj_ids] = torch.rand(n, device=self.device) * 2 * torch.pi
            self.curve_z_amp[traj_ids] = self._sample_uniform(n, self.curve_z_amp_range)
            ang = torch.rand(n, device=self.device) * 2 * torch.pi
            self.curve_line_dir[traj_ids] = torch.stack(
                [torch.cos(ang), torch.sin(ang)], dim=-1
            )
            self.curve_line_half_len[traj_ids] = (
                self._sample_uniform(n, self.curve_line_length_range) * 0.5
            )
            self._eval_curve_into(traj_ids)
            self.cmd_eef_rot_b[traj_ids] = self.init_eef_rot_b[traj_ids]
            self.cmd_eef_status[traj_ids] = (
                torch.rand(n, 1, device=self.device) < 0.5
            ).to(dtype=torch.int32)
            self.base_pos_error[traj_ids] = 0.0

    def _eval_curve_into(self, env_ids: torch.Tensor) -> None:
        """Write parametric curve targets at current phase into cmd/world EEF buffers."""
        center = self.curve_center_w[env_ids]
        phase = self.curve_phase[env_ids]
        radius = self.curve_radius[env_ids]
        circle_xy = torch.stack(
            [radius * torch.cos(phase), radius * torch.sin(phase)], dim=-1
        )
        t = torch.sin(phase)
        line_xy = self.curve_line_dir[env_ids] * (
            self.curve_line_half_len[env_ids] * t
        ).unsqueeze(-1)
        is_circle = (self.curve_kind[env_ids] == CURVE_CIRCLE).unsqueeze(-1)
        offset_xy = torch.where(is_circle, circle_xy, line_xy)
        z = center[:, 2] + self.curve_z_amp[env_ids] * torch.sin(phase)
        target = torch.stack(
            [center[:, 0] + offset_xy[:, 0], center[:, 1] + offset_xy[:, 1], z],
            dim=-1,
        )
        self.world_eef_pos_w[env_ids] = target
        self.cmd_eef_pos_w[env_ids] = target

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.eef_pos_reached[env_ids] = False
        self.sample_commands(env_ids)
        # Heading-frame EEF for newly sampled world goals (obs/reward before first update).
        self._refresh_cmd_eef_pos_b()

    @override
    def sync_state(self) -> None:
        """Tracking / reach flags for rewards at THIS step (no target mutation)."""
        root_pos_w = self.asset.data.root_link_pos_w
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.eef_status = self.get_gripper_status()

        # World track point from heading-frame cmd (same as LocoManipNew).
        offset_xy = self.cmd_eef_pos_b.clone()
        offset_xy[:, 2] = 0.0
        track_eef_pos_w = root_pos_w * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        ) + quat_rotate(root_yaw_q, offset_xy)
        track_eef_pos_w[:, 2] = self.cmd_eef_pos_b[:, 2]

        self.pos_diff_w = track_eef_pos_w - self.eef_pos_w
        self.pos_diff_b = quat_rotate_inverse(root_yaw_q, self.pos_diff_w)
        self.pos_error_norm2 = self.pos_diff_w.square().sum(dim=-1, keepdim=True)
        self.pos_error_norm = self.pos_error_norm2.sqrt()

        # Orientation: always nominal body/yaw-frame rest pose.
        self.cmd_eef_rot_b[:] = self.init_eef_rot_b
        self.cmd_eef_rot_w = quat_mul(root_yaw_q, self.cmd_eef_rot_b)

        forward_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.cmd_eef_forward_w = quat_rotate(self.cmd_eef_rot_w, forward_axis)
        self.cmd_eef_upward_w = quat_rotate(self.cmd_eef_rot_w, upward_axis)
        self.cmd_eef_forward_b = quat_rotate_inverse(root_yaw_q, self.cmd_eef_forward_w)
        self.cmd_eef_upward_b = quat_rotate_inverse(root_yaw_q, self.cmd_eef_upward_w)

        eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis)
        eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis)
        self.forward_diff_w = self.cmd_eef_forward_w - eef_forward_w
        self.upward_diff_w = self.cmd_eef_upward_w - eef_upward_w
        self.forward_diff_b = quat_rotate_inverse(root_yaw_q, self.forward_diff_w)
        self.upward_diff_b = quat_rotate_inverse(root_yaw_q, self.upward_diff_w)

        reached = self.pos_error_norm < 0.08
        is_goal = (self.sparse_mode == MODE_GOAL_REACHING).unsqueeze(-1)
        self.eef_pos_reached = torch.where(is_goal, reached, self.eef_pos_reached)

    def _refresh_cmd_eef_pos_b(self) -> None:
        """Refresh heading-frame EEF from world targets (goal: Warp; traj: torch)."""
        if (self.sparse_mode == MODE_GOAL_REACHING).any():
            root_pos_w = self.asset.data.root_link_pos_w.contiguous()
            root_yaw_xyzw = quat_wxyz_to_xyzw(
                yaw_quat(self.asset.data.root_link_quat_w)
            )
            heading_w = self.asset.data.heading_w.contiguous()
            wp.launch(
                kernel=update_sparse_world_command,
                dim=[self.num_envs],
                inputs=[
                    wp.from_torch(self.sparse_mode, dtype=wp.int32, return_ctype=True),
                    wp.from_torch(root_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(root_yaw_xyzw, dtype=wp.quat, return_ctype=True),
                    wp.from_torch(heading_w, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(self.cmd_eef_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(self.standoff_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(
                        self.standoff_yaw_w, dtype=wp.float32, return_ctype=True
                    ),
                    wp.from_torch(
                        self.world_linvel_gain, dtype=wp.float32, return_ctype=True
                    ),
                    wp.from_torch(
                        self.world_yaw_gain, dtype=wp.float32, return_ctype=True
                    ),
                    self.linvel_x_range[0],
                    self.linvel_x_range[1],
                    self.linvel_y_range[0],
                    self.linvel_y_range[1],
                    self.yaw_rate_range[0],
                    self.yaw_rate_range[1],
                ],
                outputs=[
                    wp.from_torch(self.cmd_eef_pos_b, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(
                        self.base_pos_error[:, 0], dtype=wp.float32, return_ctype=True
                    ),
                ],
                device=self._wp_device,
            )

        traj = self.sparse_mode == MODE_TRAJECTORY
        if traj.any():
            root_pos_w = self.asset.data.root_link_pos_w
            root_yaw = yaw_quat(self.asset.data.root_link_quat_w)
            xy_mask = torch.tensor([1.0, 1.0, 0.0], device=self.device)
            self.cmd_eef_pos_b[traj] = quat_rotate_inverse(
                root_yaw[traj],
                self.cmd_eef_pos_w[traj] - root_pos_w[traj] * xy_mask,
            )
            self.base_pos_error[traj] = 0.0

    @override
    def update(self) -> None:
        """Advance / resample targets; refresh obs-facing ``cmd_eef_pos_b``.

        Does not recompute tracking — that is ``sync_state`` after the next physics
        step (same split as ``LocoManipNew``).
        """
        traj = self.sparse_mode == MODE_TRAJECTORY
        if traj.any():
            self.curve_phase[traj] = (
                self.curve_phase[traj] + self.curve_omega[traj] * self.env.step_dt
            )
            self._eval_curve_into(traj.nonzero(as_tuple=False).squeeze(-1))

        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        prob_ok = torch.rand(self.num_envs, device=self.device) < self.resample_prob
        is_goal = self.sparse_mode == MODE_GOAL_REACHING
        resample = interval & prob_ok & (
            (is_goal & self.eef_pos_reached.squeeze(1)) | traj
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            flip = torch.rand(len(env_ids), device=self.device) < self.trajectory_prob
            self.sparse_mode[env_ids] = torch.where(
                flip,
                torch.full((), MODE_TRAJECTORY, dtype=torch.int32, device=self.device),
                torch.full(
                    (), MODE_GOAL_REACHING, dtype=torch.int32, device=self.device
                ),
            )
            self.eef_pos_reached[env_ids] = False
            self.sample_commands(env_ids)

        self._refresh_cmd_eef_pos_b()


class LocoManipSparseReplay(LocoManipSparseBase):
    """Online world-goal commands from relative transforms mined from a teacher rollout.

    From ``LocoManipNew`` rollouts, keeps world-mode (``mode == 0``) **sample** events
    and stores each goal as a yaw-frame relative position w.r.t. the robot pose at
    that sample. Online sampling draws a catalog entry and applies it to the live
    root (spawn near env origins). Resample schedule matches ``LocoManipNew``:
    every ``resample_interval`` steps with probability ``resample_prob``.

    Policy command layout matches ``LocoManipSparse`` (EEF-only, 20D). There is
    **no base velocity command**; standoff / loco-gain fields exist only so the
    shared ``update_world_command`` Warp path (and ``base_pos_error`` reward gate)
    stay consistent with ``LocoManipNew`` mode 0.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        rollout_path: str,
        # --- New mode-0 loco plumbing (unused as policy outputs) ---
        # Rebuild standoff + clamp ranges for update_world_command; discarded
        # linvel/yaw outputs; base_pos_error kept for reward gates only.
        standoff_reach_range: Tuple[float, float] = (0.5, 0.7),
        world_linvel_gain_range: Tuple[float, float] = (1.0, 2.0),
        world_yaw_gain_range: Tuple[float, float] = (1.0, 2.0),
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
        # ---
        goal_spawn_radius_range: Tuple[float, float] = (0.0, 0.3),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
    ) -> None:
        super().__init__()
        self.eef_body_name = eef_body_name
        self.gripper_joint_names = gripper_joint_names
        self.rollout_path = rollout_path
        # LocoManipNew-consistency only (see class docstring).
        self.standoff_reach_range = standoff_reach_range
        self.world_linvel_gain_range = world_linvel_gain_range
        self.world_yaw_gain_range = world_yaw_gain_range
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.goal_spawn_radius_range = goal_spawn_radius_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

    @staticmethod
    def _load_stacked_rollout(rollout_path: str) -> TensorDict:
        path = Path(rollout_path).expanduser().resolve()
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(payload, dict) and "stacked" in payload:
            stacked = payload["stacked"]
        elif isinstance(payload, TensorDict):
            stacked = payload
        else:
            raise ValueError(
                f"Expected rollout dict with 'stacked' or a TensorDict, got {type(payload)}"
            )
        if not isinstance(stacked, TensorDict) or len(stacked.batch_size) < 2:
            raise ValueError(
                f"Expected stacked TensorDict with batch [T, N], got {type(stacked)} "
                f"batch_size={getattr(stacked, 'batch_size', None)}"
            )
        return stacked

    @staticmethod
    def _extract_world_goal_catalog(
        stacked: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mine relative goal↔root transforms at world-mode sample events.

        A sample event is a world-mode (``mode == 0``) step that is an episode
        init, a transition into world mode, or a change of ``cmd_eef_pos_w``.

        Returns:
            goal_pos_b: [M, 3] yaw-frame relative position of the goal w.r.t.
                teacher root at sample time (xy from root xy; z = goal height)
            cmd_eef_status: [M, 1] int32
        """
        cs = stacked["command_state"]
        if hasattr(cs, "to_tensordict"):
            cs = cs.to_tensordict()

        mode = cs["mode"]
        if mode.ndim == cs["eef_pos_w"].ndim:
            mode = mode.squeeze(-1)
        is_world = mode == 0  # [T, N]

        T, N = is_world.shape
        prev_world = torch.zeros_like(is_world)
        prev_world[1:] = is_world[:-1]

        is_init = stacked.get("is_init")
        if is_init is not None:
            is_init = is_init.squeeze(-1).bool()
        else:
            is_init = torch.zeros(T, N, dtype=torch.bool)
            is_init[0] = True

        cmd_eef_pos_w = cs["cmd_eef_pos_w"]
        goal_changed = torch.zeros_like(is_world)
        goal_changed[1:] = (
            (cmd_eef_pos_w[1:] - cmd_eef_pos_w[:-1]).abs().sum(dim=-1) > 1e-5
        ) & is_world[1:] & is_world[:-1]

        entered_world = is_world & ~prev_world
        onset = is_world & (is_init | entered_world | goal_changed)
        if not onset.any():
            raise ValueError(
                "No world-mode (mode==0) goal sample events found in rollout; "
                "cannot build LocoManipSparseReplay catalog."
            )

        root_pose_w = cs["root_pose_w"][onset]  # [M, 7]
        goals_w = cmd_eef_pos_w[onset]
        cmd_eef_status = cs["cmd_eef_status"][onset]
        if cmd_eef_status.ndim == 1:
            cmd_eef_status = cmd_eef_status.unsqueeze(-1)

        root_pos_w = root_pose_w[..., :3]
        root_yaw = yaw_quat(root_pose_w[..., 3:7])
        # Same heading-frame convention as online world goals: xy vs root xy, z abs.
        xy_mask = torch.tensor([1.0, 1.0, 0.0])
        goal_pos_b = quat_rotate_inverse(root_yaw, goals_w - root_pos_w * xy_mask)
        return goal_pos_b.contiguous(), cmd_eef_status.to(dtype=torch.int32).contiguous()

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.sparse_mode[:] = MODE_GOAL_REACHING

        stacked = self._load_stacked_rollout(self.rollout_path)
        goal_pos_b, cmd_eef_status = self._extract_world_goal_catalog(stacked)
        self.catalog_goal_pos_b = goal_pos_b.to(device=self.device)
        self.catalog_cmd_eef_status = cmd_eef_status.to(device=self.device)
        self.catalog_size = int(self.catalog_goal_pos_b.shape[0])
        del stacked

        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.init_eef_rot_b = quat_mul(
            quat_conjugate(root_yaw_q),
            self.asset.data.body_link_quat_w[:, self.eef_body_idx],
        )
        self.cmd_eef_rot_b[:] = self.init_eef_rot_b

        self._wp_device = wp.get_device(str(self.device))

        self.sync_state()

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        return (
            torch.rand(num_samples, device=self.device)
            * (value_range[1] - value_range[0])
            + value_range[0]
        )

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Spawn near env origins (small polar jitter)."""
        origins = self.env.scene.get_spawn_origins(env_ids)
        robot_init = self.init_root_state[env_ids].clone()
        default_z_offset = robot_init[:, 2].clone()
        n = len(env_ids)
        radius = self._sample_uniform(n, self.goal_spawn_radius_range)
        angle = torch.rand(n, device=self.device) * 2 * torch.pi
        robot_init[:, 0] = origins[:, 0] + radius * torch.cos(angle)
        robot_init[:, 1] = origins[:, 1] + radius * torch.sin(angle)
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_z_offset
        )
        robot_init[:, 3:7] = quat_mul(
            robot_init[:, 3:7],
            sample_quat_yaw(n, device=self.device),
        )
        self.sparse_mode[env_ids] = MODE_GOAL_REACHING
        return robot_init

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        """Apply a catalog relative transform to the current robot pose."""
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        idx = torch.randint(0, self.catalog_size, (n,), device=self.device)
        goal_pos_b = self.catalog_goal_pos_b[idx]
        status = self.catalog_cmd_eef_status[idx]

        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw = yaw_quat(self.asset.data.root_link_quat_w[env_ids])
        xy_mask = torch.tensor([1.0, 1.0, 0.0], device=self.device)
        # Inverse of catalog extract: world goal from live root + relative transform.
        goal = root_pos_w * xy_mask + quat_rotate(root_yaw, goal_pos_b * xy_mask)
        goal[:, 2] = goal_pos_b[:, 2]

        self.cmd_eef_pos_w[env_ids] = goal
        self.world_eef_pos_w[env_ids] = goal
        self.cmd_eef_status[env_ids] = status
        self.cmd_eef_rot_b[env_ids] = self.init_eef_rot_b[env_ids]
        self.base_pos_error[env_ids] = 0.0  # recomputed in _refresh; reward-gate only
        self._rebuild_standoff(env_ids)

    def _rebuild_standoff(self, env_ids: torch.Tensor) -> None:
        """Fill New-style standoff / gains (not issued as base commands).

        Needed so ``update_sparse_world_command`` can write ``base_pos_error``;
        the Warp helper's linvel / yaw outputs are unused by the sparse policy.
        """
        n = env_ids.numel()
        root_xy = self.asset.data.root_link_pos_w[env_ids] * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        )
        goal_xy = self.cmd_eef_pos_w[env_ids] * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        )
        to_goal = goal_xy - root_xy
        dist = to_goal[:, :2].norm(dim=-1)
        reach = self._sample_uniform(n, self.standoff_reach_range)
        inv = torch.where(dist > 1e-6, 1.0 / dist, torch.zeros_like(dist))
        standoff = torch.where(
            (dist > reach).unsqueeze(-1),
            goal_xy - to_goal * (reach * inv).unsqueeze(-1),
            root_xy,
        )
        # Unused as policy outputs — New mode-0 consistency only:
        self.standoff_pos_w[env_ids] = standoff
        self.standoff_yaw_w[env_ids] = torch.atan2(
            goal_xy[:, 1] - standoff[:, 1],
            goal_xy[:, 0] - standoff[:, 0],
        )
        self.world_linvel_gain[env_ids] = self._sample_uniform(
            n, self.world_linvel_gain_range
        )
        self.world_yaw_gain[env_ids] = self._sample_uniform(
            n, self.world_yaw_gain_range
        )

    def _refresh_cmd_eef_pos_b(self) -> None:
        """Refresh heading-frame EEF; also writes ``base_pos_error`` via New Warp.

        Standoff / gain / linvel-yaw clamp args are New mode-0 plumbing: the kernel
        returns unused loco cmds; we only keep ``cmd_eef_pos_b`` and ``base_pos_error``.
        """
        root_pos_w = self.asset.data.root_link_pos_w.contiguous()
        root_yaw_xyzw = quat_wxyz_to_xyzw(yaw_quat(self.asset.data.root_link_quat_w))
        heading_w = self.asset.data.heading_w.contiguous()
        wp.launch(
            kernel=update_sparse_world_command,
            dim=[self.num_envs],
            inputs=[
                wp.from_torch(self.sparse_mode, dtype=wp.int32, return_ctype=True),
                wp.from_torch(root_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(root_yaw_xyzw, dtype=wp.quat, return_ctype=True),
                wp.from_torch(heading_w, dtype=wp.float32, return_ctype=True),
                wp.from_torch(self.cmd_eef_pos_w, dtype=wp.vec3, return_ctype=True),
                # Unused as base cmds — required inputs for New's world update:
                wp.from_torch(self.standoff_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.standoff_yaw_w, dtype=wp.float32, return_ctype=True),
                wp.from_torch(
                    self.world_linvel_gain, dtype=wp.float32, return_ctype=True
                ),
                wp.from_torch(self.world_yaw_gain, dtype=wp.float32, return_ctype=True),
                self.linvel_x_range[0],
                self.linvel_x_range[1],
                self.linvel_y_range[0],
                self.linvel_y_range[1],
                self.yaw_rate_range[0],
                self.yaw_rate_range[1],
            ],
            outputs=[
                wp.from_torch(self.cmd_eef_pos_b, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(
                    self.base_pos_error[:, 0], dtype=wp.float32, return_ctype=True
                ),
            ],
            device=self._wp_device,
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.eef_pos_reached[env_ids] = False
        self.sparse_mode[env_ids] = MODE_GOAL_REACHING
        self.sample_commands(env_ids)
        self._refresh_cmd_eef_pos_b()

    @override
    def sync_state(self) -> None:
        """Tracking / reach flags for rewards at THIS step (no target mutation)."""
        root_pos_w = self.asset.data.root_link_pos_w
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.eef_status = self.get_gripper_status()

        offset_xy = self.cmd_eef_pos_b.clone()
        offset_xy[:, 2] = 0.0
        track_eef_pos_w = root_pos_w * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        ) + quat_rotate(root_yaw_q, offset_xy)
        track_eef_pos_w[:, 2] = self.cmd_eef_pos_b[:, 2]

        self.pos_diff_w = track_eef_pos_w - self.eef_pos_w
        self.pos_diff_b = quat_rotate_inverse(root_yaw_q, self.pos_diff_w)
        self.pos_error_norm2 = self.pos_diff_w.square().sum(dim=-1, keepdim=True)
        self.pos_error_norm = self.pos_error_norm2.sqrt()

        self.cmd_eef_rot_b[:] = self.init_eef_rot_b
        self.cmd_eef_rot_w = quat_mul(root_yaw_q, self.cmd_eef_rot_b)

        forward_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.cmd_eef_forward_w = quat_rotate(self.cmd_eef_rot_w, forward_axis)
        self.cmd_eef_upward_w = quat_rotate(self.cmd_eef_rot_w, upward_axis)
        self.cmd_eef_forward_b = quat_rotate_inverse(root_yaw_q, self.cmd_eef_forward_w)
        self.cmd_eef_upward_b = quat_rotate_inverse(root_yaw_q, self.cmd_eef_upward_w)

        eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis)
        eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis)
        self.forward_diff_w = self.cmd_eef_forward_w - eef_forward_w
        self.upward_diff_w = self.cmd_eef_upward_w - eef_upward_w
        self.forward_diff_b = quat_rotate_inverse(root_yaw_q, self.forward_diff_w)
        self.upward_diff_b = quat_rotate_inverse(root_yaw_q, self.upward_diff_w)

        self.eef_pos_reached = self.pos_error_norm < 0.08

    @override
    def update(self) -> None:
        """Resample like ``LocoManipNew``; refresh obs-facing ``cmd_eef_pos_b``."""
        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample = interval & (
            torch.rand(self.num_envs, device=self.device) < self.resample_prob
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.sample_commands(env_ids)
        self._refresh_cmd_eef_pos_b()


__all__ = ["LocoManipSparse", "LocoManipSparseReplay"]
