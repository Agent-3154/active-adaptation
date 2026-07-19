from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import warp as wp
from typing_extensions import override
from tensordict import TensorDict

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
    """Refresh heading-frame EEF + base_pos_error for goal-reaching envs."""
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


class LocoManipSparse(CommandV2):
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
        standoff_reach_range: Tuple[float, float] = (0.5, 0.7),
        world_linvel_gain_range: Tuple[float, float] = (1.0, 2.0),
        world_yaw_gain_range: Tuple[float, float] = (1.0, 2.0),
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
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
        cmd_eef_pos_clamp_range: float = -1.0,
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
        self.cmd_eef_pos_clamp_range = cmd_eef_pos_clamp_range

    @override
    def _initialize(self, env: "_EnvBase") -> None:
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

            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_w = torch.zeros(self.num_envs, 3)

            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)

            self.forward_diff_w = torch.zeros(self.num_envs, 3)
            self.forward_diff_b = torch.zeros(self.num_envs, 3)
            self.upward_diff_w = torch.zeros(self.num_envs, 3)
            self.upward_diff_b = torch.zeros(self.num_envs, 3)

            # orientation tracking
            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)
            self.cmd_eef_rot_b = torch.zeros(self.num_envs, 4)

            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_b = torch.zeros(self.num_envs, 3)

            # gripper closedness in [0, 1]: 0 = open, 1 = closed
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.int32)

            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached_time = torch.zeros(self.num_envs, 1, dtype=torch.float)

            # payload applied at grasp point (force, unit: N)
            self.has_payload = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.payload_force_w = torch.zeros(self.num_envs, 3)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.base_pos_error = torch.zeros(self.num_envs, 1)

            # World-goal standoff state (same buffers as LocoManipNew mode 0)
            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs)
            self.world_linvel_gain = torch.ones(self.num_envs)
            self.world_yaw_gain = torch.ones(self.num_envs)

            # Trajectory-following curve parameters
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

        self.marker = None
        self.eef_pose_marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.eef_pose_marker = self.scene.create_frame_marker(
                "/Visuals/Command/target_eef_pose",
                scale=(0.1, 0.1, 0.1),
            )
        self.sync_state()

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

    # @override
    # def pre_step(self, substep: int) -> None:
    #     self.asset._external_force_b[:, self.eef_body_idx] = quat_rotate_inverse(
    #         self.asset.data.body_link_quat_w[:, self.eef_body_idx],
    #         self.payload_force_w,
    #     )
    #     self.asset.has_external_wrench = True

    def get_gripper_status(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed)."""
        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        return 1.0 - openness

    @staticmethod
    def _env_mask_prob(num_envs: int, prob: float, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, device=device) < prob

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
        """Spawn near env origin (goal) or near the curve (traj), matching teacher layout."""
        origins = self.env.scene.get_spawn_origins(env_ids)
        robot_init = self.init_root_state[env_ids].clone()
        default_z_offset = robot_init[:, 2].clone()

        is_traj = torch.rand(len(env_ids), device=self.device) < self.trajectory_prob
        self.sparse_mode[env_ids] = torch.where(
            is_traj,
            torch.full((), MODE_TRAJECTORY, dtype=torch.int32, device=self.device),
            torch.full((), MODE_GOAL_REACHING, dtype=torch.int32, device=self.device),
        )

        # Goal: small jitter about origin (like LocoManipNew). Traj: tighter ring.
        radius = torch.empty(len(env_ids), device=self.device)
        if is_traj.any():
            n_traj = int(is_traj.sum())
            radius[is_traj] = self._sample_uniform(n_traj, self.traj_spawn_radius_range)
        if (~is_traj).any():
            n_goal = int((~is_traj).sum())
            radius[~is_traj] = self._sample_uniform(n_goal, self.goal_spawn_radius_range)

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

    def _sample_goal_commands(self, env_ids: torch.Tensor) -> None:
        """Sample world goals via the shared Warp kernel (same as LocoManipNew mode 0)."""
        if env_ids.numel() == 0:
            return
        resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        resample[env_ids] = True
        self._sample_world_goals_warp(resample)

    def _sample_world_goals_warp(self, resample: torch.Tensor) -> None:
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
        # Keep world_eef_pos_w in sync for traj helpers / debug.
        self.world_eef_pos_w[resample] = self.cmd_eef_pos_w[resample]
        # Orientation: nominal rest pose (same as LocoManipNew).
        goal_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if goal_ids.numel() > 0:
            self.cmd_eef_rot_b[goal_ids] = self.init_eef_rot_b[goal_ids]
            self.base_pos_error[goal_ids] = 0.0

    def _eval_curve(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate parametric curve at current phase → world EEF targets [M, 3]."""
        center = self.curve_center_w[env_ids]
        phase = self.curve_phase[env_ids]
        z_amp = self.curve_z_amp[env_ids]
        kind = self.curve_kind[env_ids]

        # Circle
        radius = self.curve_radius[env_ids]
        circle_xy = torch.stack(
            [radius * torch.cos(phase), radius * torch.sin(phase)], dim=-1
        )

        # Line: ping-pong along direction with half-length
        half_len = self.curve_line_half_len[env_ids]
        direction = self.curve_line_dir[env_ids]
        t = torch.sin(phase)  # [-1, 1]
        line_xy = direction * (half_len * t).unsqueeze(-1)

        is_circle = kind == CURVE_CIRCLE
        offset_xy = torch.where(is_circle.unsqueeze(-1), circle_xy, line_xy)
        z = center[:, 2] + z_amp * torch.sin(phase)
        return torch.stack(
            [center[:, 0] + offset_xy[:, 0], center[:, 1] + offset_xy[:, 1], z],
            dim=-1,
        )

    def _sample_trajectory_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        origins = self.env.scene.env_origins[env_ids]
        z_offset = self._sample_uniform(n, self.eef_z_range)
        center = origins.clone()
        center[:, 2] = self.env.get_ground_height_at(origins) + z_offset
        self.curve_center_w[env_ids] = center

        is_circle = torch.rand(n, device=self.device) < self.circle_prob
        self.curve_kind[env_ids] = torch.where(
            is_circle,
            torch.full((), CURVE_CIRCLE, dtype=torch.int32, device=self.device),
            torch.full((), CURVE_LINE, dtype=torch.int32, device=self.device),
        )
        self.curve_radius[env_ids] = self._sample_uniform(n, self.curve_radius_range)
        self.curve_omega[env_ids] = self._sample_uniform(n, self.curve_omega_range)
        self.curve_phase[env_ids] = torch.rand(n, device=self.device) * 2 * torch.pi
        self.curve_z_amp[env_ids] = self._sample_uniform(n, self.curve_z_amp_range)

        # Random horizontal direction for line segments
        ang = torch.rand(n, device=self.device) * 2 * torch.pi
        self.curve_line_dir[env_ids] = torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1
        )
        self.curve_line_half_len[env_ids] = (
            self._sample_uniform(n, self.curve_line_length_range) * 0.5
        )

        self.world_eef_pos_w[env_ids] = self._eval_curve(env_ids)
        self.cmd_eef_pos_w[env_ids] = self.world_eef_pos_w[env_ids]
        self.cmd_eef_rot_b[env_ids] = self.init_eef_rot_b[env_ids]
        self.cmd_eef_status[env_ids] = (
            torch.rand(n, 1, device=self.device) < 0.5
        ).to(dtype=torch.int32)
        self.base_pos_error[env_ids] = 0.0

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        # Keep mode assigned at reset; mid-episode resample stays in the same mode.
        is_traj = self.sparse_mode[env_ids] == MODE_TRAJECTORY
        if (~is_traj).any():
            self._sample_goal_commands(env_ids[~is_traj])
        if is_traj.any():
            self._sample_trajectory_commands(env_ids[is_traj])

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        # Mode is assigned in ``sample_init`` (always called before this for env_ids).
        self.eef_pos_reached[env_ids] = False
        self.sample_commands(env_ids)
        # Populate heading-frame EEF for newly sampled world goals.
        self._update_world_commands_warp()

    def _read_robot_state(self) -> None:
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        self.eef_quat_w = self.asset.data.body_link_quat_w[:, self.eef_body_idx]
        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis_b)
        self.eef_forward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_forward_w)
        self.eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis_b)
        self.eef_upward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_upward_w)
        self.eef_status = self.get_gripper_status()

    def _maybe_resample_commands(self) -> None:
        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        prob_ok = self._env_mask_prob(self.num_envs, self.resample_prob, self.device)
        is_goal = self.sparse_mode == MODE_GOAL_REACHING
        is_traj = self.sparse_mode == MODE_TRAJECTORY
        # Goal: only resample after reaching; trajectory: free to resample on interval.
        resample = interval & prob_ok & (
            (is_goal & self.eef_pos_reached.squeeze(1)) | is_traj
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

    def _advance_trajectories(self) -> None:
        traj = self.sparse_mode == MODE_TRAJECTORY
        if not traj.any():
            return
        self.curve_phase[traj] = (
            self.curve_phase[traj] + self.curve_omega[traj] * self.env.step_dt
        )
        env_ids = traj.nonzero(as_tuple=False).squeeze(-1)
        self.world_eef_pos_w[env_ids] = self._eval_curve(env_ids)
        self.cmd_eef_pos_w[env_ids] = self.world_eef_pos_w[env_ids]

    def _update_world_commands_warp(self) -> None:
        """Refresh cmd_eef_pos_b / base_pos_error for goal envs (LocoManipNew mode 0)."""
        if not (self.sparse_mode == MODE_GOAL_REACHING).any():
            return
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

    def _sync_traj_command_from_targets(self) -> None:
        """Heading-frame EEF for trajectory envs (world target → body)."""
        traj = self.sparse_mode == MODE_TRAJECTORY
        if not traj.any():
            return
        root_pos_w = self.root_pos_w
        root_yaw = self.root_yaw_quat
        xy_mask = torch.tensor([1.0, 1.0, 0.0], device=self.device)
        self.cmd_eef_pos_b[traj] = quat_rotate_inverse(
            root_yaw[traj],
            self.cmd_eef_pos_w[traj] - root_pos_w[traj] * xy_mask,
        )
        self.base_pos_error[traj] = 0.0

    def _sync_orientation_and_errors(self) -> None:
        """Nominal orientation + tracking errors (aligned with LocoManipNew.sync_state)."""
        root_pos_w = self.root_pos_w
        root_yaw_q = self.root_yaw_quat

        # Reconstruct world track point from heading-frame cmd (same as New).
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

        self.forward_diff_w = self.cmd_eef_forward_w - self.eef_forward_w
        self.forward_diff_b = quat_rotate_inverse(root_yaw_q, self.forward_diff_w)
        self.upward_diff_w = self.cmd_eef_upward_w - self.eef_upward_w
        self.upward_diff_b = quat_rotate_inverse(root_yaw_q, self.upward_diff_w)

        reached = self.pos_error_norm < 0.08
        is_goal = (self.sparse_mode == MODE_GOAL_REACHING).unsqueeze(-1)
        self.eef_pos_reached = torch.where(is_goal, reached, self.eef_pos_reached)

    @override
    def sync_state(self) -> None:
        """Refresh tracking errors from post-physics robot state and current targets."""
        self._read_robot_state()
        self._sync_orientation_and_errors()

    @override
    def update(self) -> None:
        """Advance curves / resample, then refresh world-mode cmds via Warp."""
        self._advance_trajectories()
        self._maybe_resample_commands()
        self._update_world_commands_warp()
        self._read_robot_state()
        self._sync_traj_command_from_targets()
        self._sync_orientation_and_errors()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        if self.eef_pose_marker is not None:
            self.eef_pose_marker.visualize(
                translations=self.cmd_eef_pos_w,
                orientations=self.cmd_eef_rot_w,
            )

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

        if self.cmd_eef_pos_clamp_range > 0.0:
            cmd_eef_pos_w = root_pos_w + clamp_norm(
                cmd_eef_pos_w - root_pos_w, max=self.cmd_eef_pos_clamp_range
            )

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


__all__ = ["LocoManipSparse"]
