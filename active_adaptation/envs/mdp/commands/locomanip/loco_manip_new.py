from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple
from typing_extensions import override

import torch
import warp as wp

from active_adaptation.envs.mdp.commands.base import CommandV2
from active_adaptation.envs.mdp.commands.locomanip.loco_manip_kernels import (
    quat_wxyz_to_xyzw,
    sample_world_goal,
    update_world_command,
)
from active_adaptation.utils.math import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from tensordict import TensorClass, TensorDictBase

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


@wp.kernel(enable_backward=False)
def sample_commands(
    resample: wp.array(dtype=wp.bool),
    seed: wp.int32,
    mode_p0: wp.float32,
    mode_p1: wp.float32,
    eef_x_min: wp.float32,
    eef_x_max: wp.float32,
    eef_y_min: wp.float32,
    eef_y_max: wp.float32,
    eef_z_min: wp.float32,
    eef_z_max: wp.float32,
    world_radius_min: wp.float32,
    world_radius_max: wp.float32,
    standoff_reach_min: wp.float32,
    standoff_reach_max: wp.float32,
    linvel_x_min: wp.float32,
    linvel_x_max: wp.float32,
    linvel_y_min: wp.float32,
    linvel_y_max: wp.float32,
    yaw_rate_min: wp.float32,
    yaw_rate_max: wp.float32,
    stand_prob: wp.float32,
    linvel_gain_min: wp.float32,
    linvel_gain_max: wp.float32,
    yaw_gain_min: wp.float32,
    yaw_gain_max: wp.float32,
    root_pos_w: wp.array(dtype=wp.vec3),
    root_yaw_quat: wp.array(dtype=wp.quat),
    mode: wp.array(dtype=wp.int32),
    cmd_linvel_b: wp.array(dtype=wp.vec3),
    cmd_yawvel_b: wp.array(dtype=wp.float32),
    cmd_eef_pos_b: wp.array(dtype=wp.vec3),
    cmd_eef_pos_w: wp.array(dtype=wp.vec3),
    cmd_eef_status: wp.array(dtype=wp.int32),
    standoff_pos_w: wp.array(dtype=wp.vec3),
    standoff_yaw_w: wp.array(dtype=wp.float32),
    world_linvel_gain: wp.array(dtype=wp.float32),
    world_yaw_gain: wp.array(dtype=wp.float32),
):
    """Sample loco + EEF commands for envs marked in ``resample``."""
    tid = wp.tid()
    if not resample[tid]:
        return

    seed_ = wp.rand_init(seed, tid)

    # Mode: 0 = world, 1 = body, 2 = nominal
    u = wp.randf(seed_, 0.0, 1.0)
    if u < mode_p0:
        mode[tid] = wp.int32(0)
    elif u < mode_p0 + mode_p1:
        mode[tid] = wp.int32(1)
    else:
        mode[tid] = wp.int32(2)

    m = mode[tid]
    if m == wp.int32(0):
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
    elif m == wp.int32(1):
        oz = wp.randf(seed_, eef_z_min, eef_z_max)
        ox = wp.randf(seed_, eef_x_min, eef_x_max)
        oy = wp.randf(seed_, eef_y_min, eef_y_max)
        cmd_eef_pos_b[tid] = wp.vec3(ox, oy, oz)
    # mode 2: leave targets alone; update_command fills from init_eef_pos_b

    # Independent loco only for body / nominal modes (mode 0 uses standoff loco).
    if m != wp.int32(0):
        vx = wp.randf(seed_, linvel_x_min, linvel_x_max)
        vy = wp.randf(seed_, linvel_y_min, linvel_y_max)
        speed = wp.sqrt(vx * vx + vy * vy)
        stand = wp.randf(seed_, 0.0, 1.0) < stand_prob
        if (speed < 0.1) or stand:
            cmd_linvel_b[tid] = wp.vec3(0.0, 0.0, 0.0)
            cmd_yawvel_b[tid] = 0.0
        else:
            cmd_linvel_b[tid] = wp.vec3(vx, vy, 0.0)
            cmd_yawvel_b[tid] = wp.randf(seed_, yaw_rate_min, yaw_rate_max)

    if wp.randf(seed_, 0.0, 1.0) < 0.5:
        cmd_eef_status[tid] = wp.int32(0)
    else:
        cmd_eef_status[tid] = wp.int32(1)


@wp.kernel(enable_backward=False)
def update_command(
    mode: wp.array(dtype=wp.int32),
    root_pos_w: wp.array(dtype=wp.vec3),
    root_yaw_quat: wp.array(dtype=wp.quat),
    heading_w: wp.array(dtype=wp.float32),
    cmd_eef_pos_w: wp.array(dtype=wp.vec3),
    standoff_pos_w: wp.array(dtype=wp.vec3),
    standoff_yaw_w: wp.array(dtype=wp.float32),
    init_eef_pos_b: wp.array(dtype=wp.vec3),
    world_linvel_gain: wp.array(dtype=wp.float32),
    world_yaw_gain: wp.array(dtype=wp.float32),
    linvel_x_min: wp.float32,
    linvel_x_max: wp.float32,
    linvel_y_min: wp.float32,
    linvel_y_max: wp.float32,
    yaw_rate_min: wp.float32,
    yaw_rate_max: wp.float32,
    cmd_eef_pos_b: wp.array(dtype=wp.vec3),
    cmd_linvel_b: wp.array(dtype=wp.vec3),
    cmd_yawvel_b: wp.array(dtype=wp.float32),
    base_pos_error: wp.array(dtype=wp.float32),
):
    """Refresh body-frame EEF command from mode and world / nominal targets."""
    tid = wp.tid()
    if mode[tid] == 0:
        cmd_eef_b, cmd_linvel, cmd_yawvel, base_err = update_world_command(
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
        cmd_eef_pos_b[tid] = cmd_eef_b
        cmd_linvel_b[tid] = cmd_linvel
        cmd_yawvel_b[tid] = cmd_yawvel
        base_pos_error[tid] = base_err
    elif mode[tid] == 1:
        # Body-frame goal already stored in cmd_eef_pos_b; leave as-is.
        base_pos_error[tid] = 0.0
    elif mode[tid] == 2:
        # Nominal rest pose; recompute from cached init_eef_pos_b (height may change).
        cmd_eef_pos_b[tid] = wp.vec3(
            init_eef_pos_b[tid][0],
            init_eef_pos_b[tid][1],
            root_pos_w[tid][2] + init_eef_pos_b[tid][2],
        )
        base_pos_error[tid] = 0.0


class LocoManipNew(CommandV2):
    """A refactored and simplified version of the LocoManip command manager.

    Orientation targets stay at the nominal body-frame rest pose for now
    (no sampled orientations). Position commands still follow the three modes.

    We will be reusing the reward terms defined in loco_manip.py.

    There are several frames of reference:
    1. World frame, with the origin being each env's origin.
    2. A special body frame, where Z is still the absolute height.
    3. Body frame, with the origin being root's transform.

    The command observation is given in frame 2.

    Three command modes:
    0. Sample a world-frame EEF goal and a standoff base pose short of it.
       Loco walks to the standoff; the EEF tracks the world goal.
    1. Sample a body-frame goal, which moves with the body and does not update
    2. Keep the nominal eef pos, which need to be recomputed based on the root height.

    Mode sampling uses a linear schedule from ``mode_probs_0`` to ``mode_probs_1``
    over ``mode_transition_steps`` env updates.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str = "arm_joint[7,8]",
        workspace_range: Sequence[Tuple[float, float]] = (
            (0.25, 0.75),
            (-0.2, 0.2),
            (0.2, 0.8),
        ),
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
        world_goal_radius_range: Tuple[float, float] = (1.5, 3.0),
        standoff_reach_range: Tuple[float, float] = (0.5, 0.7),
        world_linvel_gain_range: Tuple[float, float] = (1.0, 2.0),
        world_yaw_gain_range: Tuple[float, float] = (1.0, 2.0),
        mode_probs_0: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        mode_probs_1: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        mode_transition_steps: int = 2000,
        stand_prob: float = 0.1,
        resample_interval: int = 300,
        resample_prob: float = 0.75,
    ) -> None:
        super().__init__()
        if len(workspace_range) != 3:
            raise ValueError("workspace_range must have (x, y, z) intervals")

        self.eef_body_name = eef_body_name
        self.gripper_joint_names = gripper_joint_names
        self.workspace_range = tuple(tuple(r) for r in workspace_range)
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.world_goal_radius_range = world_goal_radius_range
        self.standoff_reach_range = standoff_reach_range
        self.world_linvel_gain_range = world_linvel_gain_range
        self.world_yaw_gain_range = world_yaw_gain_range
        self.mode_probs_0 = torch.tensor(mode_probs_0)
        self.mode_probs_1 = torch.tensor(mode_probs_1)
        assert self.mode_probs_0.sum() > 0.0
        assert self.mode_probs_1.sum() > 0.0
        self.mode_probs_0 = self.mode_probs_0 / self.mode_probs_0.sum()
        self.mode_probs_1 = self.mode_probs_1 / self.mode_probs_1.sum()
        self.mode_transition_steps = max(int(mode_transition_steps), 0)
        self.stand_prob = stand_prob
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

    @property
    def mode_probs(self) -> torch.Tensor:
        """Linear schedule from ``mode_probs_0`` → ``mode_probs_1`` over env steps."""
        if not self.env.training:
            return self.mode_probs_1
        if self.mode_transition_steps <= 0:
            return self.mode_probs_1
        t = min(1.0, self._mode_schedule_step / self.mode_transition_steps)
        return torch.lerp(self.mode_probs_0, self.mode_probs_1, t)
    
    def command(self, key: str):
        # Dense layout (23D), orientation fixed at nominal body-frame rest pose:
        # [v_x, v_y, yaw_rate, eef_xyz, pos_diff, fwd, fwd_diff, up, up_diff, closed, open]
        if key == "teacher":
            result = torch.cat(
                [
                    self.cmd_linvel_b[:, :2],
                    self.cmd_yawvel_b,
                    self.cmd_eef_pos_b,
                    self.pos_diff_b,
                    self.cmd_eef_forward_b,
                    self.forward_diff_b,
                    self.cmd_eef_upward_b,
                    self.upward_diff_b,
                    self.cmd_eef_status.float(),
                    1.0 - self.cmd_eef_status.float(),
                ],
                dim=-1,
            )
        elif key == "student": # eef command only
            result = torch.cat([
                self.cmd_eef_pos_b, # [N, 3]
                self.pos_diff_b, # [N, 3]
                self.cmd_eef_forward_b, # [N, 3]
                self.forward_diff_b, # [N, 3]
                self.cmd_eef_upward_b, # [N, 3]
                self.upward_diff_b, # [N, 3]
                self.cmd_eef_status.float(), # [N, 1]
                (1 - self.cmd_eef_status.float()) # [N, 1]
            ], dim=-1)
        else:
            raise ValueError(f"Invalid key: {key}")
        return result

    def symmetry_transform(self, key: str = "teacher"):
        # Mirror left-right: flip y and yaw for all body-frame vectors.
        # Layout must match ``command(key)``.
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        forward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_upward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        upward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        eef_status = SymmetryTransform(perm=[0, 1], signs=[1, 1])
        eef_block = [
            cmd_eef_pos_b,
            pos_diff_b,
            cmd_eef_forward_b,
            forward_diff_b,
            cmd_eef_upward_b,
            upward_diff_b,
            eef_status,
        ]
        if key == "teacher":
            cmd_linvel_b = SymmetryTransform(perm=[0, 1], signs=[1, -1])
            cmd_yawvel_b = SymmetryTransform(perm=[0], signs=[-1])
            return SymmetryTransform.cat([cmd_linvel_b, cmd_yawvel_b, *eef_block])
        if key == "student":
            return SymmetryTransform.cat(eef_block)
        raise ValueError(f"Invalid key: {key}")

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        resample[env_ids] = True
        # always start in nominal mode
        self.sample_commands_warp(resample, [0.0, 0.0, 1.0])

        self.cmd_eef_rot_b[env_ids] = self.init_eef_rot_b[env_ids]
        self.base_pos_error[env_ids] = 0.0
        self.pos_diff_w[env_ids] = 0.0
        self.pos_diff_b[env_ids] = 0.0
        self.pos_error_norm2[env_ids] = 0.0
        self.pos_error_norm[env_ids] = 0.0
        self.forward_diff_w[env_ids] = 0.0
        self.forward_diff_b[env_ids] = 0.0
        self.upward_diff_w[env_ids] = 0.0
        self.upward_diff_b[env_ids] = 0.0

        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0

    @override
    def _initialize(self, env: _EnvBase) -> None:
        super()._initialize(env)
        self._mode_schedule_step = 0
        self.mode_probs_0 = self.mode_probs_0.to(self.device)
        self.mode_probs_1 = self.mode_probs_1.to(self.device)

        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, "
                f"got {len(body_ids)}"
            )
        self.eef_body_idx = int(body_ids[0])

        self.gripper_joint_ids, _ = self.asset.find_joints(self.gripper_joint_names)
        self.gripper_joint_ids = torch.as_tensor(
            self.gripper_joint_ids, device=self.device, dtype=torch.long
        )
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        with torch.device(self.device):
            # 0 = world goal, 1 = body goal, 2 = nominal rest pose
            self.mode = torch.zeros(self.num_envs, dtype=torch.int32)

            # Loco (body yaw frame)
            self.next_command_linvel = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_traveled = torch.zeros(self.num_envs, 1)

            # EEF position: observation / tracking use yaw-aligned frame (frame 2)
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)

            # EEF orientation (nominal body/yaw-frame rest pose; not resampled)
            self.cmd_eef_rot_b = torch.zeros(self.num_envs, 4)
            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_b = torch.zeros(self.num_envs, 3)
            self.forward_diff_w = torch.zeros(self.num_envs, 3)
            self.forward_diff_b = torch.zeros(self.num_envs, 3)
            self.upward_diff_w = torch.zeros(self.num_envs, 3)
            self.upward_diff_b = torch.zeros(self.num_envs, 3)

            # Gripper open/close (0 = open, 1 = closed) for eef_grasp reuse
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.int32)

            self.base_pos_error = torch.zeros(self.num_envs, 1)
            self.world_linvel_gain = torch.ones(self.num_envs)
            self.world_yaw_gain = torch.ones(self.num_envs)
            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs)

        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.init_eef_pos_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w,
            self.eef_pos_w - self.asset.data.root_link_pos_w,
        )  # z is wrt root
        # Nominal EEF orientation in yaw-aligned body frame (frame 2).
        self.init_eef_rot_b = quat_mul(
            quat_conjugate(root_yaw_q),
            self.eef_quat_w,
        )

        # Mode 2 starts at the nominal pose; other modes overwrite on resample.
        self.cmd_eef_pos_b[:] = self.init_eef_pos_b
        self.cmd_eef_rot_b[:] = self.init_eef_rot_b
        self.mode[:] = 2

        self._wp_device = wp.get_device(str(self.device))
        self._warp_seed = 0

        self.eef_pose_marker = None
        self._cmd_linvel_lines = None
        self._eef_target_lines = None
        if self.env.sim.has_gui():
            if self.env.backend == "isaac":
                from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

                self.scene: IsaacSceneAdapter = self.env.scene
                self.eef_pose_marker = self.scene.create_frame_marker(
                    "/Visuals/Command/target_eef_pose",
                    scale=(0.1, 0.1, 0.1),
                )
            elif self.env.backend == "mjlab":
                from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer

                self.eef_pose_marker = self.env.scene.create_frame_marker(
                    "target_eef_pose",
                    scale=(0.1, 0.1, 0.1),
                )
                viewer: MjLabViewer = self.env.sim.viewer
                self._cmd_linvel_lines = viewer.add_line_segments(
                    "cmd_linvel_w", (1.0, 0.0, 0.0)
                )
                self._cmd_linvel_lines.line_width = 2.0
                self._eef_target_lines = viewer.add_line_segments(
                    "cmd_eef_pos_w", (0.0, 0.0, 1.0)
                )
                self._eef_target_lines.line_width = 2.0

    @property
    def eef_pos_w(self) -> torch.Tensor:
        return self.asset.data.body_link_pos_w[:, self.eef_body_idx]

    @property
    def eef_quat_w(self) -> torch.Tensor:
        return self.asset.data.body_link_quat_w[:, self.eef_body_idx]

    def get_gripper_status(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed)."""
        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        return 1.0 - openness

    def sample_commands_warp(self, resample: torch.Tensor, mode_probs: torch.Tensor) -> None:
        self._warp_seed = (self._warp_seed + 1) % (2**31 - 1)
        root_pos_w = self.asset.data.root_link_pos_w.contiguous()
        root_yaw_xyzw = quat_wxyz_to_xyzw(
            yaw_quat(self.asset.data.root_link_quat_w)
        )
        wx, wy, wz = self.workspace_range
        mode_p0, mode_p1, _mode_p2 = mode_probs
        wp.launch(
            kernel=sample_commands,
            dim=[self.num_envs],
            inputs=[
                wp.from_torch(resample, dtype=wp.bool, return_ctype=True),
                self._warp_seed,
                mode_p0,
                mode_p1,
                wx[0],
                wx[1],
                wy[0],
                wy[1],
                wz[0],
                wz[1],
                self.world_goal_radius_range[0],
                self.world_goal_radius_range[1],
                self.standoff_reach_range[0],
                self.standoff_reach_range[1],
                self.linvel_x_range[0],
                self.linvel_x_range[1],
                self.linvel_y_range[0],
                self.linvel_y_range[1],
                self.yaw_rate_range[0],
                self.yaw_rate_range[1],
                self.stand_prob,
                self.world_linvel_gain_range[0],
                self.world_linvel_gain_range[1],
                self.world_yaw_gain_range[0],
                self.world_yaw_gain_range[1],
                wp.from_torch(root_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(root_yaw_xyzw, dtype=wp.quat, return_ctype=True),
            ],
            outputs=[
                wp.from_torch(self.mode, dtype=wp.int32, return_ctype=True),
                wp.from_torch(self.cmd_linvel_b, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(
                    self.cmd_yawvel_b[:, 0], dtype=wp.float32, return_ctype=True
                ),
                wp.from_torch(self.cmd_eef_pos_b, dtype=wp.vec3, return_ctype=True),
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

    @override
    def sync_state(self) -> None:
        # Tracking / standing for rewards at THIS step.
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.is_standing_env = (self.command_speed < 0.1) & (
            self.cmd_yawvel_b.abs() < 0.1
        )
        self.current_speed = self.asset.data.root_link_lin_vel_w.norm(
            dim=-1, keepdim=True
        )
        self.distance_commanded = (
            self.distance_commanded + self.command_speed * self.env.step_dt
        )
        self.distance_traveled = (
            self.distance_traveled + self.current_speed * self.env.step_dt
        )

        root_pos_w = self.asset.data.root_link_pos_w
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)

        # World EEF target from frame-2 cmd (mode 0 uses stored world goal equivalently).
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

        self.eef_status = self.get_gripper_status()

    @override
    def update(self) -> None: # for readability, do not extract methods
        self._mode_schedule_step += 1
        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample = interval & (
            torch.rand(self.num_envs, device=self.device) < self.resample_prob
        )
        self.sample_commands_warp(resample, self.mode_probs)

        root_pos_w = self.asset.data.root_link_pos_w.contiguous()
        root_yaw_xyzw = quat_wxyz_to_xyzw(
            yaw_quat(self.asset.data.root_link_quat_w)
        )
        heading_w = self.asset.data.heading_w.contiguous()

        wp.launch(
            kernel=update_command,
            dim=[self.num_envs],
            inputs=[
                wp.from_torch(self.mode, dtype=wp.int32, return_ctype=True),
                wp.from_torch(root_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(root_yaw_xyzw, dtype=wp.quat, return_ctype=True),
                wp.from_torch(heading_w, dtype=wp.float32, return_ctype=True),
                wp.from_torch(self.cmd_eef_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self.standoff_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(
                    self.standoff_yaw_w, dtype=wp.float32, return_ctype=True
                ),
                wp.from_torch(self.init_eef_pos_b, dtype=wp.vec3, return_ctype=True),
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
                wp.from_torch(self.cmd_linvel_b, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(
                    self.cmd_yawvel_b[:, 0], dtype=wp.float32, return_ctype=True
                ),
                wp.from_torch(
                    self.base_pos_error[:, 0], dtype=wp.float32, return_ctype=True
                ),
            ],
            device=self._wp_device,
        )

        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.cmd_linvel_w = quat_rotate(root_yaw_q, self.cmd_linvel_b)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        # Orientation stays nominal; refreshed in sync_state for the next reward pass.
        self.cmd_eef_rot_b[:] = self.init_eef_rot_b

    @override
    def debug_draw(self) -> None: # for readability, do not extract methods
        # Reconstruct world target from frame-2 cmd_eef_pos_b (works for all modes).
        root_pos_w = self.asset.data.root_link_pos_w
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        offset_xy = self.cmd_eef_pos_b.clone()
        offset_xy[:, 2] = 0.0
        cmd_eef_pos_w = root_pos_w * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        ) + quat_rotate(root_yaw_q, offset_xy)
        cmd_eef_pos_w[:, 2] = self.cmd_eef_pos_b[:, 2]
        cmd_eef_rot_w = quat_mul(root_yaw_q, self.cmd_eef_rot_b)

        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                root_pos_w,
                self.cmd_linvel_w,
                color=(1.0, 1.0, 1.0, 1.0),
            )
            self.env.debug_draw.vector(
                self.eef_pos_w,
                cmd_eef_pos_w - self.eef_pos_w,
                color=(0.0, 0.0, 1.0, 1.0),
            )
        elif self.env.backend == "mjlab" and self._cmd_linvel_lines is not None:
            start = root_pos_w + torch.tensor(
                [0.0, 0.0, 0.2], device=self.device
            )
            self._cmd_linvel_lines.points = torch.stack(
                [start, start + self.cmd_linvel_w], 1
            ).cpu()
            self._eef_target_lines.points = torch.stack(
                [self.eef_pos_w, cmd_eef_pos_w], 1
            ).cpu()

        if self.eef_pose_marker is not None:
            self.eef_pose_marker.visualize(
                translations=cmd_eef_pos_w,
                orientations=cmd_eef_rot_w,
            )

    class CommandState(TensorClass):
        """Snapshot for offline command / reward relabeling.

        Saved after ``sync_state`` so tracking fields match the reward pass.
        """

        mode: torch.Tensor  # [N] int32: 0=world, 1=body, 2=nominal
        root_pose_w: torch.Tensor  # [N, 7] pos + quat (wxyz)
        eef_pos_w: torch.Tensor  # [N, 3]
        eef_quat_w: torch.Tensor  # [N, 4]
        eef_state_w: torch.Tensor  # [N, 13] link state for angvel relabel
        cmd_eef_pos_w: torch.Tensor  # [N, 3] world goal (mode 0) or reconstructed
        cmd_eef_pos_b: torch.Tensor  # [N, 3]
        cmd_eef_rot_w: torch.Tensor  # [N, 4]
        cmd_eef_status: torch.Tensor  # [N, 1]
        eef_status: torch.Tensor  # [N, 1]
        base_pos_error: torch.Tensor  # [N, 1]

    def get_state(self) -> CommandState:
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        if hasattr(self.asset.data, "root_link_pose_w"):
            root_pose_w = self.asset.data.root_link_pose_w
        else:
            root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)

        eef_pos_w = self.eef_pos_w
        eef_quat_w = self.eef_quat_w
        if hasattr(self.asset.data, "body_link_state_w"):
            eef_state_w = self.asset.data.body_link_state_w[:, self.eef_body_idx]
        else:
            # pos(3) + quat(4) + linvel(3) + angvel(3)
            linvel = self.asset.data.body_link_lin_vel_w[:, self.eef_body_idx]
            angvel = self.asset.data.body_link_ang_vel_w[:, self.eef_body_idx]
            eef_state_w = torch.cat(
                [eef_pos_w, eef_quat_w, linvel, angvel], dim=-1
            )

        # Reconstruct world EEF target from heading-frame cmd (valid for all modes).
        # Mode 0 also keeps the persistent world goal in ``cmd_eef_pos_w``.
        root_yaw_q = yaw_quat(root_quat_w)
        offset_xy = self.cmd_eef_pos_b.clone()
        offset_xy[:, 2] = 0.0
        track_eef_pos_w = root_pos_w * torch.tensor(
            [1.0, 1.0, 0.0], device=self.device
        ) + quat_rotate(root_yaw_q, offset_xy)
        track_eef_pos_w[:, 2] = self.cmd_eef_pos_b[:, 2]
        cmd_eef_pos_w = torch.where(
            (self.mode == 0).unsqueeze(-1),
            self.cmd_eef_pos_w,
            track_eef_pos_w,
        )

        return self.CommandState(
            mode=self.mode.clone(),
            root_pose_w=root_pose_w.clone(),
            eef_pos_w=eef_pos_w.clone(),
            eef_quat_w=eef_quat_w.clone(),
            eef_state_w=eef_state_w.clone(),
            cmd_eef_pos_w=cmd_eef_pos_w.clone(),
            cmd_eef_pos_b=self.cmd_eef_pos_b.clone(),
            cmd_eef_rot_w=self.cmd_eef_rot_w.clone(),
            cmd_eef_status=self.cmd_eef_status.clone(),
            eef_status=self.eef_status.clone(),
            base_pos_error=self.base_pos_error.clone(),
            batch_size=[self.num_envs],
            device=self.device,
        )
