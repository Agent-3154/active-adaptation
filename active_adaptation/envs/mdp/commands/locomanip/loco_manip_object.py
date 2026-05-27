from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    clamp_norm,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    sample_quat_yaw,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import Command

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject


class LocoManipObject(Command):
    """Scripted FSM command for picking up an object and transporting it to a target.

    Outputs the same ``command(key="dense"|"sparse")`` interface as
    ``SingleEEFLocoManip`` (17D dense / 14D sparse, body/yaw frame) so a
    pre-trained policy can execute the full manipulation sequence without
    retraining.

    FSM states (per env):
        APPROACH   – drive base to approach standoff; EEF pre-positions above
                     grasp point; gripper open.
        GRASP_POSE – base holds at standoff; EEF descends to exact grasp point;
                     gripper open.
        CLOSE      – hold position; command gripper closed; wait for closure.
        LIFT       – raise EEF to lift height above grasp point; gripper closed.
        MOVE       – drive base to target standoff; EEF tracks lifted goal;
                     gripper closed.
        RELEASE    – hold at target standoff; lower EEF; command gripper open;
                     wait for opening.
        BACKUP     – drive base back to approach standoff; gripper open.
                     Transitions back to APPROACH to loop within the episode.
    """

    APPROACH   = 0
    GRASP_POSE = 1
    CLOSE      = 2
    LIFT       = 3
    MOVE       = 4
    RELEASE    = 5
    BACKUP     = 6

    supported_backends = ("isaac",)

    def __init__(
        self,
        env,
        eef_body_name: str,
        gripper_joint_names: str,
        object_name: str = "object",
        object_distance_range: Tuple[float, float] = (2.0, 3.0),
        target_distance_range: Tuple[float, float] = (2.0, 3.0),
        grasp_height_range: Tuple[float, float] = (0.05, 0.5),
        pre_grasp_height_offset: float = 0.25,
        lift_height: float = 0.35,
        standoff_distance_range: Tuple[float, float] = (0.5, 0.8),
        standoff_angle_range: Tuple[float, float] = (-torch.pi / 3, torch.pi / 3),
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        standoff_linvel_gain: float = 2.0,
        standoff_yaw_gain: float = 1.0,
        speed_limit: float = 0.8,
        eef_pos_threshold: float = 0.05,
        base_pos_threshold: float = 0.2,
        yaw_threshold: float = 0.15,
        gripper_close_threshold: float = 0.7,
        gripper_open_threshold: float = 0.3,
        teleop: bool = False,
    ) -> None:
        super().__init__(env, teleop)

        body_ids, _ = self.asset.find_bodies(eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {eef_body_name!r}, got {len(body_ids)}"
            )
        self.eef_body_idx = body_ids[0]

        joint_ids, _ = self.asset.find_joints(gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(joint_ids, device=self.device)
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        self.object_name = object_name
        self.object: RigidObject = self.env.scene[object_name]
        self.object_init_root_state = self.object.data.default_root_state.clone()

        self.object_distance_range = object_distance_range
        self.target_distance_range = target_distance_range
        self.grasp_height_range = grasp_height_range
        self.pre_grasp_height_offset = pre_grasp_height_offset
        self.lift_height = lift_height
        self.standoff_distance_range = standoff_distance_range
        self.standoff_angle_range = standoff_angle_range
        self.yaw_rate_range = yaw_rate_range
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain
        self.speed_limit = speed_limit
        self.eef_pos_threshold = eef_pos_threshold
        self.base_pos_threshold = base_pos_threshold
        self.yaw_threshold = yaw_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_open_threshold = gripper_open_threshold

        with torch.device(self.device):
            # FSM state per env
            self.state = torch.zeros(self.num_envs, dtype=torch.long)

            # Sampled scene layout (set in sample_init / sample_commands)
            self.grasp_height_per_env = torch.zeros(self.num_envs)
            self.grasp_point_w = torch.zeros(self.num_envs, 3)
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.approach_standoff_w = torch.zeros(self.num_envs, 3)
            self.approach_yaw_w = torch.zeros(self.num_envs)
            self.target_standoff_w = torch.zeros(self.num_envs, 3)
            self.target_yaw_w = torch.zeros(self.num_envs)

            # Command tensors – match SingleEEFLocoManip field names exactly
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_w = torch.zeros(self.num_envs, 3)

            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)

            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)

            # Gripper: eef_status = continuous closedness [0,1]; cmd = {0,1}
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.long)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        self.grasp_marker = None
        self.target_marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.grasp_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_grasp_point", (1.0, 0.4, 0.0), radius=0.03
            )
            self.target_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_target_pos", (0.0, 0.4, 1.0), radius=0.05
            )

    # ------------------------------------------------------------------ #
    # Command / symmetry interface (matches SingleEEFLocoManip exactly)   #
    # ------------------------------------------------------------------ #

    def command(self, key: str = "dense") -> torch.Tensor:
        if key == "dense":
            return torch.cat([
                self.cmd_linvel_b[:, :2],                    # 2
                self.cmd_yawvel_b,                           # 1
                self.cmd_eef_pos_b,                          # 3
                self.pos_diff_b,                             # 3
                self.cmd_eef_forward_b,                      # 3
                self.cmd_eef_forward_b - self.eef_forward_b, # 3
                self.cmd_eef_status.float(),                 # 1
                (1 - self.cmd_eef_status).float(),           # 1
            ], dim=-1)
        elif key == "sparse":
            return torch.cat([
                self.cmd_eef_pos_b,                          # 3
                self.pos_diff_b,                             # 3
                self.cmd_eef_forward_b,                      # 3
                self.cmd_eef_forward_b - self.eef_forward_b, # 3
                self.cmd_eef_status.float(),                 # 1
                (1 - self.cmd_eef_status).float(),           # 1
            ], dim=-1)
        else:
            raise ValueError(f"Invalid key: {key}")

    @override
    def symmetry_transform(self, key: str = "dense"):
        if key == "dense":
            cmd_linvel_b     = SymmetryTransform(perm=[0, 1],    signs=[1, -1])
            cmd_yawvel_b     = SymmetryTransform(perm=[0],       signs=[-1])
            cmd_eef_pos_b    = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            fwd_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status       = SymmetryTransform(perm=[0, 1],    signs=[1,  1])
            return SymmetryTransform.cat([
                cmd_linvel_b, cmd_yawvel_b, cmd_eef_pos_b, pos_diff_b,
                cmd_eef_forward_b, fwd_diff_b, eef_status,
            ])
        elif key == "sparse":
            cmd_eef_pos_b    = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            fwd_diff_b       = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status       = SymmetryTransform(perm=[0, 1],    signs=[1,  1])
            return SymmetryTransform.cat([
                cmd_eef_pos_b, pos_diff_b, cmd_eef_forward_b, fwd_diff_b, eef_status,
            ])
        else:
            raise ValueError(f"Invalid key: {key}")

    # ------------------------------------------------------------------ #
    # Init / sampling                                                      #
    # ------------------------------------------------------------------ #

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        lo, hi = value_range
        return torch.rand(num_samples, device=self.device) * (hi - lo) + lo

    def _sample_standoff(
        self,
        ref_pos_w: torch.Tensor,
        face_dir_w: torch.Tensor,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (standoff_pos_w, yaw) on a random arc around ref_pos,
        with the yaw pointing back toward ref_pos."""
        base_yaw = torch.atan2(face_dir_w[:, 1], face_dir_w[:, 0])
        angle = base_yaw + self._sample_uniform(n, self.standoff_angle_range)
        dist  = self._sample_uniform(n, self.standoff_distance_range)
        standoff = ref_pos_w.clone()
        standoff[:, 0] = ref_pos_w[:, 0] + dist * torch.cos(angle)
        standoff[:, 1] = ref_pos_w[:, 1] + dist * torch.sin(angle)
        standoff[:, 2] = self.env.get_ground_height_at(standoff)
        yaw = torch.atan2(
            ref_pos_w[:, 1] - standoff[:, 1],
            ref_pos_w[:, 0] - standoff[:, 0],
        )
        return standoff, yaw

    def _init_standoffs(
        self,
        env_ids: torch.Tensor,
        obj_pos_w: torch.Tensor,
        tgt_pos_w: torch.Tensor,
        origins: torch.Tensor,
    ) -> None:
        n = len(env_ids)
        approach_standoff, approach_yaw = self._sample_standoff(
            obj_pos_w, origins - obj_pos_w, n
        )
        self.approach_standoff_w[env_ids] = approach_standoff
        self.approach_yaw_w[env_ids] = approach_yaw

        target_standoff, target_yaw = self._sample_standoff(
            tgt_pos_w, origins - tgt_pos_w, n
        )
        self.target_standoff_w[env_ids] = target_standoff
        self.target_yaw_w[env_ids] = target_yaw

    @override
    def sample_init(self, env_ids: torch.Tensor) -> dict:
        origins = self.env.scene.get_spawn_origins(env_ids)
        n = len(env_ids)

        robot_init = self.init_root_state[env_ids].clone()
        robot_init[:, :2] += origins[:, :2]
        robot_init[:, 3:7] = quat_mul(
            robot_init[:, 3:7], sample_quat_yaw(n, device=self.device)
        )

        object_init = self.object_init_root_state[env_ids].clone()
        obj_angle = torch.rand(n, device=self.device) * 2 * torch.pi
        obj_dist  = self._sample_uniform(n, self.object_distance_range)
        object_init[:, 0] = origins[:, 0] + obj_dist * torch.cos(obj_angle)
        object_init[:, 1] = origins[:, 1] + obj_dist * torch.sin(obj_angle)
        object_init[:, 2] += self.env.get_ground_height_at(object_init[:, :3])
        object_init[:, 3:7] = sample_quat_yaw(n, device=self.device)
        if object_init.shape[-1] > 7:
            object_init[:, 7:] = 0.0

        # Target: a different direction from the origin
        tgt_angle = obj_angle + torch.pi / 2 + (torch.rand(n, device=self.device) - 0.5) * torch.pi
        tgt_dist  = self._sample_uniform(n, self.target_distance_range)
        target = origins.clone()
        target[:, 0] = origins[:, 0] + tgt_dist * torch.cos(tgt_angle)
        target[:, 1] = origins[:, 1] + tgt_dist * torch.sin(tgt_angle)
        target[:, 2] = self.env.get_ground_height_at(target)
        self.target_pos_w[env_ids] = target

        self.grasp_height_per_env[env_ids] = self._sample_uniform(n, self.grasp_height_range)
        self._init_standoffs(env_ids, object_init[:, :3], target, origins)
        self.state[env_ids] = self.APPROACH

        return {"robot": robot_init, self.object_name: object_init}

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        """Mid-episode resample: re-use existing object/target positions,
        re-sample standoffs and restart from APPROACH."""
        origins = self.env.scene.env_origins[env_ids]
        obj_pos  = self.object.data.root_pos_w[env_ids]
        tgt_pos  = self.target_pos_w[env_ids]
        self.grasp_height_per_env[env_ids] = self._sample_uniform(
            len(env_ids), self.grasp_height_range
        )
        self._init_standoffs(env_ids, obj_pos, tgt_pos, origins)
        self.state[env_ids] = self.APPROACH

    # ------------------------------------------------------------------ #
    # Per-step helpers                                                     #
    # ------------------------------------------------------------------ #

    def _update_eef_state(self) -> None:
        """Refresh EEF and gripper state tensors."""
        yaw_q   = yaw_quat(self.asset.data.root_link_quat_w)
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.eef_body_idx]
        forward_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        self.eef_pos_w   = self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        self.eef_forward_w = quat_rotate(eef_quat_w, forward_axis)
        self.eef_forward_b = quat_rotate_inverse(yaw_q, self.eef_forward_w)

        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        self.eef_status = 1.0 - openness

        self.pos_diff_w  = self.cmd_eef_pos_w - self.eef_pos_w
        self.pos_diff_b  = quat_rotate_inverse(yaw_q, self.pos_diff_w)
        self.pos_error_norm2 = self.pos_diff_w.square().sum(dim=-1, keepdim=True)
        self.pos_error_norm  = self.pos_error_norm2.sqrt()
        reached_now = self.pos_error_norm < self.eef_pos_threshold
        self.eef_pos_reaching = reached_now & (~self.eef_pos_reached)
        self.eef_pos_reached  = self.eef_pos_reached | reached_now

        self.command_speed   = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = self.command_speed < 0.1

    def _drive_base(
        self,
        ids: torch.Tensor,
        standoff_w: torch.Tensor,
        yaw_w: torch.Tensor,
    ) -> None:
        root_pos  = self.asset.data.root_link_pos_w[ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[ids])
        delta_w   = standoff_w - root_pos
        delta_w[:, 2] = 0.0
        linvel_w  = clamp_norm(
            self.standoff_linvel_gain * delta_w, max=self.speed_limit
        )
        self.cmd_linvel_w[ids] = linvel_w
        self.cmd_linvel_b[ids] = quat_rotate_inverse(root_yaw_q, linvel_w)
        yaw_err = wrap_to_pi(yaw_w - self.asset.data.heading_w[ids])
        self.cmd_yawvel_b[ids, 0] = (
            self.standoff_yaw_gain * yaw_err
        ).clamp(*self.yaw_rate_range)

    def _set_eef_target(
        self,
        ids: torch.Tensor,
        target_w: torch.Tensor,
        forward_w: torch.Tensor,
        gripper_closed: bool,
    ) -> None:
        root_pos  = self.asset.data.root_link_pos_w[ids]
        yaw_q     = yaw_quat(self.asset.data.root_link_quat_w[ids])
        self.cmd_eef_pos_w[ids] = target_w
        delta_flat = (target_w - root_pos).clone()
        delta_flat[:, 2] = 0.0
        delta_b = quat_rotate_inverse(yaw_q, delta_flat)
        self.cmd_eef_pos_b[ids, :2] = delta_b[:, :2]
        self.cmd_eef_pos_b[ids, 2]  = (
            target_w[:, 2] - self.env.get_ground_height_at(target_w)
        )
        fwd = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.cmd_eef_forward_w[ids] = fwd
        self.cmd_eef_forward_b[ids] = quat_rotate_inverse(yaw_q, fwd)
        self.cmd_eef_status[ids, 0] = int(gripper_closed)

    # ------------------------------------------------------------------ #
    # FSM                                                                  #
    # ------------------------------------------------------------------ #

    def _transition(self) -> None:
        """Per-env state advancement based on current sensor readings."""
        root_pos = self.asset.data.root_link_pos_w
        heading  = self.asset.data.heading_w

        approach_dist = (self.approach_standoff_w - root_pos)[:, :2].norm(dim=-1)
        approach_yaw_err = wrap_to_pi(self.approach_yaw_w - heading).abs()
        target_dist  = (self.target_standoff_w - root_pos)[:, :2].norm(dim=-1)

        eef_close      = self.pos_error_norm.squeeze(-1) < self.eef_pos_threshold
        at_approach    = (approach_dist < self.base_pos_threshold) & (approach_yaw_err < self.yaw_threshold)
        at_target      = target_dist < self.base_pos_threshold
        gripper_closed = self.eef_status.squeeze(-1) > self.gripper_close_threshold
        gripper_open   = self.eef_status.squeeze(-1) < self.gripper_open_threshold

        s = self.state
        s = torch.where((s == self.APPROACH)   & at_approach,  torch.full_like(s, self.GRASP_POSE), s)
        s = torch.where((s == self.GRASP_POSE) & eef_close,    torch.full_like(s, self.CLOSE),      s)
        s = torch.where((s == self.CLOSE)      & gripper_closed, torch.full_like(s, self.LIFT),     s)
        s = torch.where((s == self.LIFT)       & eef_close,    torch.full_like(s, self.MOVE),       s)
        s = torch.where((s == self.MOVE)       & at_target,    torch.full_like(s, self.RELEASE),    s)
        s = torch.where((s == self.RELEASE)    & gripper_open, torch.full_like(s, self.BACKUP),     s)
        s = torch.where((s == self.BACKUP)     & at_approach,  torch.full_like(s, self.APPROACH),   s)
        self.state = s

    def _apply_commands(self) -> None:
        """Compute grasp point and set base/EEF commands per FSM state."""
        obj_pos_w  = self.object.data.root_pos_w
        obj_quat_w = self.object.data.root_quat_w
        grasp_off  = torch.zeros(self.num_envs, 3, device=self.device)
        grasp_off[:, 2] = self.grasp_height_per_env
        self.grasp_point_w = quat_rotate(obj_quat_w, grasp_off) + obj_pos_w

        # Default: hold (updated per state below)
        self.cmd_linvel_b.zero_()
        self.cmd_linvel_w.zero_()
        self.cmd_yawvel_b.zero_()

        for st in range(self.BACKUP + 1):
            ids = (self.state == st).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue

            gp  = self.grasp_point_w[ids]
            tgt = self.target_pos_w[ids]
            app_st = self.approach_standoff_w[ids]
            app_yw = self.approach_yaw_w[ids]
            tgt_st = self.target_standoff_w[ids]
            tgt_yw = self.target_yaw_w[ids]

            if st == self.APPROACH:
                self._drive_base(ids, app_st, app_yw)
                eef_tgt = gp.clone()
                eef_tgt[:, 2] += self.pre_grasp_height_offset
                fwd = gp - app_st
                self._set_eef_target(ids, eef_tgt, fwd, gripper_closed=False)

            elif st == self.GRASP_POSE:
                self._drive_base(ids, app_st, app_yw)
                fwd = gp - app_st
                self._set_eef_target(ids, gp, fwd, gripper_closed=False)

            elif st == self.CLOSE:
                self._drive_base(ids, app_st, app_yw)
                fwd = gp - app_st
                self._set_eef_target(ids, gp, fwd, gripper_closed=True)

            elif st == self.LIFT:
                self._drive_base(ids, app_st, app_yw)
                lift_tgt = gp.clone()
                lift_tgt[:, 2] += self.lift_height
                fwd = lift_tgt - app_st
                self._set_eef_target(ids, lift_tgt, fwd, gripper_closed=True)

            elif st == self.MOVE:
                self._drive_base(ids, tgt_st, tgt_yw)
                lift_tgt = tgt.clone()
                lift_tgt[:, 2] += self.lift_height
                fwd = lift_tgt - tgt_st
                self._set_eef_target(ids, lift_tgt, fwd, gripper_closed=True)

            elif st == self.RELEASE:
                self._drive_base(ids, tgt_st, tgt_yw)
                fwd = tgt - tgt_st
                self._set_eef_target(ids, tgt, fwd, gripper_closed=False)

            elif st == self.BACKUP:
                self._drive_base(ids, app_st, app_yw)
                safe = app_st.clone()
                safe[:, 2] += self.pre_grasp_height_offset
                fwd = gp - app_st
                self._set_eef_target(ids, safe, fwd, gripper_closed=False)

    # ------------------------------------------------------------------ #
    # Overrides                                                            #
    # ------------------------------------------------------------------ #

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        self.eef_pos_reached[env_ids]  = False
        self.eef_pos_reaching[env_ids] = False

    @override
    def update(self) -> None:
        self._update_eef_state()
        self._transition()
        self._apply_commands()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w, self.eef_forward_w, color=(1.0, 0.0, 0.0, 1.0)
        )
        self.env.debug_draw.vector(
            self.eef_pos_w, self.cmd_eef_forward_w, color=(0.0, 1.0, 0.0, 1.0)
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        if self.grasp_marker is not None:
            self.grasp_marker.visualize(self.grasp_point_w)
        if self.target_marker is not None:
            self.target_marker.visualize(self.target_pos_w)


__all__ = ["LocoManipObject"]
