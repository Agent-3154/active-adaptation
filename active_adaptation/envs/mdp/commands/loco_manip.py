"""Locomotion + single end-effector position commands (scaffold)."""

from __future__ import annotations

from typing import Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    euler_from_quat,
    euler_rotate,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from .base import Command
from ..rewards.base import Reward


class SingleEEFLocoManip(Command):
    """Command vector: base velocity, yaw rate, EEF target, and EEF pitch target.

    Layout: ``[v_x, v_y, yaw_rate, eef_x, eef_y, eef_z, eef_pitch]`` (7D). The first
    two loco components are in the usual body horizontal frame (same as ``Twist``);
    ``eef_x``/``eef_y`` are **not** full body frame: they use the same **yaw-only**
    rotation as world ``(x,y)`` offsets from the root (pitch/roll of the base are
    ignored for the horizontal part). ``eef_z`` is **height above terrain**, not
    root-link ``z``: world target height is
    ``get_ground_height_at(query_xy) + eef_z``, with ``query_xy`` the horizontal target under
    the root. ``eef_pitch`` is a global pitch target used to construct a world-frame
    EEF forward vector.
    """

    def __init__(
        self,
        env,
        eef_body_name: str,
        workspace_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        | None = None,
        workspace_profile: str | None = None,
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        world_goal_prob: float = 0.5,
        standoff_distance_range: Tuple[float, float] = (1.0, 2.0),
        standoff_linvel_gain: float = 1.0,
        standoff_yaw_gain: float = 1.0,
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        teleop: bool = False,
    ) -> None:
        super().__init__(env, teleop)
        body_ids, _ = self.asset.find_bodies(eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {eef_body_name!r}, got {body_ids.numel()}"
            )
        self.eef_body_idx = body_ids[0]

        if workspace_range is None and workspace_profile is None:
            raise ValueError(
                "Either workspace_range or workspace_profile must be provided"
            )
        if workspace_range is not None and workspace_profile is not None:
            raise ValueError(
                "Only one of workspace_range or workspace_profile can be provided"
            )
        if not 0.0 <= world_goal_prob <= 1.0:
            raise ValueError("world_goal_prob must be in [0, 1]")

        self.workspace_profile = workspace_profile
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.world_goal_prob = world_goal_prob
        self.standoff_distance_range = standoff_distance_range
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

        with torch.device(self.device):
            if workspace_range is not None:
                lows = torch.tensor(
                    [workspace_range[i][0] for i in range(3)], dtype=torch.float32
                )
                highs = torch.tensor(
                    [workspace_range[i][1] for i in range(3)], dtype=torch.float32
                )
                self._eef_pos_low = lows.unsqueeze(0).expand(self.num_envs, -1).clone()
                self._eef_pos_high = highs.unsqueeze(0).expand(self.num_envs, -1).clone()

            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            # (x,y): horizontal offsets in yaw-aligned frame; z: height above ground at target xy.
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pitch_w = torch.zeros(self.num_envs, 1)
            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.eef_up_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_up_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_vel_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_vel_w = torch.zeros(self.num_envs, 3)
            self.is_world_goal_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.world_env_ids = torch.empty(0, dtype=torch.long)
            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.command_speed = torch.zeros(self.num_envs, 1)

        self.marker = None
        self.standoff_marker = None
        if (
            self.env.backend == "isaac"
            and self.env.sim.has_gui()
        ):
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.marker = self.scene.create_sphere_marker(
                "/Visuals/Command/target_eef_pos",
                (1.0, 0.4, 0.0),
                radius=0.03,
            )
            self.standoff_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/standoff_pos",
                (0.0, 0.7, 1.0),
                radius=0.04,
            )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.cmd_linvel_b[:, :2], # [N, 2]
                self.cmd_yawvel_b, # [N, 1]
                self.cmd_eef_pos_b, # [N, 3]
                self.cmd_eef_pitch_w, # [N, 1]
                self.cmd_eef_pitch_w.cos(), # [N, 1]
                self.cmd_eef_pitch_w.sin(), # [N, 1]
            ],
            dim=-1,
        )
    
    @override
    def symmetry_transform(self):
        # flip y and yaw
        cmd_linvel_b = SymmetryTransform(perm=[0, 1], signs=[1, -1])
        cmd_yawvel_b = SymmetryTransform(perm=[0], signs=[-1])
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_pitch_w = SymmetryTransform(perm=[0, 1, 2], signs=[1, 1, 1])
        return SymmetryTransform.cat(
            [cmd_linvel_b, cmd_yawvel_b, cmd_eef_pos_b, cmd_eef_pitch_w]
        )

    @staticmethod
    def _env_mask_prob(num_envs: int, prob: float, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, device=device) < prob

    def _sample_local_eef_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.workspace_profile is not None:
            raise NotImplementedError(
                "workspace_profile sampling is not implemented; use workspace_range"
            )
        low = self._eef_pos_low[env_ids]
        high = self._eef_pos_high[env_ids]
        return torch.rand_like(low) * (high - low) + low

    def sample_loco_commands(self, env_ids: torch.Tensor) -> None: # env_ids is always non-empty
        # tensor[env_ids] is advanced indexing
        # so in-place operations like tensor[env_ids, 0].uniform_() have no effects
        new_cmd_linvel_b = torch.zeros(len(env_ids), 3, device=self.device)
        new_cmd_linvel_b[:, 0].uniform_(*self.linvel_x_range)
        new_cmd_linvel_b[:, 1].uniform_(*self.linvel_y_range)
        new_cmd_linvel_b[:, 2] = 0.0
        new_cmd_yawvel_b = torch.zeros(len(env_ids), 1, device=self.device)
        new_cmd_yawvel_b[:, 0].uniform_(*self.yaw_rate_range)
        self.cmd_linvel_b[env_ids] = new_cmd_linvel_b
        self.cmd_yawvel_b[env_ids] = new_cmd_yawvel_b

    def sample_manip_commands(self, env_ids: torch.Tensor) -> None: # env_ids is always non-empty
        self.cmd_eef_pos_b[env_ids] = self._sample_local_eef_offsets(env_ids)
        self.sample_eef_pitch_commands(env_ids, self.cmd_eef_pos_b[env_ids, 2])

    def sample_eef_pitch_commands(
        self, env_ids: torch.Tensor, target_height: torch.Tensor
    ) -> None:
        pitch_down = torch.rand(len(env_ids), device=self.device) * (torch.pi / 2)
        pitch_up = torch.rand(len(env_ids), device=self.device) * (torch.pi / 2) - (
            torch.pi / 2
        )
        self.cmd_eef_pitch_w[env_ids, 0] = torch.where(
            target_height < 0.4,
            pitch_down,
            pitch_up,
        )

    def sample_world_goal_commands(self, env_ids: torch.Tensor) -> None:
        root_pos = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])

        standoff_offset_b = torch.zeros(len(env_ids), 3, device=self.device)
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        d = torch.rand(len(env_ids), device=self.device) * (self.standoff_distance_range[1] - self.standoff_distance_range[0]) + self.standoff_distance_range[0]
        standoff_offset_b[:, 0] = d * torch.cos(a)
        standoff_offset_b[:, 1] = d * torch.sin(a)
        standoff_offset_w = quat_rotate(root_yaw_q, standoff_offset_b)
        standoff_pos_w = root_pos + standoff_offset_w
        standoff_pos_w[:, 2] = self.env.get_ground_height_at(standoff_pos_w)

        eef_offset_b = self._sample_local_eef_offsets(env_ids)
        eef_offset_w = quat_rotate(root_yaw_q, eef_offset_b)
        world_eef_pos_w = standoff_pos_w + eef_offset_w
        world_eef_pos_w[:, 2] = (
            self.env.get_ground_height_at(world_eef_pos_w) + eef_offset_b[:, 2]
        )

        self.standoff_pos_w[env_ids] = standoff_pos_w
        self.world_eef_pos_w[env_ids] = world_eef_pos_w
        self.standoff_yaw_w[env_ids, 0] = self.asset.data.heading_w[env_ids]
        self.sample_eef_pitch_commands(env_ids, eef_offset_b[:, 2])

    def _split_command_strategy(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_world = int(len(env_ids) * self.world_goal_prob + 0.5)
        shuffled = env_ids[torch.randperm(len(env_ids), device=self.device)]
        world_env_ids = shuffled[:num_world]
        local_env_ids = shuffled[num_world:]
        return local_env_ids, world_env_ids

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        local_env_ids, world_env_ids = self._split_command_strategy(env_ids)
        self.is_world_goal_env[env_ids] = False
        if local_env_ids.numel() > 0:
            self.sample_loco_commands(local_env_ids)
            self.sample_manip_commands(local_env_ids)
        if world_env_ids.numel() > 0:
            self.is_world_goal_env[world_env_ids] = True
            self.sample_world_goal_commands(world_env_ids)
        keep_cached = (self.world_env_ids[:, None] != env_ids[None, :]).all(dim=1)
        self.world_env_ids = torch.cat([self.world_env_ids[keep_cached], world_env_ids])

    def _sync_world_goal_envs(self, env_ids: torch.Tensor) -> None:
        root_pos = self.asset.data.root_link_pos_w[env_ids]
        yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])

        eef_delta_w = self.world_eef_pos_w[env_ids] - root_pos
        eef_delta_w[:, 2] = 0.0
        eef_delta_b = quat_rotate_inverse(yaw_q, eef_delta_w)
        self.cmd_eef_pos_b[env_ids, :2] = eef_delta_b[:, :2]
        self.cmd_eef_pos_b[env_ids, 2] = (
            self.world_eef_pos_w[env_ids, 2]
            - self.env.get_ground_height_at(self.world_eef_pos_w[env_ids])
        )

        standoff_delta_w = self.standoff_pos_w[env_ids] - root_pos
        standoff_delta_w[:, 2] = 0.0
        standoff_delta_b = quat_rotate_inverse(yaw_q, standoff_delta_w)
        self.cmd_linvel_b[env_ids, 0] = (
            self.standoff_linvel_gain * standoff_delta_b[:, 0]
        ).clamp(*self.linvel_x_range)
        self.cmd_linvel_b[env_ids, 1] = (
            self.standoff_linvel_gain * standoff_delta_b[:, 1]
        ).clamp(*self.linvel_y_range)
        self.cmd_linvel_b[env_ids, 2] = 0.0

        yaw_error = wrap_to_pi(
            self.standoff_yaw_w[env_ids] - self.asset.data.heading_w[env_ids, None]
        )
        self.cmd_yawvel_b[env_ids] = (
            self.standoff_yaw_gain * yaw_error
        ).clamp(*self.yaw_rate_range)

    def _sync_world_frames(self) -> None:
        """Sync command tensors that are derived from the current root pose."""
        quat_w = self.asset.data.root_link_quat_w
        yaw_q = yaw_quat(quat_w)
        world_env_ids = self.world_env_ids
        if world_env_ids.numel() > 0:
            self._sync_world_goal_envs(world_env_ids)
        self.cmd_linvel_w = quat_rotate(yaw_q, self.cmd_linvel_b)

        root_pos = self.asset.data.root_link_pos_w
        exy = torch.zeros(self.num_envs, 3, device=self.device)
        exy[:, :2] = self.cmd_eef_pos_b[:, :2]
        delta_w = quat_rotate(yaw_q, exy)
        horiz_w = root_pos + delta_w
        ground_h = self.env.get_ground_height_at(horiz_w)
        self.cmd_eef_pos_w[:, :2] = horiz_w[:, :2]
        self.cmd_eef_pos_w[:, 2] = ground_h + self.cmd_eef_pos_b[:, 2]
        if world_env_ids.numel() > 0:
            self.cmd_eef_pos_w[world_env_ids] = self.world_eef_pos_w[world_env_ids]
        
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.eef_body_idx]
        eef_rpy_w = euler_from_quat(eef_quat_w)
        eef_rpy_w[:, 0] = 0.0
        eef_rpy_w[:, 1] = self.cmd_eef_pitch_w.squeeze(-1)

        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        up_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.eef_forward_w = quat_rotate(eef_quat_w, forward_axis_b)
        self.eef_up_w = quat_rotate(eef_quat_w, up_axis_b)
        self.cmd_eef_forward_w = euler_rotate(eef_rpy_w, forward_axis_b)
        self.cmd_eef_up_w = euler_rotate(eef_rpy_w, up_axis_b)

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = (self.command_speed < 0.1)

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        self._sync_world_frames()

    @override
    def update(self) -> None:
        interval = (
            (self.env.episode_length_buf - 20) % self.resample_interval == 0
        )
        resample = interval & self._env_mask_prob(
            self.num_envs, self.resample_prob, self.device
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.sample_commands(env_ids)
        self._sync_world_frames()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.eef_body_idx],
            self.eef_forward_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.eef_body_idx],
            self.cmd_eef_forward_w,
            color=(0.0, 1.0, 0.0, 1.0),
        )
        self.marker.visualize(self.cmd_eef_pos_w)
        world_env_ids = self.world_env_ids
        if self.standoff_marker is not None and world_env_ids.numel() > 0:
            self.standoff_marker.visualize(self.standoff_pos_w[world_env_ids])


class eef_pos_tracking(Reward[SingleEEFLocoManip]):
    
    def __init__(self, env, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(env, weight, enabled=True, track_var=False)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.sigma = 0.1
    
    @override
    def _compute(self) -> torch.Tensor:
        diff_w = self.command_manager.cmd_eef_pos_w - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        error_norm_sq = diff_w.square().sum(dim=-1, keepdim=True)
        error_norm = error_norm_sq.sqrt()
        rew = torch.exp(-error_norm_sq / self.sigma) - 0.2 * error_norm
        return rew.reshape(self.num_envs, 1)


class eef_vel_tracking(Reward[SingleEEFLocoManip]):
    """
    Optionally track the velocity of the end-effector.
    """
    def __init__(self, env, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(env, weight, enabled=True, track_var=False)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.sigma = 0.2
    
    @override
    def _compute(self) -> torch.Tensor:
        diff_w = self.command_manager.cmd_eef_vel_w - self.asset.data.body_link_vel_w[:, self.eef_body_idx]
        error_l2 = diff_w.square().sum(dim=-1, keepdim=True)
        rew = torch.exp(-error_l2 / self.sigma)
        return rew.reshape(self.num_envs, 1)


class eef_forward_tracking(Reward[SingleEEFLocoManip]):
    """
    Track a global EEF pitch target through the end-effector forward direction.
    """

    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        pos_error_threshold: float = 0.15,
    ):
        super().__init__(env, weight, enabled=enabled, track_var=track_var)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.forward_axis_b = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.pos_error_threshold = pos_error_threshold

    @override
    def _compute(self) -> torch.Tensor:
        rew = (self.command_manager.eef_forward_w * self.command_manager.cmd_eef_forward_w).sum(
            dim=-1, keepdim=True
        )
        pos_error = (
            self.command_manager.cmd_eef_pos_w
            - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        ).norm(dim=-1, keepdim=True)
        active = pos_error < self.pos_error_threshold
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class eef_up_tracking(Reward[SingleEEFLocoManip]):
    """
    Track a global EEF pitch target through the end-effector up direction.
    """

    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        pos_error_threshold: float = 0.15,
    ):
        super().__init__(env, weight, enabled=enabled, track_var=track_var)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.pos_error_threshold = pos_error_threshold

    @override
    def _compute(self) -> torch.Tensor:
        rew = (self.command_manager.eef_up_w * self.command_manager.cmd_eef_up_w).sum(
            dim=-1, keepdim=True
        )
        pos_error = (
            self.command_manager.cmd_eef_pos_w
            - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        ).norm(dim=-1, keepdim=True)
        active = pos_error < self.pos_error_threshold
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class eef_angvel_penalty(Reward[SingleEEFLocoManip]):
    """
    Penalize oscillation of the end-effector.
    """
    def __init__(self, env, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(env, weight, enabled=True, track_var=False)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
    
    @override
    def _compute(self) -> torch.Tensor:
        angvel = self.asset.data.body_link_ang_vel_w[:, self.eef_body_idx]
        rew = - angvel.square().sum(dim=-1, keepdim=True)
        return rew.reshape(self.num_envs, 1)


__all__ = ["SingleEEFLocoManip"]
