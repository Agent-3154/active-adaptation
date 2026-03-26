"""Locomotion + single end-effector position commands (scaffold)."""

from __future__ import annotations

from typing import Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import quat_rotate, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform
from .base import Command
from ..rewards.base import Reward


class SingleEEFLocoManip(Command):
    """Command vector: planar base velocity, yaw rate, and EEF target in a **heading frame**.

    Layout: ``[v_x, v_y, yaw_rate, eef_x, eef_y, eef_z]`` (6D). The first two loco components
    are in the usual body horizontal frame (same as ``Twist``); ``eef_x``/``eef_y`` are **not**
    full body frame: they use the same **yaw-only** rotation as world ``(x,y)`` offsets from the
    root (pitch/roll of the base are ignored for the horizontal part). ``eef_z`` is **height
    above terrain**, not root-link ``z``: world target height is
    ``get_ground_height_at(query_xy) + eef_z``, with ``query_xy`` the horizontal target under
    the root. Extend ``command`` if you add base height, wrist orientation, etc.
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

        self.workspace_profile = workspace_profile
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
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
            self.cmd_eef_vel_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_vel_w = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.command_speed = torch.zeros(self.num_envs, 1)

        self.marker = None
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

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.cmd_linvel_b[:, :2], # [N, 2]
                self.cmd_yawvel_b, # [N, 1]
                self.cmd_eef_pos_b, # [N, 3]
            ],
            dim=-1,
        )
    
    @override
    def symmetry_transform(self):
        # flip y and yaw
        cmd_linvel_b = SymmetryTransform(perm=[0, 1], signs=[1, -1])
        cmd_yawvel_b = SymmetryTransform(perm=[0], signs=[-1])
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        return SymmetryTransform.cat([cmd_linvel_b, cmd_yawvel_b, cmd_eef_pos_b])

    @staticmethod
    def _env_mask_prob(num_envs: int, prob: float, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, device=device) < prob

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
        if self.workspace_profile is not None:
            raise NotImplementedError(
                "workspace_profile sampling is not implemented; use workspace_range"
            )
        low = self._eef_pos_low[env_ids]
        high = self._eef_pos_high[env_ids]
        self.cmd_eef_pos_b[env_ids] = torch.rand_like(low) * (high - low) + low

    def _sync_world_frames(self) -> None:
        quat_w = self.asset.data.root_link_quat_w
        yaw_q = yaw_quat(quat_w)
        self.cmd_linvel_w = quat_rotate(yaw_q, self.cmd_linvel_b)

        root_pos = self.asset.data.root_link_pos_w
        exy = torch.zeros(self.num_envs, 3, device=self.device)
        exy[:, :2] = self.cmd_eef_pos_b[:, :2]
        delta_w = quat_rotate(yaw_q, exy)
        horiz_w = root_pos + delta_w
        ground_h = self.env.get_ground_height_at(horiz_w)
        self.cmd_eef_pos_w[:, :2] = horiz_w[:, :2]
        self.cmd_eef_pos_w[:, 2] = ground_h + self.cmd_eef_pos_b[:, 2]

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = (self.command_speed < 0.1)

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_loco_commands(env_ids)
        self.sample_manip_commands(env_ids)
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
            self.sample_loco_commands(env_ids)
            self.sample_manip_commands(env_ids)
        self._sync_world_frames()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.marker.visualize(self.cmd_eef_pos_w)


class eef_pos_tracking(Reward[SingleEEFLocoManip]):
    
    def __init__(self, env, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(env, weight, enabled, track_var)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.sigma = 0.1
    
    @override
    def _compute(self) -> torch.Tensor:
        diff_w = self.command_manager.cmd_eef_pos_w - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        error_l2 = diff_w.square().sum(dim=-1, keepdim=True)
        rew = torch.exp(-error_l2 / self.sigma)
        return rew.reshape(self.num_envs, 1)


class eef_vel_tracking(Reward[SingleEEFLocoManip]):
    """
    Optionally track the velocity of the end-effector.
    """
    def __init__(self, env, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(env, weight, enabled, track_var)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self.sigma = 0.2
    
    @override
    def _compute(self) -> torch.Tensor:
        diff_w = self.command_manager.cmd_eef_vel_w - self.asset.data.body_link_vel_w[:, self.eef_body_idx]
        error_l2 = diff_w.square().sum(dim=-1, keepdim=True)
        rew = torch.exp(-error_l2 / self.sigma)
        return rew.reshape(self.num_envs, 1)



__all__ = ["SingleEEFLocoManip"]
