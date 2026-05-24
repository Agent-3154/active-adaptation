from __future__ import annotations

from typing import Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    quat_mul,
    quat_rotate_inverse,
    sample_quat_yaw,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import Command


class LocoManipSparse(Command):
    """Sparse loco-manip command: EEF position target only, no base velocity command.

    Sampling strategy:
    1. Sample a world-frame EEF target at each env origin with ``xy = [0, 0]`` and
       ``z`` in ``eef_z_range`` (terrain-relative height).
    2. Spawn the robot at a random position on a ring around the env origin, given
       by ``spawn_radius_range``.

    The policy-facing command is the heading-frame EEF target
    ``[eef_x, eef_y, eef_z]``, using the same convention as ``SingleEEFLocoManip``:
    horizontal components are yaw-aligned offsets from the root; ``eef_z`` is height
    above terrain at the target ``xy``.

    We expect this task to be harder to learn from scratch due to exploration difficulties:
    there is no signal for the base movement.
    """

    def __init__(
        self,
        env,
        eef_body_name: str,
        eef_z_range: Tuple[float, float] = (0.2, 0.75),
        spawn_radius_range: Tuple[float, float] = (1.0, 3.0),
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
        self.eef_z_range = eef_z_range
        self.spawn_radius_range = spawn_radius_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

        with torch.device(self.device):
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_w = torch.zeros(self.num_envs, 3)
            
            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)

            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        self.marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.marker = self.scene.create_sphere_marker(
                "/Visuals/Command/sparse_eef_target",
                (1.0, 0.4, 0.0),
                radius=0.03,
            )

    @property
    def command(self) -> torch.Tensor:
        pos_diff_w = self.cmd_eef_pos_w - self.eef_pos_w
        pos_diff_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            pos_diff_w
        )
        return torch.cat([
            self.cmd_eef_pos_b,
            pos_diff_b,
        ], dim=-1)

    @override
    def symmetry_transform(self):
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        return SymmetryTransform.cat([cmd_eef_pos_b, pos_diff_b])

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
    
    def _update_eef_pos_error(self) -> None:
        self.pos_diff_w = self.cmd_eef_pos_w - self.eef_pos_w
        self.pos_diff_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.pos_diff_w,
        )
        self.pos_error_norm2 = self.pos_diff_w.square().sum(dim=-1, keepdim=True)
        self.pos_error_norm = self.pos_error_norm2.sqrt()

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        origins = self.env.scene.get_spawn_origins(env_ids)

        robot_init = self.init_root_state[env_ids].clone()
        default_z_offset = robot_init[:, 2].clone()

        angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        radius = self._sample_uniform(len(env_ids), self.spawn_radius_range)
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
        origins = self.env.scene.env_origins[env_ids]
        z_offset = self._sample_uniform(len(env_ids), self.eef_z_range)
        target_w = origins.clone()
        target_w[:, 2] = self.env.get_ground_height_at(origins) + z_offset
        self.world_eef_pos_w[env_ids] = target_w

    def _sync_command(self) -> None:
        root_pos = self.asset.data.root_link_pos_w
        yaw_q = yaw_quat(self.asset.data.root_link_quat_w)

        self.eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        self.cmd_eef_pos_w = self.world_eef_pos_w.clone()

        eef_delta_w = self.world_eef_pos_w - root_pos
        eef_delta_w[:, 2] = 0.0
        eef_delta_b = quat_rotate_inverse(yaw_q, eef_delta_w)
        self.cmd_eef_pos_b[:, :2] = eef_delta_b[:, :2]
        self.cmd_eef_pos_b[:, 2] = (
            self.world_eef_pos_w[:, 2]
            - self.env.get_ground_height_at(self.world_eef_pos_w)
        )

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        self._sync_command()
        self._update_eef_pos_error()

    @override
    def update(self) -> None:
        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample = interval & self._env_mask_prob(
            self.num_envs, self.resample_prob, self.device
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.sample_commands(env_ids)
        self._sync_command()
        self._update_eef_pos_error()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        if self.marker is not None:
            self.marker.visualize(self.cmd_eef_pos_w)


__all__ = ["LocoManipSparse"]
