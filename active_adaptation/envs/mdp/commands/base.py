from __future__ import annotations

import torch

from typing import TYPE_CHECKING

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.math import quat_mul, sample_quat_yaw

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Command(MDPComponent, RegistryMixin):
    def __init__(self, env: _EnvBase, teleop: bool = False) -> None:
        super().__init__(env)
        self.asset = env.scene.articulations["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self.teleop = teleop
    
    def step(self):
        pass

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.get_spawn_origins(env_ids)
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )
        return init_root_state


__all__ = ["Command"]
