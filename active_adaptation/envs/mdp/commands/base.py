from __future__ import annotations

import torch

from typing import TYPE_CHECKING

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.math import quat_mul, sample_quat_yaw

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Command(MDPComponent, RegistryMixin):
    """High-level command source for the MDP.

    Each env step, after simulation: ``update`` runs first, then rewards and
    terminations read ``command_manager``, then ``step`` runs, then
    observations are built. Override :meth:`update` to refresh command targets
    and any tensors that rewards or terminations depend on. Override
    :meth:`step` only when you need logic after reward/termination (e.g.
    bookkeeping that must not affect this step's reward/termination, or state
    consumed on the next step).
    """

    def __init__(self, env: _EnvBase, teleop: bool = False) -> None:
        super().__init__(env)
        self.asset = env.scene.articulations["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self.teleop = teleop
    
    def update(self) -> None:
        """Refresh command targets and any tensors that rewards or terminations depend on."""
        pass
    
    def step(self) -> None:
        """Hook after rewards and terminations, before observations.

        :meth:`update` runs earlier in the same env step so reward and
        termination terms see the command state it sets. Use ``step`` for
        follow-up work that should not influence this step's reward or
        termination (most command implementations only need ``update``).
        """
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
