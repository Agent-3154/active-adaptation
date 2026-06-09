from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from tensordict import TensorDict

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


class CommandV2(MDPComponent, RegistryMixin):
    """Environment-deferred command source for the MDP.

    Like :class:`Command`, subclasses implement :meth:`update` to refresh command
    targets and any tensors that rewards or terminations depend on.

    Unlike :class:`Command`, instances are constructed **without** an environment.
    Environment-bound state (``env``, ``asset``, default root/joint states) is
    created in :meth:`_initialize`, which the environment calls once at startup.
    This allows command logic to be reused for **command relabeling** on stored
    trajectories without instantiating a simulator.

    CommandV2 does not support teleop.

    Subclasses that need ``num_envs``/``device`` or sim handles should override
    :meth:`_initialize` and call ``super()._initialize(env)`` first.
    """

    def __init__(self) -> None:
        self._initialized = False

    def _initialize(self, env: _EnvBase) -> None:
        """Bind to ``env`` and cache articulation defaults. Called once at startup."""
        self.env = env
        self.asset = env.scene.articulations["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """``True`` after :meth:`_initialize` has been called."""
        return self._initialized

    def update(self) -> None:
        """Refresh command targets and any tensors that rewards or terminations depend on."""
        pass

    def step(self) -> None:
        """Hook after rewards and terminations, before observations."""
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
    
    # for relabeling
    def get_state(self) -> TensorDict:
        raise NotImplementedError()

    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        raise NotImplementedError()


__all__ = ["Command", "CommandV2"]
