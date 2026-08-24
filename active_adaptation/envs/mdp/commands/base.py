from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from tensordict import TensorDict, TensorDictBase

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.math import quat_mul, sample_quat_yaw
from abc import abstractmethod

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Command(MDPComponent, RegistryMixin):
    """High-level command source for the MDP.

    Each env step, after simulation: :meth:`sync_state` runs first, then rewards
    and terminations read ``command_manager``, then :meth:`update` runs, then
    observations are built. Override :meth:`sync_state` to refresh intermediate
    tensors that rewards or terminations depend on without changing commands.
    Override :meth:`update` when commands may change (e.g. resampling).
    """

    def __init__(self, env: _EnvBase, teleop: bool = False) -> None:
        super().__init__(env)
        self.asset = env.scene.articulations["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self.teleop = teleop

    def sync_state(self) -> None:
        """Refresh intermediate state for rewards/terminations; do not change commands."""
        pass

    def update(self) -> None:
        """Hook after rewards and terminations, before observations."""
        pass

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.sample_spawn_origin_candidates(env_ids)
        self.env.episode_origin[env_ids] = origins
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

    @abstractmethod
    def sync_state(self) -> None:
        """
        Refresh intermediate state tensors that rewards or terminations depend on.
        The command should not change during `sync_state`.
        
        We make this method abstract so that the users are explicitly aware of the
        difference between `sync_state` and `update`.
        """
    
    @abstractmethod
    def update(self) -> None:
        """Hook after rewards and terminations, before observations.

        The command may change during `update`, e.g., gets resampled.
        
        We make this method abstract so that the users are explicitly aware of the
        difference between `sync_state` and `update`.
        """

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        init_root_state = self.init_root_state[env_ids]
        origins = self.env.scene.sample_spawn_origin_candidates(env_ids)
        self.env.episode_origin[env_ids] = origins
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )
        return init_root_state
    
    def prescribe(self, tensordict: TensorDictBase) -> None:
        """Fill prescribed control inputs on the step tensordict before action processing.

        Called once at the start of each env step, **before** input managers run
        :meth:`~active_adaptation.envs.mdp.actions.base.ActionV2.process_action`.
        Subclasses may write tensors under keys that match ``task.input`` entries
        (e.g. ``arm_control``) when the policy did not supply them.

        This is the hook for **command-driven actuation**: reference trajectories,
        hand-crafted controllers, or other task logic that should not be learned
        by the policy. Only fill keys that are **missing**
        (``tensordict.get(key) is None``) so a policy or teleop can still
        override a slot when present.

        **Why ``prescribe``?** The command *writes* reference inputs into the
        step batch; it does not return a single action vector and is not the
        policy's action API. The name avoids colliding with RL *action*
        terminology and teleop ``get_action``, and reads as intentional task
        logic ("the command prescribes the arm target") rather than a passive
        lookup.

        Example (tracking task with policy base control + command-driven IK)::

            def prescribe(self, tensordict):
                if tensordict.get("arm_control") is None:
                    tensordict.set(
                        "arm_control",
                        torch.cat([self.cmd_eef_pos_b, self.cmd_eef_quat_b], dim=-1),
                    )

        Default implementation is a no-op.
        """
        pass

    
    # for relabeling
    def get_state(self) -> TensorDict:
        raise NotImplementedError(f"Method `get_state` is not implemented for {self.__class__.__name__}")

    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        raise NotImplementedError(f"Method `relabel_command` is not implemented for {self.__class__.__name__}")


__all__ = ["Command", "CommandV2"]
