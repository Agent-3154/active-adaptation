from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from tensordict import TensorDict, TensorDictBase

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.math import quat_mul, sample_quat_yaw

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Command(MDPComponent, RegistryMixin):
    """Environment-deferred command source for the MDP."""

    in_keys: Optional[Sequence[str]] = None
    out_keys: Optional[Sequence[str]] = None

    def __init__(self) -> None:
        super().__init__()

    def _initialize(self, env: _EnvBase) -> None:
        super()._initialize(env)
        self.asset = env.scene.articulations["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()

    def update(self, tensordict: TensorDictBase) -> None:
        """Do not override this method."""
        if self.in_keys is not None:
            tensors_in = (tensordict.get(in_key) for in_key in self.in_keys)
        else:
            tensors_in = ()
        tensors_out = self._update(*tensors_in)

        if tensors_out is None and self.out_keys is not None:
            return
        if not isinstance(tensors_out, tuple):
            tensors_out = (tensors_out,)
        for out_key, tensor_out in zip(self.out_keys, tensors_out, strict=True):
            tensordict.set(out_key, tensor_out)
        return tensordict

    @abstractmethod
    def _update(self) -> None:
        """Refresh current command-dependent tensors after simulation."""

    def step(self) -> None:
        """Advance command time or resample future targets."""

    def sample_init(
        self,
        env_ids: torch.Tensor,
        reset_td: TensorDictBase | None = None,
    ) -> None:
        init_root_state = self.init_root_state[env_ids].clone()
        origins = self.env.scene.sample_spawn_origin_candidates(env_ids)
        self.env.episode_origin[env_ids] = origins
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )
        self._write_initial_states({"robot": init_root_state}, env_ids)
        entity = self.env.scene["robot"]
        entity.write_joint_state_to_sim(
            self.init_joint_pos[env_ids],
            self.init_joint_vel[env_ids],
            env_ids=env_ids,
        )

    def _write_initial_states(
        self,
        states: dict[str, torch.Tensor] | torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        if not isinstance(states, dict):
            states = {"robot": states}
        for entity_name, state in states.items():
            entity = self.env.scene[entity_name]
            if self.env.backend == "mjlab" and entity.is_fixed_base:
                entity.write_mocap_pose_to_sim(state[:, :7], env_ids=env_ids)
            else:
                entity.write_root_state_to_sim(state, env_ids=env_ids)

    def prescribe(self, tensordict: TensorDictBase) -> None:
        """Fill prescribed control inputs before action processing."""
        return None

    def get_state(self) -> TensorDict:
        raise NotImplementedError(f"Method `get_state` is not implemented for {self.__class__.__name__}")

    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        raise NotImplementedError(f"Method `relabel_command` is not implemented for {self.__class__.__name__}")


__all__ = ["Command"]
