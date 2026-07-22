from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Any
from tensordict import TensorDictBase


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


def is_method_implemented(obj, base_class, method_name: str):
    """Check if a method is actually implemented (not just the base class pass)."""
    obj_method = getattr(type(obj), method_name, None)
    base_method = getattr(base_class, method_name, None)

    if obj_method is None or base_method is None:
        return False

    obj_func = getattr(obj_method, "__func__", obj_method)
    base_func = getattr(base_method, "__func__", base_method)
    return obj_func is not base_func


class MDPComponent:
    """Shared lifecycle hooks and environment access for MDP components."""

    markovian: bool # whether the component is markovian, i.e, dependent only on the current state

    def __init__(self, env: _EnvBase):
        self.env: _EnvBase = env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device
    
    def edit_spec(self, scene_config: Any) -> None:
        "The MDP term may optionally edit the scene config to, for example, add its own sensors."

    def startup(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        """Reset per-env state for ``env_ids``.

        Both arguments are required. Terms may read from and write into
        ``tensordict`` (e.g. controlled / curriculum resets). Most terms leave
        it unused.

        Note:
            Initial root/joint state is still set via ``command_manager.sample_init``
            in ``_reset_idx`` before these callbacks run. A future change will move
            that responsibility into ``reset``.
        """
        pass

    def update(self) -> None:
        pass

    def pre_step(self, substep: int) -> None:
        pass

    def post_step(self, substep: int) -> None:
        pass

    def debug_draw(self) -> None:
        pass

__all__ = [
    "MDPComponent",
    "is_method_implemented",
]
