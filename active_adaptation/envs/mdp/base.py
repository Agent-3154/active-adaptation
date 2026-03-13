from __future__ import annotations

import torch

from typing import TYPE_CHECKING


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

    def __init__(self, env: _EnvBase):
        self.env: _EnvBase = env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def update(self) -> None:
        pass

    def pre_step(self, substep: int) -> None:
        pass

    def post_step(self, substep: int) -> None:
        pass

    def startup(self) -> None:
        pass

    def debug_draw(self) -> None:
        pass

__all__ = [
    "MDPComponent",
    "is_method_implemented",
]
