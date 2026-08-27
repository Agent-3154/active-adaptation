from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List

import torch

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.string import resolve_matching_names

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.utils.symmetry import SymmetryTransform


class Action(MDPComponent, RegistryMixin):
    """Environment-deferred action term."""

    action_dim: int
    action_buf: torch.Tensor

    def __init__(self, env: "_EnvBase" | None = None) -> None:
        super().__init__()
        self._names: list[str] | None = None
        if env is not None:
            self._initialize(env)

    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    @property
    def names(self) -> List[str]:
        if self._names is None:
            raise RuntimeError("Action names not set")
        return list(self._names)

    @names.setter
    def names(self, names: List[str]) -> None:
        assert len(names) == self.action_dim, f"Expected {self.action_dim} names, got {len(names)}"
        self._names = list(names)

    def find_names(self, names: str | List[str], preserve_order: bool = False):
        indices, names = resolve_matching_names(names, self.names, preserve_order=preserve_order)
        return indices, names

    @abc.abstractmethod
    def process_action(self, action: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, substep: int):
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}

    def symmetry_transform(self) -> SymmetryTransform:
        return NotImplementedError


__all__ = ["Action"]
