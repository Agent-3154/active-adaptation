from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from tensordict import TensorDictBase

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.symmetry import SymmetryTransform

from ..base import MDPComponent
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


CT = TypeVar("CT", bound=Command)


class Observation(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred observation term."""

    def __init__(
        self,
        env: "_EnvBase" | None = None,
        *,
        functional: bool = False,
    ) -> None:
        super().__init__()
        self.functional = bool(functional)
        if env is not None:
            self._initialize(env)

    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.command_manager: CT = env.command_manager

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def fupdate(self, tensordict: TensorDictBase) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} sets functional=True but does not implement fupdate"
        )

    def fcompute(self, tensordict: TensorDictBase) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} sets functional=True but does not implement fcompute"
        )

    def symmetry_transform(self) -> SymmetryTransform:
        return NotImplementedError


__all__ = ["Observation"]
