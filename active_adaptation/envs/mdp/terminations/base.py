from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, Tuple, TypeVar

import torch

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


CT = TypeVar("CT", bound=Command)


class Termination(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred termination term."""

    def __init__(
        self,
        env: "_EnvBase" | None = None,
        *,
        is_timeout: bool = False,
        enabled: bool = True,
    ):
        super().__init__()
        self.is_timeout = is_timeout
        self.enabled = enabled
        if env is not None:
            self._initialize(env)

    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.command_manager: CT = env.command_manager

    @abc.abstractmethod
    def compute(
        self, termination: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


__all__ = ["Termination"]
