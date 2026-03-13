from __future__ import annotations

import abc
import torch

from typing import Generic, Tuple, TypeVar

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


CT = TypeVar("CT", bound=Command)


class Termination(Generic[CT], MDPComponent, RegistryMixin):
    def __init__(self, env, is_timeout: bool = False, enabled: bool = True):
        super().__init__(env)
        self.command_manager: CT = env.command_manager
        self.is_timeout = is_timeout
        self.enabled = enabled

    @abc.abstractmethod
    def compute(
        self, termination: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


__all__ = ["Termination"]
