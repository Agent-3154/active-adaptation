from __future__ import annotations

import abc
import torch

from typing import Generic, TypeVar, Tuple

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


CT = TypeVar("CT", bound=Command)


class Reward(Generic[CT], MDPComponent, RegistryMixin):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env)
        self.command_manager: CT = env.command_manager
        self.weight = weight
        self.enabled = enabled

    def compute(self) -> Tuple[torch.Tensor, int]:
        result = self._compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum()
        return self.weight * rew, count

    @abc.abstractmethod
    def _compute(self) -> torch.Tensor:
        raise NotImplementedError


__all__ = ["Reward"]
