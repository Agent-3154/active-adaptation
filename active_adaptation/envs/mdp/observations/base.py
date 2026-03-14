from __future__ import annotations

import abc
import torch

from typing import Generic, Tuple, TypeVar

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.symmetry import SymmetryTransform

from ..base import MDPComponent
from ..commands.base import Command


CT = TypeVar("CT", bound=Command)


class Observation(Generic[CT], MDPComponent, RegistryMixin):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager: CT = env.command_manager

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def symmetry_transform(self) -> SymmetryTransform:
        pass


__all__ = ["Observation"]
