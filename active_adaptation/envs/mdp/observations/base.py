from __future__ import annotations

import abc
import torch

from typing import TYPE_CHECKING, Generic, TypeVar

from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.symmetry import SymmetryTransform

from ..base import MDPComponent
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


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


class ObservationV2(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred observation term.

    Like :class:`Observation`, subclasses implement :meth:`compute`.

    Unlike :class:`Observation`, instances are constructed **without** an
    environment. Environment-bound state (``env``, ``command_manager``) is
    created in :meth:`_initialize`, which the environment calls once at
    startup. This allows observation logic to be reused without instantiating
    a simulator.

    Subclasses that need ``num_envs``/``device`` or sim handles should override
    :meth:`_initialize` and call ``super()._initialize(env)`` first.
    """

    def __init__(self) -> None:
        self._initialized = False

    def _initialize(self, env: "_EnvBase") -> None:
        """Bind to ``env``. Called once at startup."""
        self.env = env
        self.command_manager: CT = env.command_manager
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """``True`` after :meth:`_initialize` has been called."""
        return self._initialized

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def symmetry_transform(self) -> SymmetryTransform:
        pass


__all__ = ["Observation", "ObservationV2"]
