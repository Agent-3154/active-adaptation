from __future__ import annotations

import abc
import torch

from typing import TYPE_CHECKING, Generic, Tuple, TypeVar

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


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


class TerminationV2(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred termination term.

    Like :class:`Termination`, subclasses implement :meth:`compute`.

    Unlike :class:`Termination`, instances are constructed **without** an
    environment. Environment-bound state (``env``, ``command_manager``) is
    created in :meth:`_initialize`, which the environment calls once at startup.

    Subclasses that need ``num_envs``/``device`` or sim handles should override
    :meth:`_initialize` and call ``super()._initialize(env)`` first.

    Args:
        is_timeout: If ``True``, this term contributes to truncation rather than
            termination.
        enabled: If ``False``, the term can be skipped by the env.
    """

    def __init__(self, is_timeout: bool = False, enabled: bool = True):
        self.is_timeout = is_timeout
        self.enabled = enabled
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
    def compute(
        self, termination: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


__all__ = ["Termination", "TerminationV2"]
