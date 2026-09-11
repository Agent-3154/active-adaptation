from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, Optional, Sequence, Tuple, TypeVar, final

import torch
from tensordict import TensorDictBase

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent, check_update_signature
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


CT = TypeVar("CT", bound=Command)


class Termination(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred termination term."""

    in_keys: Optional[Sequence[str]] = None
    out_keys: Optional[Sequence[str]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        check_update_signature(cls, owner="Termination")
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        is_timeout: bool = False,
        enabled: bool = True,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.is_timeout = is_timeout
        self.enabled = enabled
        if in_keys is not None:
            self.in_keys = list(in_keys)
        if out_keys is not None:
            self.out_keys = list(out_keys)

    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.command_manager: CT = env.command_manager

    @final
    def update(self, tensordict: TensorDictBase) -> None:
        """Sealed dispatcher: subclasses must implement :meth:`_update` instead."""
        if self.in_keys is not None:
            tensors_in = (tensordict.get(in_key, None) for in_key in self.in_keys)
        else:
            tensors_in = ()
        tensors_out = self._update(*tensors_in)

        if tensors_out is None and self.out_keys is None:
            return
        if not isinstance(tensors_out, tuple):
            tensors_out = (tensors_out,)
        for out_key, tensor_out in zip(self.out_keys, tensors_out, strict=True):
            tensordict.set(out_key, tensor_out)
        return tensordict

    def _update(self, *tensors_in: torch.Tensor) -> None:
        """Refresh buffers after simulation. Override when the term caches state.

        ``*tensors_in`` matches :attr:`in_keys` (empty when ``in_keys`` is
        ``None``). Missing keys are passed as ``None``. Subclass parameters
        must not declare defaults. Return values are written to
        :attr:`out_keys` when set.
        """
        return None

    @abc.abstractmethod
    def compute(
        self, termination: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


__all__ = ["Termination"]
