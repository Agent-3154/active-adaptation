from __future__ import annotations

import abc
from typing import Generic, TYPE_CHECKING, Tuple, TypeVar, Optional, Sequence, final

import torch
from tensordict import TensorDictBase

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent, check_update_signature
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


CT = TypeVar("CT", bound=Command)


class Reward(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred scalar reward term."""

    _ema_decay: float = 0.99
    in_keys: Optional[Sequence[str]] = None
    out_keys: Optional[Sequence[str]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        check_update_signature(cls, owner="Reward")
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.enabled = enabled
        self.track_var = track_var

    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)
        self.command_manager: CT = env.command_manager
        self._modifier = torch.ones(self.num_envs, 1, device=self.device)
        self._ema_sum = torch.zeros(1, device=self.device)
        self._ema_cnt = torch.zeros(1, device=self.device)
        self._ema_sum_sq = torch.zeros(1, device=self.device) if self.track_var else None

    @property
    def modifier(self) -> torch.Tensor:
        return self._modifier

    def _update_ema(
        self, rew: torch.Tensor, is_active: torch.Tensor | None = None
    ) -> None:
        finite = torch.isfinite(rew)
        if is_active is not None:
            finite &= is_active.bool().expand_as(rew)
        safe_rew = torch.where(finite, rew, torch.zeros_like(rew))
        dec = self._ema_decay
        self._ema_sum.mul_(dec).add_(safe_rew.sum())
        self._ema_cnt.mul_(dec).add_(finite.sum())
        if self._ema_sum_sq is not None:
            self._ema_sum_sq.mul_(dec).add_(safe_rew.square().sum())

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

    def compute(self) -> torch.Tensor:
        result = self._compute()
        if isinstance(result, torch.Tensor):
            rew = result
            is_active = None
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
        else:
            raise TypeError(result)
        rew = self.weight * rew * self.modifier
        self._modifier = torch.ones(self.num_envs, 1, device=self.device)
        self._update_ema(rew, is_active)
        return rew

    def get_ema_stats(self) -> Tuple[torch.Tensor, torch.Tensor | None]:
        cnt = self._ema_cnt.clamp(min=1e-8)
        mean = (self._ema_sum / cnt).reshape(())
        if self._ema_sum_sq is None:
            return mean, None
        e_x2 = (self._ema_sum_sq / cnt).reshape(())
        var = (e_x2 - mean * mean).clamp(min=0.0).reshape(())
        return mean, var

    @abc.abstractmethod
    def _compute(self) -> torch.Tensor:
        raise NotImplementedError

    def relabel(self, tensordict: TensorDictBase) -> torch.Tensor:
        raise NotImplementedError(f"Relabeling not implemented for {self.__class__.__name__}")


__all__ = ["Reward"]
