from __future__ import annotations

import abc
from typing import Generic, TYPE_CHECKING, Tuple, TypeVar

import torch
from tensordict import TensorDictBase

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


CT = TypeVar("CT", bound=Command)


class Reward(Generic[CT], MDPComponent, RegistryMixin):
    """Environment-deferred scalar reward term."""

    _ema_decay: float = 0.99

    def __init__(
        self,
        env: "EnvBase" | None = None,
        *,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.enabled = enabled
        self.track_var = track_var
        if env is not None:
            self._initialize(env)

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

    def _update_ema(self, rew: torch.Tensor, count: torch.Tensor | float) -> None:
        dec = self._ema_decay
        self._ema_sum.mul_(dec).add_(rew.sum())
        self._ema_cnt.mul_(dec).add_(count)
        if self._ema_sum_sq is not None:
            self._ema_sum_sq.mul_(dec).add_(rew.square().sum())

    def compute(self) -> torch.Tensor:
        result = self._compute()
        if isinstance(result, torch.Tensor):
            rew = result
            count = float(result.numel())
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum()
        else:
            raise TypeError(result)
        rew = self.weight * rew * self.modifier
        self._modifier = torch.ones(self.num_envs, 1, device=self.device)
        self._update_ema(rew, count)
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
