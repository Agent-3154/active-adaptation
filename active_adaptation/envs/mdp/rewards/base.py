from __future__ import annotations

import abc
import torch

from typing import Generic, TypeVar, Tuple

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent
from ..commands.base import Command


CT = TypeVar("CT", bound=Command)


class Reward(Generic[CT], MDPComponent, RegistryMixin):
    """
    Scalar reward term: subclasses implement ``_compute``; ``compute`` applies ``weight`` and updates optional EMA stats for logging.

    Args:
        env: The environment object.
        weight: The weight of the reward term.
        track_var: Whether to track the variance of the reward term elements.
        ema_decay: The decay rate of the EMA.
    """

    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__(env)
        self.command_manager: CT = env.command_manager
        self.weight = weight
        self.enabled = enabled
        self.track_var = track_var
        self._ema_decay = float(ema_decay)
        d = self.device
        self._ema_sum = torch.zeros(1, device=d)
        self._ema_cnt = torch.zeros(1, device=d)
        if track_var:
            # EMA of sum(x^2) so variance can be computed as E[x^2] - E[x]^2
            self._ema_sum_sq = torch.zeros(1, device=d)
        else:
            self._ema_sum_sq = None

    def _update_ema(self, rew: torch.Tensor, count: torch.Tensor | float) -> None:
        dec = self._ema_decay
        s = rew.sum()
        self._ema_sum.mul_(dec).add_(s)
        self._ema_cnt.mul_(dec).add_(count)

        if self._ema_sum_sq is not None:
            s2 = rew.square().sum()
            self._ema_sum_sq.mul_(dec).add_(s2)

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
        rew = self.weight * rew
        self._update_ema(rew, count)
        return rew

    def get_ema_stats(self) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return EMA mean (E[x]) and EMA element-variance (E[x^2] - E[x]^2)."""
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


__all__ = ["Reward"]
