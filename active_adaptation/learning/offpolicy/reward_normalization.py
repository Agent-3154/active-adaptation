"""Discounted-return variance reward scaling (FlashSAC-style).

Running stats on scalar discounted returns ``G_r`` scale rewards at training
time: ``reward / sqrt(var(G_r) + eps)``. No task-dependent return cap.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, TypeVar

import torch
import torch.distributed as dist

Config = TypeVar("Config")


def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    done = torch.logical_or(terminated, truncated).float()
    return gamma * (1.0 - done) * G_r + reward


def _reward_denominator(G_var: torch.Tensor, eps: float) -> torch.Tensor:
    """``sqrt(var(G_r) + eps)`` used as the reward scale."""
    return torch.sqrt(G_var + eps)


def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return rewards / _reward_denominator(G_var, eps)


def _update_mean_var_count_from_moments(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return new_mean, new_var, total_count


class RunningMeanStd:
    """Tracks mean, variance, and count (Welford / parallel algorithm)."""

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )


class RewardNormalizer:
    """Normalize rewards using running variance of discounted-return estimates."""

    def __init__(
        self,
        gamma: float,
        load_rms: bool,
        device: torch.device,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
        self.load_rms = load_rms
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.G_r = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            eps=self.epsilon,
        )

    def reward_denominator(self) -> torch.Tensor:
        """Scalar S with ``r_normalized = r_raw / S`` (same S as :meth:`normalize_rewards`)."""
        return _reward_denominator(self.G_rms.var, self.epsilon)

    def denormalize_return_values(self, values: torch.Tensor) -> torch.Tensor:
        """Map Q trained on normalized rewards to PPO effective-horizon log scale.

        Undoes return-std reward scaling (``* S``) then multiplies by ``(1 - gamma)``
        so logged values match on-policy critics trained with
        ``reward * (1 - gamma)``. Without a normalizer, off-policy already uses
        that reward scale and needs no conversion.
        """
        s = self.reward_denominator().to(device=values.device, dtype=values.dtype)
        return values * s * (1.0 - float(self.gamma))

    def state_dict(self) -> dict[str, Any]:
        """Serializable running statistics (checkpoint / :meth:`torch.save`)."""
        return OrderedDict(
            [
                ("G_r", self.G_r.detach().cpu()),
                ("G_rms_mean", self.G_rms.mean.detach().cpu()),
                ("G_rms_var", self.G_rms.var.detach().cpu()),
                ("G_rms_count", self.G_rms.count.detach().cpu()),
            ]
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore running stats on :attr:`device` from :meth:`state_dict`.

        Older checkpoints may still contain unused ``G_r_max``; it is ignored.
        """
        self.G_r = state["G_r"].to(device=self.device, dtype=torch.float32)
        self.G_rms.mean = state["G_rms_mean"].to(device=self.device, dtype=torch.float32)
        self.G_rms.var = state["G_rms_var"].to(device=self.device, dtype=torch.float32)
        self.G_rms.count = state["G_rms_count"].to(device=self.device, dtype=torch.float32)
        self.G_rms.device = self.device

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def _running_stat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.G_r,
            self.G_rms.mean,
            self.G_rms.var,
            self.G_rms.count,
        )

    @torch.no_grad()
    def synchronize(self, mode: str = "broadcast") -> None:
        """Synchronize running stats across distributed ranks.

        Args:
            mode: ``"broadcast"`` copies rank-0 stats to every rank (used by
                SAC after per-rank rollout updates). ``"aggregate"`` merges
                :attr:`G_rms` with a count-weighted mean/var reduction and
                averages :attr:`G_r`.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed training is not initialized")

        if mode == "broadcast":
            for tensor in self._running_stat_tensors():
                dist.broadcast(tensor, src=0)
        elif mode == "aggregate":
            dist.all_reduce(self.G_r, op=dist.ReduceOp.AVG)

            mean = self.G_rms.mean.to(dtype=torch.float64)
            var = self.G_rms.var.to(dtype=torch.float64)
            count = self.G_rms.count.to(dtype=torch.float64)
            sqmean = var + mean.square()
            weighted_mean = mean * count
            weighted_sqmean = sqmean * count
            dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_sqmean, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

            count_global = count.clamp_min(1.0)
            mean_global = weighted_mean / count_global
            sqmean_global = weighted_sqmean / count_global
            var_global = (sqmean_global - mean_global.square()).clamp_min(
                self.G_rms.epsilon
            )
            self.G_rms.mean.copy_(mean_global.to(dtype=self.G_rms.mean.dtype))
            self.G_rms.var.copy_(var_global.to(dtype=self.G_rms.var.dtype))
            self.G_rms.count.copy_(count.to(dtype=self.G_rms.count.dtype))
        else:
            raise ValueError(f"Invalid mode: {mode}")
