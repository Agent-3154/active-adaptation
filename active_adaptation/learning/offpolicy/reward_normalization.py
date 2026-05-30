"""Discounted-return variance reward scaling (FlashSAC-style).

Running stats on scalar discounted returns ``G_r`` are used to scale rewards at
training time: ``reward / max(sqrt(var(G_r) + eps), |G_r|_max / G_max)``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, TypeVar

import torch

Config = TypeVar("Config")


def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


def _invert_scale_reward_denominator(
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    """Inverse of the denominator in :func:`_scale_reward` (scalar tensor)."""
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    return torch.maximum(var_denominator, min_required_denominator)


def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    denominator = _invert_scale_reward_denominator(G_var, G_r_max, G_max, eps)
    return rewards / denominator


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
        G_max: float,
        load_rms: bool,
        device: torch.device,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
        self.G_max = G_max
        self.load_rms = load_rms
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )

    def reward_denominator(self) -> torch.Tensor:
        """Scalar S with ``r_normalized = r_raw / S`` (same S as :meth:`normalize_rewards`)."""
        return _invert_scale_reward_denominator(
            self.G_rms.var,
            self.G_r_max,
            float(self.G_max),
            self.epsilon,
        )

    def denormalize_return_values(self, values: torch.Tensor) -> torch.Tensor:
        """Undo reward scaling for logging Q-style quantities: ``values * S`` (broadcasts)."""
        s = self.reward_denominator().to(device=values.device, dtype=values.dtype)
        return values * s

    def state_dict(self) -> dict[str, Any]:
        """Serializable running statistics (checkpoint / :meth:`torch.save`)."""
        return OrderedDict(
            [
                ("G_r", self.G_r.detach().cpu()),
                ("G_r_max", self.G_r_max.detach().cpu()),
                ("G_rms_mean", self.G_rms.mean.detach().cpu()),
                ("G_rms_var", self.G_rms.var.detach().cpu()),
                ("G_rms_count", self.G_rms.count.detach().cpu()),
            ]
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore running stats on :attr:`device` from :meth:`state_dict`."""
        self.G_r = state["G_r"].to(device=self.device, dtype=torch.float32)
        self.G_r_max = state["G_r_max"].to(device=self.device, dtype=torch.float32)
        self.G_rms.mean = state["G_rms_mean"].to(device=self.device, dtype=torch.float32)
        self.G_rms.var = state["G_rms_var"].to(device=self.device, dtype=torch.float32)
        self.G_rms.count = state["G_rms_count"].to(device=self.device, dtype=torch.float32)
        self.G_rms.device = self.device

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
