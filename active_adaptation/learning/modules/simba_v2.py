"""SimbaV2 hyperspherical layers (PyTorch port of DAVIAN-Robotics/SimbaV2).

Paper: *Hyperspherical Normalization for Scalable Deep Reinforcement Learning*
(arXiv:2502.15280). Architecture mirrors ``scale_rl/agents/simbaV2/{simbaV2_layer,simbaV2_network}.py``.

SAC drop-ins
------------
- :class:`SimbaV2Actor` — same contract as ``sac.NormalActor``: ``forward(obs) -> (loc, scale)``.
- :class:`SimbaV2CriticTrunk` — same contract as ``sac.CriticTrunk``: ``forward(x [, cond]) -> y``
  (``cond`` ignored; kept for call-site compatibility with FiLM trunks).

After each optimizer step, call :func:`normalize_hyper_dense_` so ``HyperDense`` weights stay
on the unit sphere (JAX code does this via ``l2normalize_network``).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Feature / weight ℓ₂-normalize along ``dim`` (SimbaV2 hyperspherical proj.)."""
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def _default_scaler(hidden_dim: int) -> float:
    return math.sqrt(2.0 / hidden_dim)


def _default_alpha_init(num_blocks: int) -> float:
    return 1.0 / (num_blocks + 1)


def _default_alpha_scale(hidden_dim: int) -> float:
    return 1.0 / math.sqrt(hidden_dim)


class Scaler(nn.Module):
    """Learnable per-dim gain: ``scaler * (init / scale) * x``.

    Parameter is initialized to ``scale`` so the effective multiplier starts at ``init``.
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scaler = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    """Bias-free linear with orthogonal init; weights are kept ℓ₂-normalized in training."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        # Hyperspherical / periodic ℓ₂ renorm — keep off Muon.
        self.weight._non_muon = True
        nn.init.orthogonal_(self.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class HyperMLP(nn.Module):
    """``HyperDense → Scaler → ReLU+ε → HyperDense → ℓ₂-normalize``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + self.eps
        x = self.w2(x)
        return l2_normalize(x, dim=-1)


class HyperEmbedder(nn.Module):
    """Input embedder: concat constant shift → ℓ₂ → HyperDense → Scaler → ℓ₂."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.c_shift = float(c_shift)
        self.w = HyperDense(input_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = x.new_full((*x.shape[:-1], 1), self.c_shift)
        x = torch.cat([x, shift], dim=-1)
        x = l2_normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        return l2_normalize(x, dim=-1)


class HyperLERPBlock(nn.Module):
    """LERP residual block: ``x ← ℓ₂(x + α ⊙ (MLP(x) - x))`` with expansion MLP."""

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        exp = int(expansion)
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * exp,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(exp),
            scaler_scale=scaler_scale / math.sqrt(exp),
        )
        self.alpha_scaler = Scaler(hidden_dim, init=alpha_init, scale=alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        return l2_normalize(x, dim=-1)


class SimbaV2Encoder(nn.Module):
    """Shared SimbaV2 trunk: :class:`HyperEmbedder` + stacked :class:`HyperLERPBlock`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        c_shift: float = 3.0,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float | None = None,
        alpha_scale: float | None = None,
    ) -> None:
        super().__init__()
        s_init = _default_scaler(hidden_dim) if scaler_init is None else float(scaler_init)
        s_scale = _default_scaler(hidden_dim) if scaler_scale is None else float(scaler_scale)
        a_init = (
            _default_alpha_init(num_blocks) if alpha_init is None else float(alpha_init)
        )
        a_scale = (
            _default_alpha_scale(hidden_dim)
            if alpha_scale is None
            else float(alpha_scale)
        )
        self.hidden_dim = hidden_dim
        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=s_init,
            scaler_scale=s_scale,
            c_shift=c_shift,
        )
        self.blocks = nn.ModuleList(
            [
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=s_init,
                    scaler_scale=s_scale,
                    alpha_init=a_init,
                    alpha_scale=a_scale,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return x


class _PredictorDense(nn.Module):
    """Final SimbaV2 head: HyperDense → Scaler → HyperDense + bias (no feature ℓ₂)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.bias._non_muon = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        return self.w2(x) + self.bias


class SimbaV2CriticTrunk(nn.Module):
    """Drop-in replacement for ``sac.CriticTrunk`` (obs∥act features → ``output_dim``).

    ``cond`` is accepted and ignored so existing FiLM call sites keep working.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 1,
        num_blocks: int = 2,
        c_shift: float = 3.0,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float | None = None,
        alpha_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SimbaV2Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            c_shift=c_shift,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
        )
        self.head = _PredictorDense(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        del cond
        return self.head(self.encoder(x))


class SimbaV2Actor(nn.Module):
    """Drop-in replacement for ``sac.NormalActor``: diagonal Gaussian ``(loc, scale)``.

    Log-std uses the same tanh squash mapping as SimbaV2 / our ``NormalActor``.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 1,
        c_shift: float = 3.0,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float | None = None,
        alpha_scale: float | None = None,
        std_max: float = 1.0,
        std_min: float = 0.001,
    ) -> None:
        super().__init__()
        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_max = math.log(std_max)
        self.log_std_min = math.log(std_min)

        self.encoder = SimbaV2Encoder(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            c_shift=c_shift,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
        )
        self.mean_head = _PredictorDense(
            hidden_dim, hidden_dim, act_dim, scaler_init=1.0, scaler_scale=1.0
        )
        self.std_head = _PredictorDense(
            hidden_dim, hidden_dim, act_dim, scaler_init=1.0, scaler_scale=1.0
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        loc = self.mean_head(z)
        raw = self.std_head(z)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw)
        )
        return loc, torch.exp(log_std)


@torch.no_grad()
def normalize_hyper_dense_(module: nn.Module) -> None:
    """In-place ℓ₂-normalize every :class:`HyperDense` weight (post-optimizer, like JAX).

    For ``weight`` shaped ``[out, in]``, normalize along the input axis (dim=1) so each
    output row sits on the unit sphere — matching Flax ``column_axis=0`` on ``[in, out]``.
    """
    for m in module.modules():
        if isinstance(m, HyperDense):
            m.weight.copy_(l2_normalize(m.weight, dim=1))


def iter_hyper_dense(module: nn.Module) -> Iterable[HyperDense]:
    """Yield all :class:`HyperDense` submodules (useful for custom optim / logging)."""
    for m in module.modules():
        if isinstance(m, HyperDense):
            yield m
