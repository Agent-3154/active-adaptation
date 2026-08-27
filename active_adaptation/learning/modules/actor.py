import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Maps backbone features to a diagonal Gaussian policy (mean and scale).

    The final linear uses ``nn.LazyLinear``; its weight is marked ``_non_muon`` so
    optimizers such as ``MuonAdamWWrapper`` keep the output head on AdamW.

    If ``predict_std`` is false, per-action standard deviation is a single learned
    vector ``actor_std`` broadcast to the batch; ``scale_mapping`` is identity. If
    true, the linear output is split; the second half is mapped with ``torch.exp`` to
    a positive diagonal scale.

    Args:
        action_dim: Size of the action vector (output dim of the mean, or half of
            the linear output when ``predict_std`` is true).
        predict_std: If true, one linear layer outputs ``2 * action_dim`` and is
            chunked into ``loc`` and ``scale``; if false, only ``loc`` is predicted.
    """

    def __init__(self, action_dim: int, predict_std: bool = False) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
            self.scale_mapping = torch.exp
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.ones(action_dim))
            self.scale_mapping = nn.Identity()
        self.actor_mean.weight._non_muon = True

    def forward(self, features: torch.Tensor):
        """Return ``(loc, scale)`` tensors for ``IndependentNormal`` (or equivalent)."""
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale


def _softplus_inv(y: float) -> float:
    """Inverse of ``softplus`` for positive ``y`` (used for parameter init)."""
    return math.log(math.expm1(y))


class ActorCov(nn.Module):
    """Maps backbone features to a scale–correlation factored Gaussian.

    Covariance::

        Σ = diag(σ) (I + F diag(α) Fᵀ) diag(σ)

    with column-normalized ``F ∈ R^{N×K}`` and per-factor ``α_k ≥ 0``. Equivalently,
    for ``LowRankMultivariateNormal``::

        cov_diag   = σ²
        cov_factor = diag(σ) F diag(√α)

    so ``Σ = cov_factor cov_factorᵀ + diag(cov_diag)``.

    Separating **scale** (``σ``) from **correlation** (``F``, ``α``) avoids the
    ``∂/∂L ∝ L`` trap of a raw ``LLᵀ + diag(D)`` factor: ``α`` is first-order in the
    loss, can be initialized to a non-trivial value, and is easy to ablate
    (``α → 0`` recovers a diagonal policy).

    ``σ``, ``F``, and ``α`` are state-independent by default (stable for PPO).
    Set ``predict_cov=True`` to predict them from features.

    Args:
        action_dim: Action dimension ``N``.
        rank: Number of correlation factors ``K`` (``1 ≤ K ≤ N``).
        eps: Positive floor on ``σ`` and on column norms of ``F``.
        alpha_init: Initial correlation strength for each factor (softplus target).
        predict_cov: If true, predict ``σ``, ``F``, and ``α`` from features.
    """

    def __init__(
        self,
        action_dim: int,
        rank: int = 2,
        eps: float = 1e-5,
        alpha_init: float = 0.3,
        predict_cov: bool = False,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if rank > action_dim:
            raise ValueError(
                f"rank ({rank}) cannot exceed action_dim ({action_dim})"
            )
        if alpha_init <= 0.0:
            raise ValueError(f"alpha_init must be > 0, got {alpha_init}")
        self.action_dim = action_dim
        self.rank = rank
        self.eps = eps
        self.predict_cov = predict_cov
        self.alpha_init = alpha_init

        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_mean.weight._non_muon = True

        # softplus^{-1}(1) → σ starts at 1 (same exploration scale as Actor).
        scale_init = _softplus_inv(1.0)
        alpha_raw_init = _softplus_inv(alpha_init)
        if predict_cov:
            # σ (N) + F (N·K) + α (K)
            self.cov_head = nn.LazyLinear(action_dim + action_dim * rank + rank)
            self.cov_head.weight._non_muon = True
            self.register_buffer(
                "_scale_offset", torch.tensor(scale_init), persistent=False
            )
            self.register_buffer(
                "_alpha_offset", torch.tensor(alpha_raw_init), persistent=False
            )
        else:
            self.scale_param = nn.Parameter(torch.full((action_dim,), scale_init))
            self.factor_param = nn.Parameter(torch.randn(action_dim, rank))
            self.alpha_param = nn.Parameter(
                torch.full((rank,), alpha_raw_init)
            )

    def _sigma(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return F.softplus(unconstrained) + self.eps

    def _alpha(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return F.softplus(unconstrained)

    def _normalize_factor(self, factor: torch.Tensor) -> torch.Tensor:
        """Column-normalize ``factor`` over the action dimension."""
        return factor / factor.norm(dim=-2, keepdim=True).clamp_min(self.eps)

    def _pack(
        self,
        sigma: torch.Tensor,
        factor: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(cov_factor, cov_diag)`` for ``LowRankMultivariateNormal``."""
        # sigma: (..., N), factor: (..., N, K), alpha: (..., K)
        factor = self._normalize_factor(factor)
        cov_diag = sigma.square()
        cov_factor = sigma.unsqueeze(-1) * factor * alpha.clamp_min(0.0).sqrt().unsqueeze(-2)
        return cov_factor, cov_diag

    @torch.no_grad()
    def cov_stats(self) -> dict[str, torch.Tensor]:
        """Current ``σ`` / ``α`` (state-independent path) for logging."""
        if self.predict_cov:
            return {}
        sigma = self._sigma(self.scale_param)
        alpha = self._alpha(self.alpha_param)
        return {
            "actor/cov/alpha_mean": alpha.mean(),
            "actor/cov/alpha_max": alpha.max(),
            "actor/cov/alpha_min": alpha.min(),
            "actor/cov/sigma_mean": sigma.mean(),
            "actor/cov/sigma_min": sigma.min(),
            "actor/cov/sigma_max": sigma.max(),
        }

    def forward(self, features: torch.Tensor):
        """Return ``(loc, cov_factor, cov_diag)`` for ``LowRankMultivariateNormal``."""
        loc = self.actor_mean(features)
        batch_shape = loc.shape[:-1]

        if self.predict_cov:
            raw = self.cov_head(features)
            n, k = self.action_dim, self.rank
            scale_raw, factor_flat, alpha_raw = raw.split([n, n * k, k], dim=-1)
            sigma = self._sigma(scale_raw + self._scale_offset)
            factor = factor_flat.reshape(*batch_shape, n, k)
            alpha = self._alpha(alpha_raw + self._alpha_offset)
        else:
            sigma = self._sigma(self.scale_param).expand(*batch_shape, -1)
            factor = self.factor_param.expand(*batch_shape, -1, -1)
            alpha = self._alpha(self.alpha_param).expand(*batch_shape, -1)

        cov_factor, cov_diag = self._pack(sigma, factor, alpha)
        return loc, cov_factor, cov_diag
