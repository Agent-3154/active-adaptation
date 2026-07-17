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


class ActorCov(nn.Module):
    """Maps backbone features to a low-rank-plus-diagonal Gaussian.

    Covariance::

        Σ = L Lᵀ + diag(D),   L ∈ R^{N×K}, K ≪ N,
        Dᵢ = softplus(uᵢ) + ε.

    Returns ``(loc, cov_factor, cov_diag)`` for
    ``torch.distributions.LowRankMultivariateNormal``.

    Stability choices (aligned with the default diagonal :class:`Actor`):

    * ``L`` and ``u`` are **state-independent** by default — shared noise is much
      easier to optimize with on-policy methods than a state-dependent SPD matrix.
    * ``L`` is initialized to **0**, so training starts as a unit-variance
      diagonal Gaussian and correlations are introduced only when useful.
    * The diagonal uses **softplus + ε** (not ``exp``) to stay SPD without the
      exploding-std failure mode of unconstrained log-std heads.
    * ``L`` is scaled by ``1/√K`` so the low-rank term's magnitude does not grow
      automatically with rank.

    Args:
        action_dim: Action dimension ``N``.
        rank: Low-rank factor width ``K`` (must satisfy ``1 ≤ K ≤ N``).
        eps: Positive floor added to each diagonal entry.
        predict_cov: If true, predict ``L`` and ``u`` from features instead of
            using shared parameters.
    """

    def __init__(
        self,
        action_dim: int,
        rank: int = 2,
        eps: float = 1e-5,
        predict_cov: bool = False,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if rank > action_dim:
            raise ValueError(
                f"rank ({rank}) cannot exceed action_dim ({action_dim})"
            )
        self.action_dim = action_dim
        self.rank = rank
        self.eps = eps
        self.predict_cov = predict_cov
        self._factor_scale = 1.0 / math.sqrt(rank)

        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_mean.weight._non_muon = True

        # softplus^{-1}(1) so initial diagonal variance matches Actor's std=1.
        diag_init = math.log(math.e - 1.0)
        if predict_cov:
            self.cov_head = nn.LazyLinear(action_dim * rank + action_dim)
            self.cov_head.weight._non_muon = True
            # Added to the unconstrained diagonal so a near-zero head output
            # still yields unit variance before the head is trained.
            self.register_buffer(
                "_diag_offset", torch.tensor(diag_init), persistent=False
            )
        else:
            self.cov_factor = nn.Parameter(torch.zeros(action_dim, rank))
            self.cov_diag_param = nn.Parameter(
                torch.full((action_dim,), diag_init)
            )

    def _diagonal(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return F.softplus(unconstrained) + self.eps

    def forward(self, features: torch.Tensor):
        """Return ``(loc, cov_factor, cov_diag)`` for ``LowRankMultivariateNormal``."""
        loc = self.actor_mean(features)
        batch_shape = loc.shape[:-1]

        if self.predict_cov:
            raw = self.cov_head(features)
            factor_flat, diag_raw = raw.split(
                [self.action_dim * self.rank, self.action_dim], dim=-1
            )
            cov_factor = factor_flat.reshape(
                *batch_shape, self.action_dim, self.rank
            )
            cov_diag = self._diagonal(diag_raw + self._diag_offset)
        else:
            cov_factor = self.cov_factor.expand(*batch_shape, -1, -1)
            cov_diag = self._diagonal(self.cov_diag_param).expand(
                *batch_shape, -1
            )

        cov_factor = cov_factor * self._factor_scale
        return loc, cov_factor, cov_diag
