"""Distributional RL helpers (C51-style discrete value atoms).

A small ``ValueDistribution`` bundles logits over a fixed 1-D support. That pairs
naturally with Bellman projection: the tuple is "whatever the target network
outputs at (s', a')", and :meth:`ValueDistribution.project` maps the softmax
through the categorical backup onto the same atom grid.

Keeping :func:`project_categorical_bellman` as a pure function avoids duplication
in critics (SAC, TD3, etc.) while keeping call sites explicit when you already
have loose tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import NamedTuple


def expected_from_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Expected scalar Q under a softmax over atoms: logits [B, N] -> [B, 1]."""
    p = F.softmax(logits, dim=-1)
    z = support.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    return (p * z).sum(dim=-1, keepdim=True)


def _cvar_tail_from_probs(
    p: torch.Tensor,
    z: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    """One-sided conditional mean for a discrete law; see :func:`cvar_from_logits`.

    Args:
        p: Probabilities ``[..., N]`` (row-stochastic last dim).
        z: Atom values ``[..., N]``, same shape as ``p``.
        mass: Tail probability per batch slice, broadcastable to ``p.shape[:-1]``, entries in ``(0, 1]``.
    """
    # p, z: [..., N]; mass: [...] broadcastable to p.shape[:-1], in (0, 1]
    n = p.shape[-1]
    cp = p.cumsum(dim=-1)
    czp = (p * z).cumsum(dim=-1)

    idx = torch.searchsorted(
        cp.reshape(-1, n),
        mass.reshape(-1, 1),
        right=False,
    ).reshape(mass.shape)
    idx = idx.clamp(max=n - 1)

    idx_prev = (idx - 1).clamp(min=0)
    # At idx==0, cp_prev and czp_prev must be 0 (ignore gather at -1).
    mask_prev = idx > 0
    cp_prev = cp.gather(-1, idx_prev.unsqueeze(-1)).squeeze(-1)
    cp_prev = torch.where(mask_prev, cp_prev, cp.new_zeros(()))
    czp_prev = czp.gather(-1, idx_prev.unsqueeze(-1)).squeeze(-1)
    czp_prev = torch.where(mask_prev, czp_prev, czp.new_zeros(()))
    z_k = z.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    numer = czp_prev + (mass - cp_prev).clamp(min=0.0) * z_k
    return (numer / mass).unsqueeze(-1)


def cvar_from_logits(
    logits: torch.Tensor,
    support: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Conditional tail mean of the return implied by ``softmax(logits)`` on ``support``.

    The support must be **non-decreasing** (e.g. C51 grid ``v_min … v_max``). Let :math:`Z`
    be the random return with :math:`P(Z=z_i)=p_i`.

    * **Risk-averse (left tail):** entries with ``0 < alpha <= 1``. CVaR is the expectation of
      :math:`Z` over the worst :math:`\\alpha` mass (smallest outcomes first). For ``alpha == 1``
      this matches :func:`expected_q_from_logits` (per element).

    * **Risk-seeking (right tail):** entries with ``-1 < alpha < 0``. Uses tail mass
      :math:`\\beta=-\\alpha` from the **largest** outcomes (right tail conditional mean).

    ``alpha`` is a tensor broadcastable to ``logits.shape[:-1]`` (or a Python float). Per-element
    signs may differ across the batch.

    Args:
        logits: Raw scores, shape ``[..., N]`` (last dim matches atoms).
        support: Atom locations ``z_0 \\le … \\le z_{N-1}``, shape ``[N]``.
        alpha: Tail level(s), broadcastable to ``[...,]`` (same as ``logits`` without the atom
            axis). Each entry must lie in ``(0, 1]`` (left CVaR) or ``(-1, 0)`` (right CVaR).

    Returns:
        Tensor shaped ``logits.shape[:-1] + (1,)`` with the tail conditional mean per batch slice.
    """
    p = F.softmax(logits, dim=-1)
    z = support.to(device=logits.device, dtype=logits.dtype)
    if z.ndim != 1:
        raise ValueError(f"support must be 1-D, got shape {tuple(z.shape)}")
    n = z.shape[0]
    if logits.shape[-1] != n:
        raise ValueError(
            f"logits last dim {logits.shape[-1]} != len(support) {n}"
        )

    batch_shape = logits.shape[:-1]
    alpha_t = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype)
    try:
        alpha_t = alpha_t.broadcast_to(batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"alpha with shape {tuple(torch.as_tensor(alpha).shape)} is not broadcastable "
            f"to logits batch shape {tuple(batch_shape)}."
        ) from e

    valid = ((alpha_t > 0) & (alpha_t <= 1)) | ((alpha_t < 0) & (alpha_t > -1))
    if not valid.all():
        raise ValueError(
            "alpha entries must be in (0, 1] for left-tail CVaR or in (-1, 0) for right-tail CVaR."
        )

    # Broadcast z to p's leading dims: [..., N]
    z_b = z
    # while z_b.ndim < p.ndim:
    #     z_b = z_b.unsqueeze(0)
    z_b = z_b.expand_as(p)

    mass = alpha_t.abs()
    out_left = _cvar_tail_from_probs(p, z_b, mass)
    out_right = _cvar_tail_from_probs(p.flip(-1), z_b.flip(-1), mass)
    return torch.where(alpha_t.unsqueeze(-1) > 0, out_left, out_right)


def project_categorical_bellman(
    next_logits: torch.Tensor,
    rewards: torch.Tensor,
    discount: torch.Tensor | float,
    support: torch.Tensor,
) -> torch.Tensor:
    """C51 categorical projection for a one-step (or n-step folded) Bellman backup.

    Args:
        next_logits: Target network logits at the bootstrap state, [B, num_atoms].
        rewards: Return term already folded (n-step sum, entropy in reward, etc.), [B] or [B, 1].
        discount: Factor on next-atom values; must include bootstrap mask
            (e.g. ``gamma * (1 - term)`` or :class:`MultiStepReturn` output).
        support: 1-D atom locations, shape [num_atoms], equally spaced.

    Returns:
        Projected target probabilities [B, num_atoms] (non-negative, row-stochastic).
    """
    num_atoms = support.shape[0]
    if num_atoms < 3:
        raise ValueError("support must contain more than two atoms (num_atoms > 2).")

    device = next_logits.device
    dtype = next_logits.dtype
    support = support.to(device=device, dtype=dtype)
    batch_size = next_logits.shape[0]
    if next_logits.shape[-1] != num_atoms:
        raise ValueError(
            f"next_logits last dim {next_logits.shape[-1]} != len(support) {num_atoms}"
        )

    v_lo = support[0]
    v_hi = support[-1]
    delta_raw = (v_hi - v_lo) / (num_atoms - 1)
    # Extremely small spans or dtype noise could make delta_z degenerate / unstable.
    min_delta = torch.finfo(dtype).tiny * torch.tensor(
        256.0, device=device, dtype=dtype
    )
    delta_z = torch.clamp(delta_raw, min=min_delta)

    rewards = rewards.reshape(batch_size, 1).to(dtype=dtype)
    if not isinstance(discount, torch.Tensor):
        discount_t = torch.full((batch_size, 1), float(discount), device=device, dtype=dtype)
    else:
        discount_t = discount.reshape(batch_size, 1).to(dtype=dtype)

    target_z = rewards + discount_t * support.view(1, -1)
    target_z = target_z.clamp(v_lo, v_hi)

    # Continuous index on the support grid: b=0 -> v_lo, b=num_atoms-1 -> v_hi.
    b = (target_z - v_lo) / delta_z
    b_max = float(num_atoms - 1)
    b = torch.nan_to_num(b, nan=0.0, neginf=0.0, posinf=b_max)
    b = b.clamp(0.0, b_max)

    # C51 projection: split each atom's mass between floor(b) and floor(b)+1 (Bellemare et al.).
    # Using adjacent bins avoids the old ceil/floor "same bin" hack that doubled mass on interior
    # grid points; at b = num_atoms-1, upper clamps to the same index and (1-frac)+frac preserves p.
    lower = torch.floor(b).long().clamp(0, num_atoms - 1)
    upper = (lower + 1).clamp(max=num_atoms - 1)
    frac = b - lower.to(dtype=b.dtype)

    next_dist = F.softmax(next_logits, dim=-1)
    m_l = next_dist * (1.0 - frac)
    m_u = next_dist * frac

    proj_dist = next_dist.new_zeros(batch_size, num_atoms)
    proj_dist.scatter_add_(1, lower, m_l)
    proj_dist.scatter_add_(1, upper, m_u)
    return proj_dist


class ValueDistribution(NamedTuple):
    """Softmax distribution over a fixed scalar support (e.g. C51 atoms).

    Typical use: wrap **target** logits at ``(s', a')`` and call :meth:`project`
    with bootstrapped ``rewards`` and ``discount`` to obtain a target categorical
    for cross-entropy training of the online logits at ``(s, a)``.
    """

    logits: torch.Tensor # [..., num_atoms]
    support: torch.Tensor # [num_atoms]

    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    def expected_value(self) -> torch.Tensor:
        return expected_from_logits(self.logits, self.support)

    def project(
        self,
        rewards: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> torch.Tensor:
        """Bellman projection of ``softmax(logits)`` onto ``support``."""
        return project_categorical_bellman(
            self.logits, rewards, discount, self.support
        )
