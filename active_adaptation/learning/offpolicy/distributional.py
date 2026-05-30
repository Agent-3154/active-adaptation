"""Distributional RL helpers (C51-style discrete value atoms) and off-policy Q critics.

:func:`project_categorical_bellman` performs the categorical Bellman backup onto a fixed atom
support. Callers typically pass target-network logits at :math:`(s', a')` with folded rewards and
discount. Concrete critic wrappers (:class:`ScalarCritic`, :class:`C51Critic`, etc.) use this API
along with :func:`expected_from_logits` / :func:`cvar_from_logits`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override


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
    z_b = z.expand_as(p)

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


def _huber(x: torch.Tensor, kappa: float) -> torch.Tensor:
    abs_x = x.abs()
    return torch.where(abs_x <= kappa, 0.5 * x.square(), kappa * (abs_x - 0.5 * kappa))


def _pairwise_quantile_huber_loss_per_example(
    u: torch.Tensor, taus: torch.Tensor, kappa: float
) -> torch.Tensor:
    """Quantile Huber loss averaged over (online quantile × target quantile); one scalar per batch row.

    Args:
        u: ``[B, N, Np]`` with ``u[b, n, j] = T_j - Z_n``.
        taus: Online quantile levels ``[B, N]``.
        kappa: Huber threshold.
    Returns:
        ``[B]`` loss per example.
    """
    hub = _huber(u, kappa) / kappa
    ind = (u.detach() < 0).to(hub.dtype)
    taus_e = taus.unsqueeze(-1)
    w = (taus_e - ind).abs()
    return (w * hub).mean(dim=(1, 2))


def _twin_quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Twin-head quantile Huber loss; ``pred`` ``[B, N, 2]``, ``target`` ``[B, N_p]``, ``tau`` ``[B, N]`` -> ``[B]``."""
    z1 = pred[..., 0]
    z2 = pred[..., 1]
    u1 = target.unsqueeze(1) - z1.unsqueeze(2)
    u2 = target.unsqueeze(1) - z2.unsqueeze(2)
    l1 = _pairwise_quantile_huber_loss_per_example(u1, tau, kappa)
    l2 = _pairwise_quantile_huber_loss_per_example(u2, tau, kappa)
    return 0.5 * (l1 + l2)


class CriticBase(ABC):
    @abstractmethod
    def get_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:  # [..., 2]
        """return the Q values"""

    @abstractmethod
    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target Q object for the given inputs."""

    @abstractmethod
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the given target (no aggregation over batch)"""


class ScalarCritic(nn.Module, CriticBase):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.module(obs, act)

    @override
    def get_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # [..., 2]
        return self(obs, act)

    @override
    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """return the scalar Q values"""
        qs: torch.Tensor = self(next_obs, next_act)
        q = qs.min(dim=-1, keepdim=True).values
        return reward + discount * q

    @override
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).square().sum(dim=-1)


class C51Critic(nn.Module, CriticBase):
    def __init__(
        self,
        module: nn.Module,
        v_min: float,
        v_max: float,
        num_atoms: int,
    ):
        super().__init__()
        self.module = module
        q_support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("q_support", q_support)
        self.q_support: torch.Tensor

    def _twin_atom_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split twin C51 logits into two ``[..., num_atoms]`` tensors."""
        n_atom = self.q_support.shape[0]
        if logits.shape[-1] == 2 * n_atom:
            return logits.chunk(2, dim=-1)
        if logits.shape[-1] == 2 and logits.shape[-2] == n_atom:
            return logits[..., 0], logits[..., 1]
        raise ValueError(
            f"Expected logits [..., {n_atom}, 2] or [..., {2 * n_atom}], "
            f"got shape {tuple(logits.shape)}."
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """return the C51-style logits"""
        logits = self.module(obs, act)
        return logits

    @override
    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projected categorical backup; keep the twin head with lower post-projection mean."""
        logits = self(next_obs, next_act)
        l1, l2 = self._twin_atom_logits(logits)
        p1 = project_categorical_bellman(l1, reward, discount, self.q_support)
        p2 = project_categorical_bellman(l2, reward, discount, self.q_support)
        z = self.q_support.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        ev1 = (p1 * z).sum(dim=-1, keepdim=True)
        ev2 = (p2 * z).sum(dim=-1, keepdim=True)
        return torch.where(ev1 < ev2, p1, p2)

    def expected_values(
        self,
        logits: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the expected values from logits"""
        l1, l2 = self._twin_atom_logits(logits)
        if risk_alpha is not None:
            e1 = cvar_from_logits(l1, self.q_support, risk_alpha)
            e2 = cvar_from_logits(l2, self.q_support, risk_alpha)
        else:
            e1 = expected_from_logits(l1, self.q_support)
            e2 = expected_from_logits(l2, self.q_support)
        return torch.cat([e1, e2], dim=-1)

    @override
    def get_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute both twin values from logits and support"""
        logits = self(obs, act)
        return self.expected_values(logits, risk_alpha)

    @override
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of both twins vs the same categorical target ``[B, num_atoms]``."""
        l1, l2 = self._twin_atom_logits(pred)
        target = target.to(dtype=l1.dtype)
        log_p1 = F.log_softmax(l1, dim=-1).clamp(min=-30.0)
        log_p2 = F.log_softmax(l2, dim=-1).clamp(min=-30.0)
        # Match distributional SAC: sum of CE(target, softmax(l_k)) over twins (per-batch row).
        return -((target * log_p1).sum(dim=-1) + (target * log_p2).sum(dim=-1))


class IQNCritic(nn.Module, CriticBase):
    """Twin implicit-quantile critic (IQN; arXiv:1806.06923).

    The injected ``module`` maps ``(s, a)`` and quantile levels ``\\tau`` to twin value samples
    (e.g. a FiLM-conditioned trunk: embed ``\\tau`` and pass it as ``cond`` to
    :class:`~active_adaptation.learning.modules.common.ConditionalBlock`). It must satisfy::

        forward(obs, act, tau) -> [B, N, 2]

    with ``obs`` ``[B, *]``, ``act`` ``[B, *]``, ``tau`` ``[B, N]`` in ``(0, 1)``. Head ``2`` is
    the clipped-double pair at the same ``\\tau``.

    **Tensors**

    - ``module(...)`` returns ``[B, N, 2]``; ``tau`` is ``[B, N]`` per batch row.
    - :meth:`forward` is the training entry point: stores ``tau`` for :meth:`compute_loss` (call on
      the **online** net).
    - :meth:`compute_target` (on the **target** net): i.i.d. ``\\tau'``, shape
      ``[B, N_{\\mathrm{target}}]``, then :math:`T_j = r + \\gamma \\min_k Z_k(s',a',\\tau'_j)`.
    - :meth:`compute_loss`: pairwise quantile Huber (Dabney et al.).

    **Policy values:** :meth:`get_values` averages over a uniform grid in ``(0, 1)`` with
    ``n_quantiles_expectation`` interior points (bootstrap still uses random ``\\tau'`` only).
    """

    _last_tau: torch.Tensor | None

    def __init__(
        self,
        module: nn.Module,
        n_quantiles_target: int = 32,
        n_quantiles_expectation: int = 32,
        huber_kappa: float = 1.0,
        tau_eps: float = 1e-6,
    ):
        super().__init__()
        self.module = module
        if n_quantiles_target < 1:
            raise ValueError("n_quantiles_target must be >= 1")
        if n_quantiles_expectation < 1:
            raise ValueError("n_quantiles_expectation must be >= 1")
        if huber_kappa <= 0:
            raise ValueError("huber_kappa must be positive")
        if tau_eps < 0 or tau_eps >= 0.5:
            raise ValueError("tau_eps must be in [0, 0.5)")

        self.n_quantiles_target = n_quantiles_target
        self.n_quantiles_expectation = n_quantiles_expectation
        self.huber_kappa = huber_kappa
        self.tau_eps = tau_eps
        self._last_tau = None

    def quantile_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to ``module``; must return ``[B, N, 2]`` for ``tau`` ``[B, N]``."""
        return self.module(obs, act, tau)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Evaluate :meth:`quantile_values` and record ``tau`` for the subsequent :meth:`compute_loss`."""
        if tau.dim() != 2 or tau.shape[0] != obs.shape[0]:
            raise ValueError(
                f"tau must be [B, N] with B={obs.shape[0]}, got shape {tuple(tau.shape)}"
            )
        self._last_tau = tau
        return self.quantile_values(obs, act, tau)

    def _sample_target_tau(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand(batch, self.n_quantiles_target, device=device, dtype=dtype)
        if self.tau_eps > 0:
            u = u.mul(1.0 - 2.0 * self.tau_eps).add(self.tau_eps)
        return u

    def _expected_values_pair(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        tau = torch.linspace(
            0.0,
            1.0,
            steps=self.n_quantiles_expectation + 2,
            device=obs.device,
            dtype=obs.dtype,
        )[1:-1]
        tau = tau.unsqueeze(0).expand(obs.shape[0], -1)
        z = self.quantile_values(obs, act, tau)
        return z.mean(dim=1)

    @override
    def get_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Mean return per twin head, shape ``[..., 2]`` (uniform ``\\tau`` grid in ``(0,1)``)."""
        if act.dim() == 2:
            return self._expected_values_pair(obs, act)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_exp = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, obs.shape[-1])
            act_flat = act.reshape(b * k, act.shape[-1])
            ev = self._expected_values_pair(obs_exp, act_flat)
            return ev.reshape(b, k, 2)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")

    @override
    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """One-step distributional backup samples ``[B, N_{target}]``; use on the **target** network."""
        with torch.no_grad():
            b = next_obs.shape[0]
            tau_t = self._sample_target_tau(b, next_obs.device, next_obs.dtype)
            z = self.quantile_values(next_obs, next_act, tau_t)
            z_min = z.min(dim=-1).values
            r = reward.reshape(b, 1).to(dtype=z_min.dtype)
            d = discount.reshape(b, 1).to(dtype=z_min.dtype)
            return r + d * z_min

    @override
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quantile Huber loss; ``pred`` ``[B, N, 2]``, ``target`` ``[B, N_p]``."""
        if self._last_tau is None or self._last_tau.shape[0] != pred.shape[0]:
            raise RuntimeError(
                "Call IQNCritic.forward(obs, act, tau) on the same batch before compute_loss "
                "so quantile levels are recorded."
            )
        return _twin_quantile_huber_loss(pred, target, self._last_tau, self.huber_kappa)


class QRCritic(nn.Module, CriticBase):
    """Twin quantile regression with **fixed** τ midpoints (QR-DQN-style; arXiv:1710.10044).

    Same forward contract as :class:`IQNCritic`: ``module(obs, act, tau) -> [B, N, 2]`` with
    ``\\tau`` / FiLM handled **inside** ``module`` (e.g. τ embedding as ``cond`` for
    :class:`~active_adaptation.learning.modules.common.ConditionalBlock`).

    **Differences from IQN**

    - Training: ``forward(obs, act)`` only; uses ``\\tau_i = (i + \\tfrac12) / N_{\\mathrm{online}}``.
    - Bootstrap: fixed ``tau_target`` on the target net.
    - :meth:`get_values`: mean over ``tau_online`` (not an auxiliary linspace).
    """

    tau_online: torch.Tensor
    tau_target: torch.Tensor
    _last_tau: torch.Tensor | None

    def __init__(
        self,
        module: nn.Module,
        n_quantiles_online: int = 32,
        n_quantiles_target: int | None = None,
        huber_kappa: float = 1.0,
    ):
        super().__init__()
        self.module = module
        if n_quantiles_online < 1:
            raise ValueError("n_quantiles_online must be >= 1")
        n_tgt = n_quantiles_target if n_quantiles_target is not None else n_quantiles_online
        if n_tgt < 1:
            raise ValueError("n_quantiles_target must be >= 1")
        if huber_kappa <= 0:
            raise ValueError("huber_kappa must be positive")

        self.n_quantiles_online = n_quantiles_online
        self.n_quantiles_target = n_tgt
        self.huber_kappa = huber_kappa
        self._last_tau = None

        def _midpoint_tau_grid(n: int, *, device: torch.device | None = None) -> torch.Tensor:
            return (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / float(n)

        self.register_buffer("tau_online", _midpoint_tau_grid(n_quantiles_online))
        self.register_buffer("tau_target", _midpoint_tau_grid(n_tgt))

    def quantile_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to ``module``; must return ``[B, N, 2]`` for ``tau`` ``[B, N]``."""
        return self.module(obs, act, tau)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """``[B, N_{online}, 2]`` at fixed ``tau_online``; records ``tau`` for :meth:`compute_loss`."""
        b = obs.shape[0]
        tau = self.tau_online.to(device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(b, -1)
        self._last_tau = tau
        return self.quantile_values(obs, act, tau)

    def _expected_values_pair(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        tau = (
            self.tau_online.to(device=obs.device, dtype=obs.dtype)
            .unsqueeze(0)
            .expand(obs.shape[0], -1)
        )
        z = self.quantile_values(obs, act, tau)
        return z.mean(dim=1)

    @override
    def get_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Mean over ``tau_online`` per twin; shape ``[..., 2]``."""
        if act.dim() == 2:
            return self._expected_values_pair(obs, act)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_exp = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, obs.shape[-1])
            act_flat = act.reshape(b * k, act.shape[-1])
            ev = self._expected_values_pair(obs_exp, act_flat)
            return ev.reshape(b, k, 2)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")

    @override
    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, N_{target}]`` backups on fixed ``tau_target``; use on the **target** network."""
        with torch.no_grad():
            b = next_obs.shape[0]
            tau_t = (
                self.tau_target.to(device=next_obs.device, dtype=next_obs.dtype)
                .unsqueeze(0)
                .expand(b, -1)
            )
            z = self.quantile_values(next_obs, next_act, tau_t)
            z_min = z.min(dim=-1).values
            r = reward.reshape(b, 1).to(dtype=z_min.dtype)
            d = discount.reshape(b, 1).to(dtype=z_min.dtype)
            return r + d * z_min

    @override
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._last_tau is None or self._last_tau.shape[0] != pred.shape[0]:
            raise RuntimeError(
                "Call QRCritic.forward(obs, act) on the same batch before compute_loss."
            )
        return _twin_quantile_huber_loss(pred, target, self._last_tau, self.huber_kappa)
