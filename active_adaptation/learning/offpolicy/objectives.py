from typing import Any, Optional

import torch
import torch.nn as nn
import einops
from tensordict import TensorDictBase
from torchrl.objectives import hold_out_net
from jaxtyping import Float, Bool
from active_adaptation.learning.ppo.common import ACTION_KEY, OBS_KEY
from active_adaptation.learning.offpolicy.distribution import ScaledTanhNormal


def _policy_dist(actor: nn.Module, obs: torch.Tensor) -> Any:
    """Gaussian in pre-tanh space + tanh upscale (same as rollout :class:`ScaledTanhNormal`)."""
    loc, scale = actor(obs)
    return ScaledTanhNormal(loc, scale, upscale=actor.upscale)


class MultiStepReturn(nn.Module):
    def __init__(self, gamma: float, n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.register_buffer("gamma", torch.tensor(gamma))
        self.gamma: torch.Tensor

    def forward(
        self,
        next_observations: Float[torch.Tensor, "T N obs_dim"],
        rewards: Float[torch.Tensor, "T N 1"],
        terminated: Float[torch.Tensor, "T N 1"],
        done: Float[torch.Tensor, "T N 1"],
        env_discount: Optional[Float[torch.Tensor, "T N 1"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "N obs_dim"],
        Float[torch.Tensor, "N 1"],
        Float[torch.Tensor, "N 1"],
        Bool[torch.Tensor, "N 1"],
    ]:
        T, N = next_observations.shape[:2]
        assert T == self.n_steps

        device = rewards.device
        dtype = rewards.dtype
        gamma = self.gamma.to(device=device, dtype=dtype)

        cum_not_done = (~done).cumprod(dim=0)

        step_discount = torch.full((T, N), gamma, device=device, dtype=dtype)
        if env_discount is not None:
            step_discount = step_discount * env_discount.squeeze(-1).to(
                device=device, dtype=dtype
            )

        gammas_cumprod = step_discount.cumprod(dim=0)
        gammas_pow = torch.cat(
            [torch.ones(1, N, device=device, dtype=dtype), gammas_cumprod[:-1]],
            dim=0,
        ).unsqueeze(-1)

        cum_reward = (rewards * gammas_pow).cumsum(dim=0)
        alive_steps = cum_not_done.sum(dim=0)

        last_indices = alive_steps.clamp_max(self.n_steps - 1).reshape(N)
        batch_indices = torch.arange(N, device=device)

        next_observations = next_observations[last_indices, batch_indices]
        rewards_out = cum_reward[last_indices, batch_indices]
        terminated = terminated[last_indices, batch_indices]

        discount = (
            gammas_cumprod[last_indices, batch_indices]
            .reshape_as(terminated.float())
            * (1.0 - terminated.float())
        )

        return next_observations, rewards_out, discount, terminated


def _actor_q_per_sample(critic: nn.Module, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Scalar Q(twin mean) per leading batch row via :meth:`~TwinQNetwork.get_values`."""
    with hold_out_net(critic):
        v = critic.get_values(obs, act)
        q = v.mean(dim=-1)
    return q.reshape(obs.shape[0])


def _per_row_action_nll(dist: Any, act: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Negative log-density of ``act`` under ``dist``, one scalar per batch row."""
    lp = dist.log_prob(act)
    assert lp.shape == act.shape[:-1], (lp.shape, act.shape)
    nll = -lp.reshape(batch_size, -1).sum(dim=-1)
    return nll


class SACLoss:
    """Vanilla SAC actor loss on Q only: minimize ``-E[Q(s,a)]`` for ``a = rsample()`` from ``actor``. Entropy / ``alpha`` is added outside.

    Optional **behavior anchoring**: when ``behavior_coef > 0`` and ``tensordict`` contains
    :data:`~active_adaptation.learning.ppo.common.ACTION_KEY` (replay actions), adds
    ``behavior_coef * (-log pi(a_buf|s))``. That penalizes moving mass away from behavioral
    actions and curbs exploiting a miscalibrated Q (AWAC / BC-regularized actor style).

    Returns per-batch ``q_term`` (``-Q`` plus behavior term if enabled), ``entropy_est`` =
    ``-log pi(a|s)`` for the reparameterized sample, the policy ``dist``, and ``act``.
    """

    def __init__(self, behavior_coef: float = 0.0):
        if behavior_coef < 0:
            raise ValueError("behavior_coef must be >= 0.")
        self.behavior_coef = behavior_coef

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        b = obs.shape[0]
        dist = _policy_dist(actor, obs)
        act = dist.rsample()
        q = _actor_q_per_sample(critic, obs, act)
        entropy_est = -dist.log_prob(act)
        q_term = -q
        if self.behavior_coef > 0 and ACTION_KEY in tensordict:
            act_buf = tensordict[ACTION_KEY]
            q_term = q_term + self.behavior_coef * _per_row_action_nll(dist, act_buf, b)
        return q_term, entropy_est, dist, act


class RatioLoss:
    """Advantage-weighted log-likelihood: ``-(Q(s,a') - Q(s,a_b))_stopgrad * log pi(a'|s)`` with ``a' \sim \pi(\cdot|s)`` and ``a_b`` from the buffer."""

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        act_buf = tensordict[ACTION_KEY]
        dist = _policy_dist(actor, obs)
        act_new = dist.rsample()
        logp = dist.log_prob(act_new)
        assert logp.shape == act_new.shape[:-1]
        logp = logp.reshape(obs.shape[0])
        q_buf = _actor_q_per_sample(critic, obs, act_buf)
        q_new = _actor_q_per_sample(critic, obs, act_new)
        adv = (q_new - q_buf).detach()
        entropy_est = -dist.log_prob(act_new)
        return -(adv * logp), entropy_est


class AdvantageWeightedRegression:
    """AWR-style advantage-weighted log-likelihood (sample MC baseline, no learned ``V``).

    Draw ``num_candidates`` actions from ``\\pi(\\cdot|s)``. Estimate **sample-based**
    ``\\hat V(s)``: mean of ``Q(s,a_k)`` over candidates, and if replay ``ACTION_KEY`` is
    present, include ``Q(s, a_{buf})`` in that mean.

    Let ``A(s,a_k) = Q(s,a_k) - \\hat V(s)`` (stopgrad). The policy term matches the usual
    AWR form (up to Monte Carlo samples from ``\\pi``)::

        ``E_k [ \\exp(A_k / \\beta) \\log \\pi(a_k|s) ]``

    minimized as ``-mean_k [ \\exp(A_k / \\beta) \\log \\pi(a_k|s) ]``.

    Here ``\\beta`` is the constructor arg ``temperature`` (Hydra: ``wr_temperature``).
    ``\\exp`` is stabilized by subtracting ``\\max_k A_k/\\beta`` per state before exponentiation
    (constant factor across ``k``; unchanged relative weighting).

    Optional per-state standardization of ``A`` across ``k`` (``normalize_advantage``) is applied
    before scaling by ``\\beta``.

    Same optional behavior anchoring as :class:`SACLoss` when ``behavior_coef > 0``.

    Returns ``policy_term``, ``entropy_est`` = mean over ``k`` of ``-log \\pi(a_k|s)``, ``dist``,
    ``act_diag`` = first candidate.
    """

    def __init__(
        self,
        num_candidates: int = 8,
        temperature: float = 1.0,
        normalize_advantage: bool = False,
    ):
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature (AWR beta) must be > 0.")
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.normalize_advantage = normalize_advantage

    def compute(
        self, tensordict: TensorDictBase, actor: nn.Module, critic: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        obs = tensordict[OBS_KEY]
        dist = _policy_dist(actor, obs)
        k = self.num_candidates
        if k == 1 and ACTION_KEY not in tensordict:
            raise ValueError(
                "AdvantageWeightedRegression with num_candidates=1 requires replay "
                f"{ACTION_KEY!r} on the tensordict (for a non-trivial MC baseline)."
            )

        cand_act = dist.rsample((k,))  # [K, N, act_dim]
        logp = dist.log_prob(cand_act)
        assert logp.shape == cand_act.shape[:-1]
        logp = einops.rearrange(logp, "sample batch -> batch sample")
        act_flat = einops.rearrange(cand_act, "sample batch act -> (batch sample) act")
        obs_flat = einops.repeat(obs, "batch obs -> (batch sample) obs", sample=k)
        q_flat = _actor_q_per_sample(critic, obs_flat, act_flat)
        q = einops.rearrange(q_flat, "(batch sample) -> batch sample", sample=k)

        baseline = q.mean(dim=-1)
        adv = (q - baseline.unsqueeze(-1)).detach()
        if self.normalize_advantage:
            adv = adv / adv.std(dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = adv / self.temperature
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        w = torch.exp(scaled)
        policy_term = -(w * logp).mean(dim=-1) - q_flat.mean(dim=-1)
        entropy_est = (-logp).mean(dim=-1)
        act_diag = cand_act[0]
        return policy_term, entropy_est, dist, act_diag


# Backward-compatible alias.
WeightedRegression = AdvantageWeightedRegression
