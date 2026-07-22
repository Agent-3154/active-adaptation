"""Long-Horizon Q-Learning (LQL) with a Soft Actor-Critic policy.

References
----------
Abraham, Shi, Finn. *Long-Horizon Q-Learning: Accurate Value Learning via
n-Step Inequalities*. arXiv:2605.05812.
Code: https://github.com/armaan-abraham/lql.

This module ports the single-action LQL critic into the ``train_offpolicy``
contract used by :mod:`sac`. Actor / temperature updates follow SAC. Critic
bootstraps use ``v_next``; set ``entropy_bonus>0`` to add a soft entropy term
as in :mod:`sac` (``0`` recovers the paper's hard reward-return TD).

Core idea
---------
Standard 1-step TD is kept, and trajectory-level *optimality inequalities*
are enforced with asymmetric squared hinges. Any logged action sequence lower-
bounds what the optimal policy can achieve; violations of that ordering are
penalized without changing the bootstrap action interface.

Notation for a contiguous replay segment of length ``L``
(``trajectory_length``)::

    G_{i:j}  = sum_{u=i}^{j-1} gamma^{u-i} r_u     # discounted partial return
    a*_s     = sample from the (target) actor at s  # continuous maximizer proxy
    Q, Qbar  = online / Polyak target twin critics

One-step TD (paper Eq. 2; per transition ``k``)::

    ell_TD(k) = ( Q(s_k, a_k) - [r_k + gamma * (1 - term_k) * Qbar(s_{k+1}, a*_{k+1})] )^2

Lower-bound hinge (paper Eq. 6; ``ell > k``, typically skip ``ell = k+1``)::

    delta_LB(k, ell) = [ G_{k:ell} + gamma^{ell-k} Qbar(s_ell, a*_ell) - Q(s_k, a_k) ]_+^2

    If the observed multi-step return plus a later target bootstrap exceeds the
    current Q at ``k``, push ``Q(s_k, a_k)`` upward.

Upper-bound hinge (paper Eq. 8; ``i <= k``, including same-state ``i = k``)::

    delta_UB(i, k) = [ G_{i:k} + gamma^{k-i} Q(s_k, a_k) - Qbar(s_i, a*_i) ]_+^2

    If a later logged-action Q is too large relative to acting near-optimally
    earlier, push that Q downward. The special case ``i = k`` is
    ``[Q(s_k, a_k) - Qbar(s_k, a*_k)]_+^2``.

Total critic loss (paper Eq. 10)::

    ell_LQL(k) = ell_TD(k) + lambda_UB * mean_i delta_UB(i, k)
                           + lambda_LB * mean_ell delta_LB(k, ell)

    L_LQL = (1/L) sum_k ell_LQL(k)

Design of this port
-------------------
* **Scalar twin critics** (:class:`~sac.TwinScalarCritic`); no C51 / IQN.
* **Soft / hard backup via ``entropy_bonus``**: ``v_next`` is
  ``min(Q1_bar, Q2_bar)`` at a target-policy sample; when
  ``entropy_bonus != 0``, add ``entropy_bonus * (-α log π) * entropy_scale``
  (same construction as :meth:`sac.SAC._compute_target`). ``0`` matches the
  paper's hard TD. Actor entropy maximization is always on (SAC).
* **Clipped double Q**: online twin heads share that detached ``v_next`` for
  TD and both hinges.
* **Contiguous trajectories** from :meth:`ReplayBuffer.sample_trajectory`
  (no newest→oldest ring wrap). Episode cuts inside a segment are masked via
  ``done`` / ``terminated`` / ``is_init``.
* **Batch sizing**: ``critic_batch_size`` counts *transitions*; the number of
  trajectories is ``critic_batch_size // trajectory_length``.
* **Complexity**: hinge pairs are ``O(B L^2)`` arithmetic on already-computed
  Q / ``v_next`` (same asymptotic cost as the JAX reference); network
  forwards remain ``O(B L)``.
* **Optimizers / AMP**: optional Muon+AdamW (``muon``) and CUDA FP16 AMP
  (``use_amp``), matching :mod:`sac`. Temperature ``AlphaModule`` stays fp32.
* **Not in v1**: DDP, symmetry aug, action chunking, n-step TD folding,
  distributional critics.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import einops
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from torch.amp import GradScaler, autocast
from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

from active_adaptation.learning.modules import CatTensors, IndependentNormal, VecNorm
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.sac import (
    AlphaModule,
    NormalActor,
    SACRolloutPolicy,
    TwinScalarCritic,
    gaussian_target_entropy,
)
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.utils.profiling import ScopedTimer

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase


cs = ConfigStore.instance()
clip_grad_norm_ = nn.utils.clip_grad_norm_


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def lql_critic_loss(
    q: torch.Tensor,
    v_next: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    terminated: torch.Tensor,
    continuation: torch.Tensor,
    transition_valid: torch.Tensor | None = None,
    *,
    lambda_lb: float = 1.0,
    lambda_ub: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single-action LQL loss.

    Args:
        q: Online twin values ``[num_critics, batch, time]``.
        v_next: Detached target-policy values at each next state ``[batch, time]``.
        rewards: Per-step rewards ``[batch, time]``.
        discounts: Per-step discount before termination masking ``[batch, time]``.
        terminated: True terminal transitions ``[batch, time]``.
        continuation: Whether the recorded trajectory continues after each step.
        transition_valid: Optional validity mask for individual transitions.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [critic,batch,time], got {tuple(q.shape)}")
    num_critics, batch_size, seq_len = q.shape
    expected = (batch_size, seq_len)
    for name, value in {
        "v_next": v_next,
        "rewards": rewards,
        "discounts": discounts,
        "terminated": terminated,
        "continuation": continuation,
    }.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

    if transition_valid is None:
        transition_valid = torch.ones_like(continuation, dtype=torch.bool)
    elif transition_valid.shape != expected:
        raise ValueError(
            f"transition_valid must have shape {expected}, got {tuple(transition_valid.shape)}"
        )

    terminated = terminated.bool()
    continuation = continuation.bool() & ~terminated
    transition_valid = transition_valid.bool()
    v_next = v_next.detach()

    td_target = rewards + discounts * (~terminated).to(rewards.dtype) * v_next
    td_errors = (q - td_target.unsqueeze(0)).square()
    td_mask = transition_valid.unsqueeze(0).expand_as(td_errors)
    td_loss = _masked_mean(td_errors, td_mask)

    zero = q.sum() * 0.0
    lb_error_sum = zero
    ub_error_sum = zero
    lb_valid_count = q.new_zeros(())
    ub_valid_count = q.new_zeros(())
    lb_active_count = q.new_zeros(())
    ub_active_count = q.new_zeros(())

    # Lower bound: Q(s_k,a_k) must dominate following logged actions through
    # j and then bootstrapping with the target policy at s_{j+1}.  j=k is
    # omitted because it duplicates the one-step TD target.
    for k in range(seq_len):
        partial_return = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
        coefficient = torch.ones_like(partial_return)
        path_valid = transition_valid[:, k].clone()
        for j in range(k, seq_len):
            partial_return = partial_return + coefficient * rewards[:, j]
            if j > k:
                pair_valid = path_valid & transition_valid[:, j]
                bound = (
                    partial_return
                    + coefficient
                    * discounts[:, j]
                    * (~terminated[:, j]).to(q.dtype)
                    * v_next[:, j]
                )
                violation = torch.relu(bound.unsqueeze(0) - q[:, :, k])
                mask = pair_valid.unsqueeze(0)
                lb_error_sum = lb_error_sum + (violation.square() * mask).sum()
                lb_valid_count = lb_valid_count + pair_valid.sum()
                lb_active_count = lb_active_count + (
                    (violation > 0) & mask
                ).sum().to(q.dtype) / num_critics
            coefficient = coefficient * discounts[:, j]
            path_valid = path_valid & continuation[:, j]

    # Upper bound: target-policy value after transition i must dominate the
    # return from following logged actions until the later Q(s_k,a_k).
    # k=i+1 is the paper's same-state upper bound.
    for i in range(seq_len - 1):
        partial_return = torch.zeros(batch_size, device=q.device, dtype=q.dtype)
        coefficient = torch.ones_like(partial_return)
        path_valid = transition_valid[:, i] & continuation[:, i]
        for k in range(i + 1, seq_len):
            pair_valid = path_valid & transition_valid[:, k]
            implied = partial_return.unsqueeze(0) + coefficient.unsqueeze(0) * q[:, :, k]
            violation = torch.relu(implied - v_next[:, i].unsqueeze(0))
            mask = pair_valid.unsqueeze(0)
            ub_error_sum = ub_error_sum + (violation.square() * mask).sum()
            ub_valid_count = ub_valid_count + pair_valid.sum()
            ub_active_count = ub_active_count + (
                (violation > 0) & mask
            ).sum().to(q.dtype) / num_critics

            partial_return = partial_return + coefficient * rewards[:, k]
            coefficient = coefficient * discounts[:, k]
            path_valid = path_valid & continuation[:, k]

    lb_loss = lb_error_sum / (lb_valid_count * num_critics).clamp_min(1.0)
    ub_loss = ub_error_sum / (ub_valid_count * num_critics).clamp_min(1.0)
    total = td_loss + float(lambda_lb) * lb_loss + float(lambda_ub) * ub_loss
    # Diagnostics (logged under ``critic/`` by :meth:`LQLSAC.train_critic`):
    # - ``*_loss``: mean squared hinge / TD residual over valid terms.
    # - ``td_target_mean``: average 1-step bootstrap target on valid transitions
    #   (training scale here; denormalized to match ``q_value`` when logged).
    # - ``num_valid_*_terms``: how many mask-accepted TD / hinge pairs entered the mean
    #   (drops near zero when segments are mostly truncated or ``is_init``).
    # - ``*_activation``: fraction of valid LB/UB pairs with a positive hinge
    #   (high ≈ frequent inequality violations; ~0 ≈ constraints already satisfied).
    info = {
        "td_loss": td_loss.detach(),
        "lower_bound_loss": lb_loss.detach(),
        "upper_bound_loss": ub_loss.detach(),
        "td_target_mean": _masked_mean(td_target, transition_valid).detach(),
        "num_valid_td_terms": transition_valid.sum().detach(),
        "num_valid_lower_bound_terms": lb_valid_count.detach(),
        "num_valid_upper_bound_terms": ub_valid_count.detach(),
        "lower_bound_activation": (
            lb_active_count / lb_valid_count.clamp_min(1.0)
        ).detach(),
        "upper_bound_activation": (
            ub_active_count / ub_valid_count.clamp_min(1.0)
        ).detach(),
    }
    return total, info


@dataclass
class LQLSACConfig:
    _target_: str = "active_adaptation.learning.offpolicy.lql_sac.LQLSACConfig"
    name: str = "lql_sac"
    train_every: int = 4
    buffer_size: int = 2000
    warm_up_steps: int = 200
    lr: float = 5e-4
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper`.
    muon: bool = True
    weight_decay: float = 0.02

    trajectory_length: int = 8
    lambda_lb: float = 1.0
    lambda_ub: float = 1.0
    gamma: float = 0.99
    utd_ratio: int = 4

    actor_init: str = "zeros"
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    target_action_noise: float = 0.01
    use_correlated: bool = True

    # Soft TD weight on (-α log π) inside ``v_next`` (sac-style). 0 = hard TD.
    entropy_bonus: float = 1.0
    alpha_init: float = 4e-3
    target_entropy_sigma: float | None = 0.15
    soft_bound: float = 2.0 * math.pi
    tau_actor: float = 0.1
    tau_Q: float = 0.02
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); separate GradScalers for critic and actor (alpha stays fp32).
    use_amp: bool = True
    clamp_reward: bool = True
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8

    prior_data: str | None = None
    prior_data_ratio: float = 0.4
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)

    def get_class(self):
        return LQLSAC

cs.store(name="lql_sac", node=LQLSACConfig, group="algo")


class LQLSAC(TensorDictModuleBase):
    """Paper-faithful, single-action Long-Horizon Q-Learning with a SAC actor."""

    train_keys = (
        CMD_KEY,
        OBS_KEY,
        ("next", OBS_KEY),
        ("next", CMD_KEY),
        ACTION_KEY,
        REWARD_KEY,
        TERM_KEY,
        DONE_KEY,
        ("next", "discount"),
        "is_init",
    )

    def __init__(
        self,
        cfg: LQLSACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device: torch.device | str,
    ):
        super().__init__()
        if cfg.trajectory_length < 1:
            raise ValueError("trajectory_length must be at least one.")
        if cfg.critic_batch_size % cfg.trajectory_length:
            raise ValueError("critic_batch_size must be divisible by trajectory_length.")
        if cfg.lambda_lb < 0 or cfg.lambda_ub < 0:
            raise ValueError("LQL hinge weights must be nonnegative.")

        self.cfg = cfg
        self.device = torch.device(device)
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec

        fake = observation_spec.zero()
        preproc: list[nn.Module] = []
        if CMD_KEY in observation_spec.keys(True, True):
            obs_dim = fake[OBS_KEY].shape[-1] + fake[CMD_KEY].shape[-1]
            preproc.append(CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False))
        else:
            obs_dim = fake[OBS_KEY].shape[-1]
            preproc.append(Mod(nn.Identity(), [OBS_KEY], ["_input"]))

        self.act_dim = action_spec.shape[-1]
        self.entropy_scale = 1.0 / math.sqrt(self.act_dim)
        self.vecnorm_obs = (
            VecNorm(obs_dim).to(self.device)
            if cfg.vecnorm
            else nn.Identity()
        )
        preproc.append(Mod(self.vecnorm_obs, ["_input"], ["_input_normed"]))
        self.preproc = Seq(*preproc).to(self.device)

        self.Q = TwinScalarCritic(obs_dim, self.act_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()

        self.actor = NormalActor(
            obs_dim,
            self.act_dim,
            std_max=1.0,
            std_min=0.001,
            action_init=cfg.actor_init,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.requires_grad_(False)
        self.actor_target.eval()
        self.DistClass = IndependentNormal

        if cfg.target_entropy_sigma is None:
            self.target_entropy = -0.5 * float(self.act_dim)
        else:
            self.target_entropy = gaussian_target_entropy(
                self.act_dim, cfg.target_entropy_sigma
            )
        self.alpha = AlphaModule(cfg.alpha_init).to(self.device)
        self.opt_alpha = torch.optim.Adam(self.alpha.parameters(), lr=cfg.lr_alpha)
        if cfg.muon:
            self.opt_actor = MuonAdamWWrapper(
                [self.actor],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            self.opt_Q = MuonAdamWWrapper(
                [self.Q],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.opt_actor = torch.optim.AdamW(
                self.actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            self.opt_Q = torch.optim.AdamW(
                self.Q.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        self.reward_normalizer: RewardNormalizer | None = None
        if cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=cfg.gamma,
                load_rms=False,
                device=self.device,
                epsilon=cfg.reward_norm_epsilon,
            )

        self._amp_device_type = self.device.type
        self._amp_enabled = bool(cfg.use_amp and self.device.type == "cuda")
        # Separate scalers so critic/actor loss scales and Inf/NaN skips stay independent.
        self.grad_scaler_Q = GradScaler(self._amp_device_type, enabled=self._amp_enabled)
        self.grad_scaler_actor = GradScaler(
            self._amp_device_type, enabled=self._amp_enabled
        )

        self.global_step = 0
        self.rb: ReplayBuffer
        self.rb_prior: ReplayBuffer | None = None

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return autocast(
            device_type=self._amp_device_type,
            dtype=torch.float16,
        )

    def reward_collate_fn(self, reward: torch.Tensor | TensorDict) -> torch.Tensor:
        if isinstance(reward, TensorDict):
            reward = torch.cat(list(reward.values()), dim=-1)
        reward = reward.sum(-1, keepdim=True)
        return reward.clamp_min(0.0) if self.cfg.clamp_reward else reward

    def make_tensordict_primer(self):
        from torchrl.data import BoundedContinuous, Composite, UnboundedContinuous
        from torchrl.envs import TensorDictPrimer

        shape = tuple(self.action_spec.shape)
        spec = {
            "prev_noise": UnboundedContinuous(shape, device=self.device),
            "rho": BoundedContinuous(
                low=0.0,
                high=1.0,
                shape=[shape[0], 1],
                device=self.device,
            ),
        }
        return TensorDictPrimer(
            Composite(spec, shape=[shape[0]], device=self.device),
            random=self.cfg.use_correlated,
            reset_key="done",
            expand_specs=False,
        )

    @classmethod
    def from_env(cls, cfg: LQLSACConfig, env: _EnvBase, device: torch.device):
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
        )

    def get_rollout_policy(
        self, mode: str = "train", critic: bool = False
    ) -> TensorDictModuleBase:
        return SACRolloutPolicy(
            self.preproc,
            self.actor,
            self.DistClass,
            use_correlated=self.cfg.use_correlated,
            Q=self.Q if critic else None,
            reward_normalizer=self.reward_normalizer,
            critic=critic,
        )

    def on_stage_start(self, stage: str, env: _EnvBase):
        fake_rb = env.fake_tensordict().exclude(("next", "stats"), "collector")
        fake_rb["loc"] = torch.zeros(
            fake_rb.shape[0], self.actor.act_dim, device=fake_rb.device
        )
        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys -= {"prev_noise", "rho"}
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size,
            fake_rb,
            fake_bootstrap=True,
            observation_keys=list(observation_keys),
        )
        if self.cfg.prior_data is None:
            self.rb_prior = None
        else:
            self.rb_prior = ReplayBuffer.from_rollout(
                self.cfg.prior_data,
                fake_bootstrap=True,
                observation_keys=list(observation_keys),
            )
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    def step(self, tensordict: TensorDict):
        self.global_step += 1
        td = tensordict.exclude(("next", "stats"), "collector")
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(
                reward=self.reward_collate_fn(td[REWARD_KEY]),
                terminated=td[TERM_KEY],
                truncated=td["next", "truncated"],
            )
        self.rb.push(td)
        if (
            self.global_step > self.cfg.warm_up_steps
            and self.global_step % self.cfg.train_every == 0
        ):
            return self.train_op()
        return {}

    @ScopedTimer("lql_sac_train")
    @VecNorm.freeze()
    def train_op(self):
        infos: dict[str, float | int] = {"rb_size": len(self.rb)}
        trajectories = self.cfg.critic_batch_size // self.cfg.trajectory_length
        critic_iters = self.cfg.train_every * self.cfg.utd_ratio
        for i in range(critic_iters):
            batch = self.rb.sample_trajectory(
                batch_size=trajectories,
                steps=self.cfg.trajectory_length,
                next_obs=True,
            ).to(self.device, non_blocking=True)
            batch_prior = None
            if self.rb_prior is not None:
                prior_trajectories = int(trajectories * self.cfg.prior_data_ratio)
                if prior_trajectories > 0:
                    batch_prior = self.rb_prior.sample_trajectory(
                        batch_size=prior_trajectories,
                        steps=self.cfg.trajectory_length,
                        next_obs=True,
                    ).to(self.device, non_blocking=True)
            info = self.train_critic(
                batch,
                batch_prior=batch_prior,
                diagnostics=i == critic_iters - 1,
            )
            if info:
                infos.update(info)

        for i in range(self.cfg.train_every):
            info = self.train_actor(diagnostics=i == self.cfg.train_every - 1)
            if info:
                infos.update(info)
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _target_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Target-policy bootstrap values for TD and LQL hinges.

        Matches :meth:`sac.SAC._compute_target`: evaluate ``Q`` at a noisy
        action, but form the entropy bonus from ``log π`` of the clean sample.
        With ``entropy_bonus=0`` this is hard ``min Q``; otherwise soft
        ``min Q + entropy_bonus * (-α log π) * entropy_scale``.
        """
        loc, scale = self.actor_target(next_obs)
        dist = self.DistClass(loc, scale)
        action = dist.sample()
        action_q = action
        if self.cfg.target_action_noise:
            noise = torch.randn_like(action).clamp(-3.0, 3.0)
            action_q = action + noise * self.cfg.target_action_noise
        q = self.Q_target.get_values(next_obs, action_q).min(dim=-1).values
        if self.cfg.entropy_bonus:
            log_prob = dist.log_prob(action)
            alpha = self.alpha()
            q = q + (
                float(self.cfg.entropy_bonus)
                * (-alpha * log_prob)
                * self.entropy_scale
            )
        return q

    @ScopedTimer("train_critic")
    def train_critic(
        self,
        batch: TensorDict,
        batch_prior: TensorDict | None = None,
        diagnostics: bool = False,
    ):
        self.Q.train()
        batch = batch.select(*self.train_keys, inplace=True, strict=False)
        online_batch = batch.shape[1]
        if batch_prior is not None:
            batch_prior = batch_prior.select(*self.train_keys, inplace=True, strict=False)
            batch = torch.cat([batch, batch_prior], dim=1)

        reward = self.reward_collate_fn(batch[REWARD_KEY])
        if self.cfg.debug:
            reward = torch.full_like(reward, 1.0 - self.cfg.gamma)
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            reward = reward * (1.0 - self.cfg.gamma)

        self.preproc(batch)
        self.preproc(batch["next"])
        obs = batch["_input_normed"]
        next_obs = batch["next", "_input_normed"]
        action = batch[ACTION_KEY]
        seq_len, batch_size = obs.shape[:2]

        rewards = reward.squeeze(-1).transpose(0, 1)
        terminated = batch[TERM_KEY].squeeze(-1).transpose(0, 1).bool()
        done = batch[DONE_KEY].squeeze(-1).transpose(0, 1).bool()
        is_init = batch["is_init"]
        if is_init.ndim == 3:
            is_init = is_init.squeeze(-1)
        transition_valid = ~is_init.transpose(0, 1).bool()

        env_discount = batch.get(("next", "discount"))
        if env_discount is None:
            discounts = torch.full_like(rewards, self.cfg.gamma)
        else:
            discounts = (
                env_discount.squeeze(-1).transpose(0, 1).to(rewards.dtype)
                * self.cfg.gamma
            )

        with self._autocast():
            obs_flat = obs.reshape(seq_len * batch_size, -1)
            next_obs_flat = next_obs.reshape(seq_len * batch_size, -1)
            action_flat = action.reshape(seq_len * batch_size, -1)
            pred = self.Q.get_values(obs_flat, action_flat)
            q = pred.reshape(seq_len, batch_size, 2).permute(2, 1, 0)
            v_next = (
                self._target_values(next_obs_flat)
                .reshape(seq_len, batch_size)
                .transpose(0, 1)
            )
            q_loss, loss_info = lql_critic_loss(
                q,
                v_next,
                rewards,
                discounts,
                terminated,
                ~done,
                transition_valid,
                lambda_lb=self.cfg.lambda_lb,
                lambda_ub=self.cfg.lambda_ub,
            )

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_Q.scale(q_loss).backward()
            # Unscale before clip / grad norm; scaler.step still checks Inf/NaN.
            self.grad_scaler_Q.unscale_(self.opt_Q)
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_Q.step(self.opt_Q)
            self.grad_scaler_Q.update()
        else:
            q_loss.backward()
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_Q.step()
        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        if not diagnostics:
            return {}
        with torch.no_grad():
            q_online = q[:, :online_batch]
            q_values = q_online.permute(1, 2, 0)
            td_target_mean = loss_info["td_target_mean"]
            if self.reward_normalizer is not None:
                q_values = self.reward_normalizer.denormalize_return_values(q_values)
                td_target_mean = self.reward_normalizer.denormalize_return_values(
                    td_target_mean
                )
            info = {
                "critic/q_loss": q_loss.item(),
                "critic/grad_norm": critic_grad_norm.item(),
                "critic/q_value": q_values.mean().item(),
                "critic/q_max": q_values.max().item(),
                "critic/q_std": q_values.std(dim=-1).mean().item(),
            }
            for key, value in loss_info.items():
                if key == "td_target_mean":
                    info["critic/td_target_mean"] = td_target_mean.item()
                else:
                    info[f"critic/{key}"] = value.item()

            # Terminal transitions should have small Q (bootstrap masked; target ≈ r).
            terminated_online = terminated[:online_batch]
            if terminated_online.any():
                q_terminated = q_values[terminated_online]
                info["critic/q_value_terminated"] = q_terminated.mean().item()
                td_target = rewards + discounts * (~terminated).to(rewards.dtype) * v_next
                td_err = (q_online - td_target[:online_batch].unsqueeze(0)).square()
                info["critic/q_loss_terminated"] = (
                    td_err.mean(dim=0)[terminated_online].mean().item()
                )
        return info

    @ScopedTimer("train_actor")
    def train_actor(self, diagnostics: bool = False):
        batch = (
            self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1)
            .to(self.device)
            .select(*self.train_keys, strict=False)
        )
        batch_prior = None
        if self.rb_prior is not None:
            prior_size = int(self.cfg.actor_batch_size * self.cfg.prior_data_ratio)
            if prior_size > 0:
                batch_prior = (
                    self.rb_prior.sample(batch_size=prior_size, steps=1)
                    .to(self.device)
                    .select(*self.train_keys, strict=False)
                )
                batch = torch.cat([batch, batch_prior], dim=0)

        self.preproc(batch)
        obs = batch["_input_normed"]
        is_init = batch["is_init"]
        n_online = obs.shape[0] - (0 if batch_prior is None else batch_prior.shape[0])

        with hold_out_net(self.Q), self._autocast():
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale)
            action_update = dist.rsample((4,))
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            q = self.Q.get_values(
                obs,
                einops.rearrange(action_update, "sample batch act -> batch sample act"),
            ).mean(dim=-1)
            policy_term = -q.mean(dim=1)

        alpha = self.alpha()
        # ``entropy_bonus`` only scales the critic soft backup (as in sac);
        # the actor always maximizes entropy.
        actor_per_sample = (
            policy_term
            - alpha.detach()
            * entropy_est.reshape_as(policy_term)
            * self.entropy_scale
            + 0.01
            * ((loc / self.cfg.soft_bound) ** 6).sum(-1).reshape_as(policy_term)
        )
        valid = (1.0 - is_init.float()).reshape_as(actor_per_sample)
        actor_loss = (actor_per_sample * valid).sum() / valid.sum().clamp_min(1.0)

        # Alpha stays in fp32 (outside AMP).
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_actor.scale(actor_loss).backward()
            self.grad_scaler_actor.unscale_(self.opt_actor)
            actor_grad_norm = clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_actor.step(self.opt_actor)
            self.grad_scaler_actor.update()
        else:
            actor_loss.backward()
            actor_grad_norm = clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        if not diagnostics:
            return {}
        with torch.no_grad():
            q_for_log = q[:n_online]
            if self.reward_normalizer is not None:
                q_for_log = self.reward_normalizer.denormalize_return_values(q_for_log)
            return {
                "actor/loss": actor_loss.item(),
                "actor/grad_norm": actor_grad_norm.item(),
                "actor/alpha": alpha.item(),
                "actor/entropy": entropy_est.mean().item(),
                "actor/q_std": q_for_log.std(dim=1).mean().item(),
                "actor/mean_loc": loc.abs().mean().item(),
                "actor/mean_scale": scale.mean().item(),
            }

    def state_dict(self):
        state = OrderedDict(
            Q=self.Q.state_dict(),
            Q_target=self.Q_target.state_dict(),
            actor=self.actor.state_dict(),
            actor_target=self.actor_target.state_dict(),
            alpha=self.alpha.state_dict(),
            vecnorm_obs=self.vecnorm_obs.state_dict(),
        )
        if self.reward_normalizer is not None:
            state["reward_normalizer"] = self.reward_normalizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.Q.load_state_dict(state_dict["Q"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.Q_target.load_state_dict(state_dict.get("Q_target", state_dict["Q"]), strict=strict)
        self.actor_target.load_state_dict(
            state_dict.get("actor_target", state_dict["actor"]), strict=strict
        )
        if "alpha" in state_dict:
            self.alpha.load_state_dict(state_dict["alpha"], strict=strict)
        elif "log_alpha" in state_dict:
            self.alpha.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])
        reward_state = state_dict.get("reward_normalizer")
        if self.reward_normalizer is not None and reward_state is not None:
            self.reward_normalizer.load_state_dict(reward_state)
