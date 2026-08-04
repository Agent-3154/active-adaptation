# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
InterPrior variational distillation (Stage II).

Paper: https://arxiv.org/pdf/2602.06035

Codebase adaptations vs. the paper
----------------------------------
- Goal ``G_t`` maps to **command** observation groups in our MDPs.
- Observation tensors are assumed to **already include history**, so there is
  no separate history length / RNN — prior & decoder are MLPs on ``in_keys``.
- Future references ``y_{t+k}`` for encoder conditioning are built at train time by
  sampling ``k ∈ future_offsets`` from the replay ring (paper short-horizon
  preview ``K = {1,2,4,16}``). Keys are listed in ``future_keys`` (e.g. dense
  teacher command). Optional static ``aux_keys`` add privileged context at ``t``.
- Student post-training / RL finetuning (Stage III): a separate ``student``
  network (initialized from ``prior`` via trunk+loc copy) is the latent-space
  PPO actor; critic + GAE + clipped surrogate. By default the student uses a
  state-independent ``actor_std`` (``student_pred_std=False``, regular PPO);
  set ``student_pred_std=True`` to keep a state-dependent scale head.
  Rollout stochasticity is controlled by TorchRL ``interaction_type()``
  (not a ``mode`` flag).
- Teacher and student are separate modules (no weight sharing).

Model (paper Sec. 3.3, adapted)
-------------------------------
- Prior:   ``p_ψ(z_t | x_t, G_t)``   (Stage II / distill)
- Student: ``π_student(z_t | x_t, G_t)``  (Stage III / rl; copy of prior)
- Encoder: ``q_ϕ(z_t | x_t, G_t, y_{t+K})``  (training only; ``K = future_offsets``)
- Decoder: ``f_θ(a_t | x_t, G_t, z_t)``

Residual posterior: ``N(μ_p + μ_q, Σ_q)`` (or non-residual ``N(μ_q, Σ_q)`` via
``residual_posterior=False``). Latents are optionally L2-normalized
after sampling (``normalize_z``); KL / PPO log-probs use the pre-projection
Gaussians. KL supports per-dim free bits (``free_bits``). Online distillation
uses ``L = L_ELBO + λ_scale L_scale + λ_tc L_tc``. Teacher action labels and
any DAgger control mixing are provided by ``train_imitation.py``, not this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torchrl.data import Composite, TensorSpec, UnboundedContinuous
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from tensordict.nn.probabilistic import interaction_type, InteractionType

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

from active_adaptation.learning.modules import IndependentNormal, VecNorm, MLP, CatTensors
from active_adaptation.learning.ppo.common import (
    OBS_KEY,
    CMD_KEY,
    ACTION_KEY,
    REWARD_KEY,
    TERM_KEY,
    DONE_KEY,
    Critic,
    GAE,
    hard_copy_,
    make_batch,
    ppo_clipped_loss,
    resolve_clip_param,
)
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.utils.profiling import ScopedTimer


@dataclass
class InterPriorCfg:
    _target_: str = (
        "active_adaptation.learning.imitation.interprior.InterPriorCfg"
    )
    name: str = "interprior"
    # ``distill``: Stage II variational distillation. ``rl``: Stage III latent PPO
    # finetuning with a separate student actor (initialized from prior).
    stage: str = "distill"

    # Prior / decoder conditioning: ``G_t`` ≈ command, ``x_t`` ≈ policy (w/ history).
    prior_in_keys: Tuple[str, ...] = ("goal", "policy")
    encoder_in_keys: Tuple[str, ...] = ("goal", "policy", "future")
    in_keys: Tuple[str, ...] = ("goal", "policy", "future")

    latent_dim: int = 64
    num_units: Tuple[int, ...] = (1024, 1024, 512)
    activation: str = "ReLU"
    # L2-normalize z after sampling (paper projects latents onto the unit sphere).
    normalize_z: bool = True
    # Per-dim free bits: latent dims whose KL is below this threshold contribute
    # no KL gradient (prevents posterior collapse). 0 disables.
    free_bits: float = 0.0
    # Residual posterior q = N(μ_p + μ_q, Σ_q) (paper); if False the encoder
    # predicts the posterior mean directly: q = N(μ_q, Σ_q).
    residual_posterior: bool = True

    lr: float = 2e-5
    buffer_size: int = 500  # ring length in steps; storage is [buffer_size, num_envs]
    batch_size: int = 1024
    train_every: int = 32
    warm_up_steps: int = 32
    updates_per_train: int = 4

    # DAgger: fraction of steps / envs controlled by the student.
    # Annealed via ``step_schedule`` from 0 → ``student_frac_end``.
    student_frac: float = 0.0
    student_frac_end: float = 0.95
    dagger_warmup: float = 0.05  # progress fraction with student_frac=0

    # Loss weights (paper Sec. D.2)
    beta_kl: float = 1e-3
    beta_kl_end: float = 1e-2
    lambda_scale: float = 1e-3
    lambda_tc: float = 1e-3
    lambda_goal: float = 0.0  # optional cmd reconstruction; off by default

    # Critic / GAE / latent PPO (Stage III)
    critic_num_units: Tuple[int, ...] = (1024, 1024, 512)
    gamma: float = 0.99
    lmbda: float = 0.95
    clamp_reward: bool = False
    value_loss_coef: float = 1.0
    ppo_epochs: int = 4
    num_minibatches: int = 4
    clip_param: Any = (0.2, 0.2)
    entropy_coef: float = 0.0
    # Stage-III student scale: False → learned state-independent ``actor_std``
    # (regular PPO); True → state-dependent softplus head (same as prior).
    student_pred_std: bool = False

    compile: bool = False
    debug: bool = False

    def get_class(self):
        return InterPriorPolicy


cs = ConfigStore.instance()
cs.store(name="interprior", node=InterPriorCfg, group="algo")
cs.store(
    name="interprior_vanilla",
    node=InterPriorCfg(normalize_z=False, lambda_scale=0.0, residual_posterior=False),
    group="algo",
)
cs.store(
    name="interprior_rl",
    node=InterPriorCfg(stage="rl", student_frac=1.0, student_frac_end=1.0, dagger_warmup=0.0),
    group="algo",
)
cs.store(
    name="interprior_vanilla_rl",
    node=InterPriorCfg(stage="rl", student_frac=1.0, student_frac_end=1.0, dagger_warmup=0.0, normalize_z=False, lambda_scale=0.0, residual_posterior=False),
    group="algo",
)


class DiagGaussianHead(nn.Module):
    """Linear head → diagonal Gaussian ``(loc, scale)``.

    If ``predict_std`` is True (prior / encoder), the linear outputs ``2 * out_dim``
    and scale is ``softplus(raw) + scale_lb``. If False (PPO-style student), only
    ``loc`` is predicted and scale is a learned state-independent ``actor_std``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        predict_std: bool = True,
        scale_lb: float = 1e-4,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.predict_std = predict_std
        self.scale_lb = scale_lb
        if predict_std:
            self.linear = nn.Linear(in_dim, out_dim * 2)
        else:
            self.linear = nn.Linear(in_dim, out_dim)
            self.actor_std = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.predict_std:
            loc, raw_scale = self.linear(x).chunk(2, dim=-1)
            scale = F.softplus(raw_scale) + self.scale_lb
        else:
            loc = self.linear(x)
            scale = torch.ones_like(loc) * self.actor_std
        return loc, scale


def _wasserstein2_diag(
    mu0: torch.Tensor,
    std0: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
) -> torch.Tensor:
    """Appendix D.2 of the paper:
    Temporal consistency loss via Squared 2-Wasserstein between diagonal Gaussians, mean over latent dim.
    """
    return (mu0 - mu1).square().sum(-1) + (std0 - std1).square().sum(-1)


class InterPriorRolloutPolicy(TensorDictModuleBase):
    """Latent Gaussian actor → decode action.

    Stochasticity is controlled by TorchRL ``interaction_type()`` (MODE vs
    RANDOM), matching SAC rollout policies — not by a ``mode`` string.
    """

    def __init__(
        self,
        vecnorm: nn.Module,
        actor: nn.Module,
        decoder: nn.Module,
        in_keys: Tuple[str, ...],
        normalize_z: bool = True,
    ):
        super().__init__()
        self.vecnorm = vecnorm
        self.actor = actor
        self.decoder = decoder
        self.normalize_z = normalize_z
        self.in_keys = list(in_keys) + ["prior_eps", "is_init"]
        self.out_keys = [
            ACTION_KEY,
            "z",
            "action_log_prob",
            "loc",
            "scale",
            ("next", "prior_eps"),
        ]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.vecnorm(tensordict)
        loc, scale = self.actor(tensordict["prior_inp"])
        tensordict["loc"] = loc
        tensordict["scale"] = scale
        dist = IndependentNormal(loc, scale)

        if interaction_type() == InteractionType.MODE:
            z = loc.clone()
        else:
            z = dist.rsample()
        # PPO / KL use pre-projection latents; decoder may see the unit vector.
        tensordict["action_log_prob"] = dist.log_prob(z).unsqueeze(-1)
        z_dec = F.normalize(z, dim=-1, eps=1e-6) if self.normalize_z else z
        tensordict["z"] = z

        action = self.decoder(torch.cat([tensordict["_policy_normed"], z_dec], dim=-1))
        tensordict[ACTION_KEY] = action
        # Keep episode-constant ε for the next step when the primer is present.
        if "prior_eps" in tensordict.keys(True, True):
            tensordict["next", "prior_eps"] = tensordict["prior_eps"]
        return tensordict


class InterPriorPolicy(TensorDictModuleBase):
    """Variational distillation student (no shared parameters with the teacher)."""

    def __init__(
        self,
        cfg: InterPriorCfg,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.action_dim = int(action_spec.shape[-1])
        self.latent_dim = int(cfg.latent_dim)

        self.student_frac = float(cfg.student_frac)
        self.beta_kl = float(cfg.beta_kl)
        self.clip_param = resolve_clip_param(cfg.clip_param)
        self.entropy_coef = float(cfg.entropy_coef)

        fake = observation_spec.zero().to(self.device)

        Activation = getattr(nn, cfg.activation)
        prior_in_dim = fake["goal"].shape[-1] + fake["policy"].shape[-1]
        encoder_in_dim = fake["goal"].shape[-1] + fake["policy"].shape[-1] + fake["future"].shape[-1]
        hidden = list(cfg.num_units)
        trunk_out = hidden[-1]

        self.vecnorm = Seq(
            Mod(VecNorm(fake["goal"].shape[-1]), ["goal"], ["_goal_normed"]),
            Mod(VecNorm(fake["policy"].shape[-1]), ["policy"], ["_policy_normed"]),
            Mod(VecNorm(fake["future"].shape[-1]), ["future"], ["_future_normed"]),
            CatTensors(["_goal_normed", "_policy_normed"], "prior_inp", sort=False),
            CatTensors(["_goal_normed", "_policy_normed", "_future_normed"], "encoder_inp", sort=False)
        ).to(self.device)

        def make_latent_actor(*, predict_std: bool) -> nn.Sequential:
            return nn.Sequential(
                MLP(
                    num_units=[prior_in_dim, *hidden],
                    activation=Activation,
                    first_non_muon=True,
                ),
                DiagGaussianHead(
                    trunk_out, self.latent_dim, predict_std=predict_std
                ),
            ).to(self.device)

        self.prior = make_latent_actor(predict_std=True)
        # Separate RL actor (Stage III); trunk+loc synced from prior in on_stage_start.
        self.student = make_latent_actor(predict_std=bool(cfg.student_pred_std))

        self.encoder = nn.Sequential(
            MLP(num_units=[encoder_in_dim, *hidden], activation=Activation, first_non_muon=True),
            DiagGaussianHead(trunk_out, self.latent_dim, predict_std=True),
        ).to(self.device)

        decoder_in_dim = fake["policy"].shape[-1] + self.latent_dim
        self.decoder_trunk = nn.Sequential(
            MLP(
                num_units=[decoder_in_dim, *hidden],
                activation=Activation,
                first_non_muon=True,
            ),
        ).to(self.device)
        self.decoder_action = nn.Linear(trunk_out, self.action_dim).to(self.device)
        self.decoder_goal = nn.Linear(trunk_out, fake["goal"].shape[-1]).to(self.device)

        # Critic on privileged encoder input (goal + policy + future), paper Sec. E.
        critic_hidden = list(cfg.critic_num_units)
        critic_mlp = MLP(
            num_units=[encoder_in_dim, *critic_hidden],
            activation=Activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            Mod(critic_mlp, ["encoder_inp"], ["_critic_feature"]),
            Mod(Critic(1), ["_critic_feature"], ["state_value"]),
        ).to(self.device)
        self.gae = GAE(cfg.gamma, cfg.lmbda)
        self.critic_loss_fn = nn.MSELoss(reduction="none")

        # test run
        with torch.no_grad():
            self.vecnorm(fake)
            self.prior(fake["prior_inp"])
            self.student(fake["prior_inp"])
            self.encoder(fake["encoder_inp"])
            self.critic(fake)
            z0 = F.normalize(
                torch.zeros(fake.shape[0], self.latent_dim, device=self.device), dim=-1
            )
            feat = self.decoder_trunk(torch.cat([fake["_policy_normed"], z0], dim=-1))
            self.decoder_action(feat)
            self.decoder_goal(feat)

        def init_(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.prior.apply(init_)
        self.student.apply(init_)
        self.encoder.apply(init_)
        self.decoder_trunk.apply(init_)
        self.critic.apply(init_)

        self.train_keys: List[str] = [
            ACTION_KEY,
            "teacher_action",
            "prior_eps",
            "is_init",
            *cfg.in_keys,
        ]
        # Keys needed for latent-PPO minibatch updates.
        self.ppo_keys: List[str] = [
            "is_init",
            "adv",
            "ret",
            "z",
            "action_log_prob",
            *cfg.in_keys,
        ]

        self.global_step = 0
        self.rb: Optional[ReplayBuffer] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.observation_keys: List[str] = []
        self._student_loaded = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, cfg: InterPriorCfg, env: _EnvBase, device: str):
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
        )

    def make_tensordict_primer(self):
        """Episode-constant ``prior_eps`` for temporally consistent latents."""
        from torchrl.envs import TensorDictPrimer

        num_envs = int(self.observation_spec.shape[0])
        dev = torch.device(self.device)
        spec = {
            "prior_eps": UnboundedContinuous(
                (num_envs, self.latent_dim), device=dev
            ),
        }
        return TensorDictPrimer(
            Composite(spec, shape=[num_envs], device=dev),
            random=True,
            reset_key="done",
            expand_specs=False,
        )

    def on_stage_start(self, _stage: str, env: _EnvBase):
        fake_rb = env.fake_tensordict().exclude(("next", "stats"), "collector")
        if "teacher_action" not in fake_rb.keys(True, True):
            fake_rb["teacher_action"] = torch.zeros_like(fake_rb[ACTION_KEY])
        if "prior_eps" not in fake_rb.keys(True, True):
            fake_rb["prior_eps"] = torch.zeros(
                fake_rb.shape[0], self.latent_dim, device=fake_rb.device
            )
        if "z" not in fake_rb.keys(True, True):
            fake_rb["z"] = torch.zeros(
                fake_rb.shape[0], self.latent_dim, device=fake_rb.device
            )
        if "action_log_prob" not in fake_rb.keys(True, True):
            fake_rb["action_log_prob"] = torch.zeros(
                fake_rb.shape[0], 1, device=fake_rb.device
            )

        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys -= {"prior_eps"}
        self.observation_keys = list(observation_keys)
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size,
            fake_rb,
            fake_bootstrap=True,
            observation_keys=self.observation_keys,
        )
        print("InterPrior buffer:")
        print(self.rb)

        if self.cfg.stage == "distill":
            distill_params = (
                list(self.prior.parameters())
                + list(self.encoder.parameters())
                + list(self.decoder_trunk.parameters())
                + list(self.decoder_action.parameters())
                + list(self.decoder_goal.parameters())
            )
            self.opt = torch.optim.Adam(
                [
                    {"params": distill_params, "lr": self.cfg.lr},
                    {"params": self.critic.parameters(), "lr": self.cfg.lr},
                ]
            )
        elif self.cfg.stage == "rl":
            # Sync from distilled prior unless a finetuned student was already loaded.
            if not self._student_loaded:
                self._init_student_from_prior()
            # the action decoder is freezed
            self.opt = torch.optim.Adam(
                [
                    {"params": self.student.parameters(), "lr": self.cfg.lr},
                    {"params": self.critic.parameters(), "lr": self.cfg.lr},
                ]
            )
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage!r}; expected 'distill' or 'rl'.")

    def _init_student_from_prior(self) -> None:
        """Copy distilled prior into the Stage-III student.

        When ``student_pred_std`` matches the prior (state-dependent), a full
        ``hard_copy_`` works. With a state-independent student head, only the
        MLP trunk and loc weights are copied; ``actor_std`` keeps its PPO-style
        ones init.
        """
        if self.cfg.student_pred_std:
            hard_copy_(self.prior, self.student)
            return

        hard_copy_(self.prior[0], self.student[0])
        src_head: DiagGaussianHead = self.prior[1]
        tgt_head: DiagGaussianHead = self.student[1]
        d = self.latent_dim
        with torch.no_grad():
            tgt_head.linear.weight.copy_(src_head.linear.weight[:d])
            if tgt_head.linear.bias is not None and src_head.linear.bias is not None:
                tgt_head.linear.bias.copy_(src_head.linear.bias[:d])

    # ------------------------------------------------------------------
    # Rollout / inference
    # ------------------------------------------------------------------

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        # Sampling vs mean is controlled by interaction_type / set_exploration_type;
        # ``mode`` only selects whether VecNorm stats keep updating.
        actor = self.student if self.cfg.stage == "rl" else self.prior
        if self.cfg.stage == "rl" or mode != "train":
            vecnorm = VecNorm.freeze()(self.vecnorm)
        else:
            vecnorm = self.vecnorm
        policy = InterPriorRolloutPolicy(
            vecnorm=vecnorm,
            actor=actor,
            decoder=nn.Sequential(self.decoder_trunk, self.decoder_action),
            in_keys=self.cfg.in_keys,
            normalize_z=self.cfg.normalize_z,
        )
        if critic:
            # vecnorm inside the rollout policy writes ``encoder_inp`` for the critic.
            policy = Seq(policy, self.critic)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    def step_schedule(self, progress: float) -> None:
        """Anneal DAgger ``student_frac`` and KL ``beta`` with progress in ``[0, 1]``.

        ``student_frac`` is exposed for the training script to use when mixing
        teacher/student control; labeling itself happens in ``train_imitation.py``.
        """
        progress = float(min(max(progress, 0.0), 1.0))
        warm = float(self.cfg.dagger_warmup)
        if progress <= warm or warm >= 1.0:
            self.student_frac = 0.0
        else:
            t = (progress - warm) / (1.0 - warm)
            self.student_frac = min(self.cfg.student_frac_end, t * self.cfg.student_frac_end)

        # β-VAE schedule: beta_kl → beta_kl_end
        self.beta_kl = self.cfg.beta_kl + progress * (
            self.cfg.beta_kl_end - self.cfg.beta_kl
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def step(self, tensordict: TensorDict) -> dict:
        """Push one transition; train when due.

        Distill stage expects ``teacher_action`` from the training script.
        RL stage collects on-policy latent actions from the student rollout.
        """
        self.global_step += 1
        if self.cfg.stage == "distill":
            if "teacher_action" not in tensordict.keys(True, True):
                raise KeyError(
                    "Missing teacher_action; label it in train_imitation rollout."
                )

        td = tensordict.exclude(("next", "stats"), "collector")
        self.rb.push(td)

        if (
            self.global_step > self.cfg.warm_up_steps
            and self.global_step % self.cfg.train_every == 0
            and len(self.rb) > 0
        ):
            return self.train_op()
        return {}

    @ScopedTimer("interprior_train")
    @VecNorm.freeze()
    def train_op(self) -> dict:
        infos = []
        if self.cfg.stage == "distill":
            for _ in range(self.cfg.updates_per_train):
                infos.append(self._update_distillation())
            # Warm the critic on recent on-policy segments (no actor updates).
            ppo_info = self._update_ppo(actor=False, critic=True)
            if ppo_info:
                infos.append(ppo_info)
        elif self.cfg.stage == "rl":
            ppo_info = self._update_ppo(actor=True, critic=True)
            if ppo_info:
                infos.append(ppo_info)
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage!r}")

        if not infos:
            return {
                "train/student_frac": self.student_frac,
                "train/beta_kl": self.beta_kl,
                "train/buffer_size": float(len(self.rb)),
            }

        # Average metrics (union of keys across updates)
        out = {}
        keys = set().union(*(d.keys() for d in infos))
        for k in keys:
            vals = [d[k] for d in infos if k in d]
            out[k] = sum(vals) / len(vals)
        out["train/student_frac"] = self.student_frac
        out["train/beta_kl"] = self.beta_kl
        out["train/buffer_size"] = float(len(self.rb))
        out["train/stage"] = 0.0 if self.cfg.stage == "distill" else 1.0
        return out

    @torch.no_grad()
    def compute_advantage(
        self,
        tensordict: TensorDict,
        critic: Mod,
        adv_key: str = "adv",
        ret_key: str = "ret",
        clamp_reward: bool = False,
    ):
        """GAE on a trajectory batch shaped ``[N, T, …]`` (same recipe as ``ppo_symaug``)."""
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as flat:
                critic(self.vecnorm(flat))
                critic(self.vecnorm(flat["next"]))

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if isinstance(rewards, TensorDict):
            rewards = torch.concat(list(rewards.values()), dim=-1)
        rewards = rewards.sum(-1, keepdim=True)
        tensordict["next", "reward_aggregated"] = rewards
        if clamp_reward:
            rewards = rewards.clamp_min(0.0)
        rewards = rewards * (1.0 - self.gae.gamma)

        discount = tensordict.get(("next", "discount"), None)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)
        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update_distillation(self) -> dict:
        need_tc = self.cfg.lambda_tc > 0.0
        batch = self.rb.sample(self.cfg.batch_size, next_obs=True)

        self.vecnorm(batch)
        mu_p, std_p = self.prior(batch["prior_inp"])
        mu_q, std_q = self.encoder(batch["encoder_inp"])

        # Posterior q(z) = N(μ_post, Σ_q), prior p(z) = N(μ_p, Σ_p).
        # Residual (paper): μ_post = μ_p + μ_q; non-residual: μ_post = μ_q.
        mu_post = mu_p + mu_q if self.cfg.residual_posterior else mu_q
        eps = batch["prior_eps"]
        z = mu_post + std_q * eps
        if self.cfg.normalize_z:
            z = F.normalize(z, dim=-1, eps=1e-6)

        feat = self.decoder_trunk(torch.cat([batch["_policy_normed"], z], dim=-1))
        action_pred = self.decoder_action(feat)
        goal_pred = self.decoder_goal(feat)

        valid = (~batch["is_init"].bool()).float().reshape(-1)
        valid_cnt = valid.sum().clamp_min(1.0)

        act_err = (action_pred - batch["teacher_action"]).square().mean(dim=-1)
        loss_act = (act_err * valid).sum() / valid_cnt
        g_err = (goal_pred - batch["goal"]).square().mean(dim=-1)
        loss_goal = (g_err * valid).sum() / valid_cnt

        # Per-dim KL(q ‖ p) with q=N(mu_post, std_q), p=N(mu_p, std_p)
        kl_dim = (
            torch.log(std_p) - torch.log(std_q)
            + (std_q.square() + (mu_p - mu_post).square()) / (2.0 * std_p.square())
            - 0.5
        )
        kl_true = kl_dim.sum(-1)
        if self.cfg.free_bits > 0.0:
            # Free bits: dims whose KL is already below the threshold get no
            # KL gradient. Diagnostics still report the true (unmasked) KL.
            fb_mask = (kl_dim > self.cfg.free_bits).detach()
            kl_obj = (kl_dim * fb_mask).sum(-1)
            kl_active_frac = fb_mask.float().mean()
        else:
            kl_obj = kl_true
            kl_active_frac = kl_dim.new_ones(())
        loss_kl = (kl_obj * valid).sum() / valid_cnt
        kl_diag = (kl_true * valid).sum() / valid_cnt

        loss_scale = (mu_p.norm(dim=-1).square() - 1.0).square()
        loss_scale = (loss_scale * valid).sum() / valid_cnt

        loss_tc = action_pred.new_zeros(())
        if need_tc and batch["next"] is not None:
            self.vecnorm(batch["next"])
            mu_p1, std_p1 = self.prior(batch["next", "prior_inp"])
            # Skip pairs that cross episode boundaries.
            cont = (~batch["next", "is_init"].bool()).float().reshape(-1)
            w2 = _wasserstein2_diag(mu_p.detach(), std_p.detach(), mu_p1, std_p1)
            cont_cnt = cont.sum().clamp_min(1.0)
            loss_tc = (w2 * cont).sum() / cont_cnt

        loss = (
            loss_act
            + self.beta_kl * loss_kl
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_tc * loss_tc
            + self.cfg.lambda_goal * loss_goal
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.opt.param_groups[0]["params"], 1.0
        )
        self.opt.step()

        with torch.no_grad():
            return {
                "distill/beta_kl": float(self.beta_kl),
                "distill/action_loss": float(loss_act.detach()),
                "distill/kl": float(kl_diag.detach()),
                "distill/kl_loss": float(loss_kl.detach()),
                "distill/kl_active_frac": float(kl_active_frac.detach()),
                "distill/scale_loss": float(loss_scale.detach()),
                "distill/tc_loss": float(loss_tc.detach()),
                "distill/goal_loss": float(loss_goal.detach()),
                "distill/total_loss": float(loss.detach()),
                "distill/grad_norm": float(
                    grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm
                ),
                "distill/z_norm": float(z.norm(dim=-1).mean().detach()),
                "distill/mu_p_norm": float(mu_p.norm(dim=-1).mean().detach()),
                "distill/mu_q_norm": float(mu_q.norm(dim=-1).mean().detach()),
            }

    def _update_ppo(self, actor: bool = True, critic: bool = True) -> dict:
        """GAE on the latest on-policy segment, then minibatch PPO updates.

        Args:
            actor: If True, apply the clipped latent-policy surrogate (Stage III).
            critic: If True, fit the value head to GAE returns. Use
                ``actor=False, critic=True`` during distillation to warm the
                critic without moving the policy.
        """
        if not (actor or critic):
            return {}
        if len(self.rb) < self.cfg.train_every + 1:
            return {}

        # [T, N, ...] with next obs reconstructed; transpose to [N, T, ...] for GAE.
        batch = self.rb.last(self.cfg.train_every, next_obs=True).transpose(0, 1)
        batch = batch.contiguous() # to make `view` happy

        with ScopedTimer("compute_advantage"):
            self.compute_advantage(
                batch,
                self.critic,
                "adv",
                "ret",
                clamp_reward=self.cfg.clamp_reward,
            )
            adv = batch["adv"]
            adv_mean = adv.mean()
            adv_std = adv.std().clamp_min(1e-7)
            batch["adv"] = (adv - adv_mean) / adv_std

        ret_valid = batch["ret"][~batch["is_init"].bool()]
        ret_var = ret_valid.var().clamp_min(1e-7)

        keys = list(self.ppo_keys)
        if not actor:
            keys = [k for k in keys if k not in ("z", "action_log_prob")]
        td = batch.select(*keys)
        infos = []
        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(td, self.cfg.num_minibatches):
                infos.append(
                    self._update_actor_critic(
                        minibatch, ret_var, update_actor=actor, update_critic=critic
                    )
                )

        out = {}
        keys = infos[0].keys()
        for k in keys:
            out[k] = sum(d[k] for d in infos) / len(infos)
        out["critic/adv_mean"] = float(adv_mean.detach())
        out["critic/adv_std"] = float(adv_std.detach())
        out["critic/value_mean"] = float(batch["ret"].mean().detach())
        out["critic/value_std"] = float(batch["ret"].std().detach())
        return out

    def _update_actor_critic(
        self,
        minibatch: TensorDict,
        ret_var: torch.Tensor,
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> dict:
        """One minibatch; actor and/or critic controlled by flags."""
        self.vecnorm(minibatch)
        valid = (~minibatch["is_init"].bool()).float()
        valid_cnt = valid.sum().clamp_min(1.0)

        loss = minibatch["ret"].new_zeros(())
        info: dict = {}

        if update_actor:
            loc, scale = self.student(minibatch["prior_inp"])
            dist = IndependentNormal(loc, scale)
            z = minibatch["z"]
            log_probs = dist.log_prob(z).unsqueeze(-1)
            log_probs_data = minibatch["action_log_prob"]

            adv = minibatch["adv"]
            log_ratio = (log_probs - log_probs_data).reshape_as(adv)
            ratio = torch.exp(log_ratio)
            policy_loss = ppo_clipped_loss(ratio, adv, self.clip_param)
            policy_loss = (policy_loss.reshape_as(valid) * valid).sum() / valid_cnt

            entropy = dist.entropy().reshape_as(valid)
            entropy = (entropy * valid).sum() / valid_cnt
            entropy_loss = -self.entropy_coef * entropy

            loss = loss + policy_loss + entropy_loss
            with torch.no_grad():
                eps_neg, eps_pos = self.clip_param
                ratio_det = ratio.detach()
                info.update(
                    {
                        "actor/policy_loss": float(policy_loss.detach()),
                        "actor/entropy": float(entropy.detach()),
                        "actor/approx_kl": float(
                            ((ratio_det - 1.0) - log_ratio.detach()).mean()
                        ),
                        "actor/clamp_pos": float(
                            (ratio_det > 1.0 + eps_pos).float().mean()
                        ),
                        "actor/clamp_neg": float(
                            (ratio_det < 1.0 - eps_neg).float().mean()
                        ),
                    }
                )

        if update_critic:
            values = self.critic(minibatch)["state_value"]
            value_loss = self.critic_loss_fn(minibatch["ret"], values)
            value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt
            loss = loss + self.cfg.value_loss_coef * value_loss
            with torch.no_grad():
                info["critic/value_loss"] = float(value_loss.detach())
                info["critic/explained_var"] = float(
                    (1.0 - value_loss / ret_var).detach()
                )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if update_actor:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.opt.param_groups[0]["params"], 1.0
            )
            info["actor/grad_norm"] = float(
                actor_grad_norm.detach()
                if torch.is_tensor(actor_grad_norm)
                else actor_grad_norm
            )
        if update_critic:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.opt.param_groups[1]["params"], 1.0
            )
            info["critic/grad_norm"] = float(
                critic_grad_norm.detach()
                if torch.is_tensor(critic_grad_norm)
                else critic_grad_norm
            )
        self.opt.step()
        return info

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["global_step"] = self.global_step
        state_dict["student_frac"] = self.student_frac
        state_dict["beta_kl"] = self.beta_kl
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _sd = state_dict.get(name, {})
            if not _sd:
                continue
            try:
                module.load_state_dict(_sd, strict=strict)
                succeed_keys.append(name)
                if name == "student":
                    self._student_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        if "global_step" in state_dict:
            self.global_step = int(state_dict["global_step"])
        if "student_frac" in state_dict:
            self.student_frac = float(state_dict["student_frac"])
        if "beta_kl" in state_dict:
            self.beta_kl = float(state_dict["beta_kl"])
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys
