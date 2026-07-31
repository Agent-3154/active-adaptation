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
- Student post-training / RL finetuning (Stage III) is out of scope.
- Teacher and student are separate modules (no weight sharing).

Model (paper Sec. 3.3, adapted)
-------------------------------
- Prior:   ``p_ψ(z_t | x_t, G_t)``
- Encoder: ``q_ϕ(z_t | x_t, G_t, y_{t+K})``  (training only; ``K = future_offsets``)
- Decoder: ``f_θ(a_t | x_t, G_t, z_t)``

Residual posterior: ``N(μ_p + μ_q, Σ_q)``. Latents are L2-normalized after
sampling; KL uses the pre-projection Gaussians. Online distillation uses
``L = L_ELBO + λ_scale L_scale + λ_tc L_tc``. Teacher action labels and any
DAgger control mixing are provided by ``train_imitation.py``, not this module.
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

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Tuple, Optional, List, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

from active_adaptation.learning.modules import VecNorm, IndependentNormal, MLP, CatTensors
from active_adaptation.learning.ppo.common import OBS_KEY, CMD_KEY, ACTION_KEY
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.utils.profiling import ScopedTimer


@dataclass
class InterPriorCfg:
    _target_: str = (
        "active_adaptation.learning.imitation.interprior.InterPriorCfg"
    )
    name: str = "interprior"

    # Prior / decoder conditioning: ``G_t`` ≈ command, ``x_t`` ≈ policy (w/ history).
    prior_in_keys: Tuple[str, ...] = ("goal", "policy")
    encoder_in_keys: Tuple[str, ...] = ("goal", "policy", "future")
    in_keys: Tuple[str, ...] = ("goal", "policy", "future")

    latent_dim: int = 64
    num_units: Tuple[int, ...] = (1024, 1024, 512)
    activation: str = "ReLU"

    lr: float = 2e-5
    buffer_size: int = 500  # ring length in steps; storage is [buffer_size, num_envs]
    batch_size: int = 512
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

    compile: bool = False
    debug: bool = False

    def get_class(self):
        return InterPriorPolicy


cs = ConfigStore.instance()
cs.store(name="interprior", node=InterPriorCfg, group="algo")


class DiagGaussianHead(nn.Module):
    """Linear head → diagonal Gaussian ``(loc, scale)`` with softplus scale."""

    def __init__(self, in_dim: int, out_dim: int, scale_lb: float = 1e-4):
        super().__init__()
        self.out_dim = out_dim
        self.scale_lb = scale_lb
        self.linear = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc, raw_scale = self.linear(x).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.scale_lb
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
    """Prior → sample z → decode action (student only; teacher labeling is in the script)."""

    def __init__(
        self,
        vecnorm: nn.Module,
        prior: nn.Module,
        decoder: nn.Module,
        in_keys: Tuple[str, ...],
        mode: str = "train",
    ):
        super().__init__()
        self.vecnorm = vecnorm
        self.prior = prior
        self.decoder = decoder
        self.mode = mode
        self.in_keys = list(in_keys) + ["prior_eps", "is_init"]
        self.out_keys = [
            ACTION_KEY,
            "mu_p",
            "std_p",
            ("next", "prior_eps"),
        ]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.vecnorm(tensordict)
        mu_p, std_p = self.prior(tensordict["prior_inp"])
        tensordict["mu_p"] = mu_p
        tensordict["std_p"] = std_p

        eps = tensordict["prior_eps"]
        if self.mode == "train":
            z = mu_p + std_p * eps
        else:
            z = mu_p
        z = F.normalize(z, dim=-1, eps=1e-6)
        tensordict["z"] = z

        action = self.decoder(torch.cat([tensordict["_policy_normed"], z], dim=-1))
        tensordict[ACTION_KEY] = action
        # Keep episode-constant ε for the next step (primer resets on done).
        tensordict["next", "prior_eps"] = eps
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

        self.prior = nn.Sequential(
            MLP(num_units=[prior_in_dim, *hidden], activation=Activation, first_non_muon=True),
            DiagGaussianHead(trunk_out, self.latent_dim),
        ).to(self.device)

        self.encoder = nn.Sequential(
            MLP(num_units=[encoder_in_dim, *hidden], activation=Activation, first_non_muon=True),
            DiagGaussianHead(trunk_out, self.latent_dim),
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

        # test run
        with torch.no_grad():
            self.vecnorm(fake)
            self.prior(fake["prior_inp"])
            self.encoder(fake["encoder_inp"])
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
        self.encoder.apply(init_)
        self.decoder_trunk.apply(init_)
        

        self.train_keys: List[str] = [
            ACTION_KEY,
            "teacher_action",
            "prior_eps",
            "is_init",
            *cfg.in_keys,
        ]

        self.global_step = 0
        self.rb: Optional[ReplayBuffer] = None
        self.opt: Optional[torch.optim.Optimizer] = None

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

        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys -= {"prior_eps"}
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size,
            fake_rb,
            fake_bootstrap=True,
            observation_keys=list(observation_keys),
        )
        print("InterPrior buffer:")
        print(self.rb)

        params = (
            list(self.prior.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder_trunk.parameters())
            + list(self.decoder_action.parameters())
            + list(self.decoder_goal.parameters())
        )
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr)

    # ------------------------------------------------------------------
    # Rollout / inference
    # ------------------------------------------------------------------

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if critic:
            raise NotImplementedError("Critic is not supported for InterPriorPolicy.")
        policy = InterPriorRolloutPolicy(
            vecnorm=self.vecnorm,
            prior=self.prior,
            decoder=nn.Sequential(self.decoder_trunk, self.decoder_action),
            in_keys=self.cfg.in_keys,
            mode=mode,
        )
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
        """Push one transition; expects ``teacher_action`` from the training script."""
        self.global_step += 1
        if "teacher_action" not in tensordict.keys(True, True):
            raise KeyError(
                "Missing teacher_action; label it in train_imitation rollout_policy."
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
        for _ in range(self.cfg.updates_per_train):
            infos.append(self._update())
        # Average metrics
        out = {}
        keys = infos[0].keys()
        for k in keys:
            out[k] = sum(d[k] for d in infos) / len(infos)
        out["train/student_frac"] = self.student_frac
        out["train/beta_kl"] = self.beta_kl
        out["train/buffer_size"] = float(len(self.rb))
        return out

    def _update(self) -> dict:
        need_tc = self.cfg.lambda_tc > 0.0
        batch = self.rb.sample(self.cfg.batch_size, next_obs=True)

        self.vecnorm(batch)
        mu_p, std_p = self.prior(batch["prior_inp"])
        mu_q, std_q = self.encoder(batch["encoder_inp"])

        # Residual posterior q(z) = N(μ_p + μ_q, Σ_q), prior p(z) = N(μ_p, Σ_p)
        mu_post = mu_p + mu_q
        eps = batch["prior_eps"]
        z = F.normalize(mu_post + std_q * eps, dim=-1, eps=1e-6)

        feat = self.decoder_trunk(torch.cat([batch["_policy_normed"], z], dim=-1))
        action_pred = self.decoder_action(feat)
        goal_pred = self.decoder_goal(feat)

        valid = (~batch["is_init"].bool()).float().reshape(-1)
        valid_cnt = valid.sum().clamp_min(1.0)

        act_err = (action_pred - batch["teacher_action"]).square().mean(dim=-1)
        loss_act = (act_err * valid).sum() / valid_cnt
        g_err = (goal_pred - batch["goal"]).square().mean(dim=-1)
        loss_goal = (g_err * valid).sum() / valid_cnt

        # KL(q ‖ p) with q=N(mu_post, std_q), p=N(mu_p, std_p)
        kl = IndependentNormal.kl(mu_post, std_q, mu_p, std_p)
        loss_kl = (kl * valid).sum() / valid_cnt

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
                "distill/kl": float(loss_kl.detach()),
                "distill/scale_loss": float(loss_scale.detach()),
                "distill/tc_loss": float(loss_tc.detach()),
                "distill/goal_loss": float(loss_goal.detach()),
                "distill/total_loss": float(loss.detach()),
                "distill/grad_norm": float(
                    grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm
                ),
                "distill/z_norm": float(z.norm(dim=-1).mean().detach()),
                "distill/mu_p_norm": float(mu_p.norm(dim=-1).mean().detach()),
            }

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
