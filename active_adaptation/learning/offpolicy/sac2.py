from __future__ import annotations

import copy
import math
import einops
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal, Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules import VecNorm, IndependentNormal, ConditionalBlock, CatTensors
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    OBS_KEY,
    CMD_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import (
    expected_from_logits,
    cvar_from_logits,
    project_categorical_bellman,
)
from active_adaptation.learning.offpolicy.objectives import (
    MultiStepReturn,
    prior_bc_loss,
)
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.distribution import FasterTransformedDistribution
from active_adaptation.learning.offpolicy.sac import AlphaModule
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import (
    check_parameters,
    unwrap_ddp,
    wrap_ddp,
)
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform
from tensordict.nn.probabilistic import interaction_type, InteractionType

cs = ConfigStore.instance()


clip_grad_norm_ = nn.utils.clip_grad_norm_


def gaussian_target_entropy(act_dim: int, sigma: float) -> float:
    """Differential entropy of independent \\mathcal N(0, \\sigma^2) in \\mathbb R^d (FlashSAC-style).

    H = (d/2) * log(2 * pi * e * sigma^2). Used as SAC log-alpha target when
    :attr:`~SACConfig.target_entropy_sigma` is set.
    """
    if sigma <= 0:
        raise ValueError("target_entropy_sigma must be positive for principled entropy.")
    return 0.5 * float(act_dim) * math.log(2.0 * math.pi * math.e * sigma * sigma)


def _init_sac_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


@dataclass
class SACConfig:
    _target_: str = "active_adaptation.learning.offpolicy.sac2.SACConfig"
    name: str = "sac2"
    train_every: int = 4
    buffer_size: int = 2000
    warm_up_steps: int = 200
    lr: float = 5e-4
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper` (see ``ppo_symaug``).
    muon: bool = True
    weight_decay: float = 0.02
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    # architecture
    actor_init: str = "zeros"
    distributional: bool = True
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    sym_aug: bool = False
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # AR(1) pre-tanh exploration noise on rollout only: eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * N(0,I).
    use_correlated: bool = True
    # sac specific
    # Scales the entropy stream: soft Q = Q_task + entropy_bonus * Q_ent (0 = hard task Q).
    entropy_bonus: float = 1.0
    alpha_init: float = 4e-3
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common heuristic for tanh-squashed SAC).
    target_entropy_sigma: float | None = 0.15
    soft_bound: float = 2.0 * math.pi

    tau_actor: float = 0.1 # a relatively large value for faster convergence
    tau_Q: float = 0.02  # a relatively large value for faster convergence
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); separate GradScalers for critic and actor (alpha stays fp32).
    use_amp: bool = True
    # Clamp aggregated rewards at 0 before TD / reward-norm (avoids suicide from negative rewards).
    clamp_reward: bool = True
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8

    # path to prior data for RLPD
    prior_data: str | None = None
    prior_data_ratio: float = 0.4
    # Gated BC toward prior actions when online Q < MC return-to-go (0 disables).
    bc_loss: str = "mse"  # "mse" | "nll"
    bc_coef: float = 0.0
    bc_coef_mse: float = 1.0
    bc_coef_nll: float = 0.05

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)

    def __post_init__(self):
        if self.bc_loss == "mse":
            self.bc_coef = self.bc_coef_mse
        elif self.bc_loss == "nll":
            self.bc_coef = self.bc_coef_nll
        else:
            raise ValueError(f"Unknown bc_loss={self.bc_loss!r}; expected 'mse' or 'nll'.")

    def get_class(self):
        return SAC

cs.store(name="sac2", node=SACConfig, group="algo")


class DualHeadCriticTrunk(nn.Module):
    """Shared backbone with separate task and entropy linear heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        task_output_dim: int = 1,
        ent_output_dim: int = 1,
        activation: type[nn.Module] = nn.SiLU,
        norm: str | None = "rms",
        condition_dim: int = 0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.in_layer.weight._non_muon = True
        self.task_out = nn.Linear(hidden_dim, task_output_dim)
        self.task_out.weight._non_muon = True
        self.ent_out = nn.Linear(hidden_dim, ent_output_dim)
        self.ent_out.weight._non_muon = True

        self.block1 = ConditionalBlock(
            hidden_dim=hidden_dim,
            activation=activation,
            norm=norm,
            condition_dim=condition_dim,
            dropout=0.005,
        )
        self.block2 = ConditionalBlock(
            hidden_dim=hidden_dim,
            activation=activation,
            norm=norm,
            condition_dim=condition_dim,
            dropout=0.005,
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.apply(_init_sac_linear)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_layer(x)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.norm(x)
        return self.task_out(x), self.ent_out(x)


class TwinDualStreamModule(nn.Module):
    """Twin dual-head critics; share backbone within each twin only."""

    def __init__(self, fn: Callable[[], DualHeadCriticTrunk]):
        super().__init__()
        self.critic_1 = fn()
        self.critic_2 = fn()

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if act.dim() == 2:
            inp = torch.cat([obs, act], dim=-1)
            t1, e1 = self.critic_1(inp)
            t2, e2 = self.critic_2(inp)
            # Task/ent: [B, 2] (scalar) or [B, 2 * num_atoms] (C51).
            return torch.cat([t1, t2], dim=-1), torch.cat([e1, e2], dim=-1)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "batch obs -> (batch k) obs", k=k)
            act_flat = einops.rearrange(act, "batch k act_dim -> (batch k) act_dim")
            task, ent = self.forward(obs_flat, act_flat)
            task = einops.rearrange(task, "(batch k) fused -> batch k fused", batch=b, k=k)
            ent = einops.rearrange(ent, "(batch k) fused -> batch k fused", batch=b, k=k)
            return task, ent
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")


class DualStreamCritic(nn.Module):
    """Twin critics with shared-backbone dual heads: soft Q = Q_task + λ Q_ent.

    Distributional mode uses separate C51 supports for task vs entropy (different
    scales). Scalar mode uses MSE on both heads. ``entropy_bonus`` (λ) scales the
    entropy stream in soft values only; TD uses unscaled ``(-α log π) * scale``.
    """

    def __init__(
        self,
        module: TwinDualStreamModule,
        *,
        distributional: bool,
        entropy_bonus: float = 1.0,
        task_v_range: tuple[float, float] = (-0.5, 5.0),
        ent_v_range: tuple[float, float] = (-1.0, 1.0),
        num_atoms: int = 101,
    ):
        super().__init__()
        self.module = module
        self.distributional = distributional
        self.entropy_bonus = float(entropy_bonus)
        if distributional:
            t_lo, t_hi = task_v_range
            e_lo, e_hi = ent_v_range
            if not (t_hi > t_lo and e_hi > e_lo):
                raise ValueError(
                    f"Value ranges must satisfy max > min; got task={task_v_range}, ent={ent_v_range}"
                )
            self.register_buffer("task_support", torch.linspace(t_lo, t_hi, num_atoms))
            self.register_buffer("ent_support", torch.linspace(e_lo, e_hi, num_atoms))
            self.task_support: torch.Tensor
            self.ent_support: torch.Tensor
        else:
            self.register_buffer("task_support", torch.empty(0))
            self.register_buffer("ent_support", torch.empty(0))

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.module(obs, act)

    def expected(
        self,
        pred: torch.Tensor,
        support: torch.Tensor,
        risk_alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Twin values ``[..., 2]``: identity if scalar, else C51 expectation on ``support``."""
        if not self.distributional:
            return pred
        n_atom = int(support.shape[0])
        if pred.shape[-1] == 2 * n_atom:
            l1, l2 = pred.chunk(2, dim=-1)
        elif pred.shape[-1] == 2 and pred.shape[-2] == n_atom:
            l1, l2 = pred[..., 0], pred[..., 1]
        else:
            raise ValueError(
                f"Expected logits [..., {n_atom}, 2] or [..., {2 * n_atom}], "
                f"got shape {tuple(pred.shape)}."
            )
        if risk_alpha is not None:
            e1 = cvar_from_logits(l1, support, risk_alpha)
            e2 = cvar_from_logits(l2, support, risk_alpha)
        else:
            e1 = expected_from_logits(l1, support)
            e2 = expected_from_logits(l2, support)
        return torch.cat([e1, e2], dim=-1)

    def soft_q(
        self,
        q_task: torch.Tensor,
        q_ent: torch.Tensor,
        *,
        clip: bool = False,
    ) -> torch.Tensor:
        """``Q_task + λ Q_ent``; if ``clip``, use per-stream min (shape ``[..., 1]``)."""
        if clip:
            q_task = q_task.min(dim=-1, keepdim=True).values
            q_ent = q_ent.min(dim=-1, keepdim=True).values
        return q_task + self.entropy_bonus * q_ent

    def get_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        *,
        clip: bool = False,
        risk_alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Soft Q from a single forward (dropout-safe)."""
        task_pred, ent_pred = self.module(obs, act)
        return self.soft_q(
            self.expected(task_pred, self.task_support, risk_alpha),
            self.expected(ent_pred, self.ent_support, risk_alpha),
            clip=clip,
        )

    def _c51_backup(
        self,
        logits: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        n_atom = int(support.shape[0])
        l1, l2 = logits.chunk(2, dim=-1) if logits.shape[-1] == 2 * n_atom else (logits[..., 0], logits[..., 1])
        p1 = project_categorical_bellman(l1, reward, discount, support)
        p2 = project_categorical_bellman(l2, reward, discount, support)
        z = support.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        ev1 = (p1 * z).sum(dim=-1, keepdim=True)
        ev2 = (p2 * z).sum(dim=-1, keepdim=True)
        return torch.where(ev1 < ev2, p1, p2)

    def _stream_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        if not self.distributional:
            return (pred - target).square().sum(dim=-1)
        n_atom = int(support.shape[0])
        l1, l2 = pred.chunk(2, dim=-1) if pred.shape[-1] == 2 * n_atom else (pred[..., 0], pred[..., 1])
        target = target.to(dtype=l1.dtype)
        log_p1 = F.log_softmax(l1, dim=-1).clamp(min=-30.0)
        log_p2 = F.log_softmax(l2, dim=-1).clamp(min=-30.0)
        return -((target * log_p1).sum(dim=-1) + (target * log_p2).sum(dim=-1))

    @torch.no_grad()
    def compute_targets(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        ent_bonus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Task TD on ``reward``; entropy TD on ``ent_bonus`` (= ``(-α log π) * scale``)."""
        task_pred, ent_pred = self.module(next_obs, next_act)
        reward = reward.reshape(-1, 1)
        discount = discount.reshape(-1, 1)
        e = ent_bonus.reshape(-1, 1).to(dtype=ent_pred.dtype)

        if self.distributional:
            task_target = self._c51_backup(task_pred, reward, discount, self.task_support)
        else:
            task_target = reward + discount * task_pred.min(dim=-1, keepdim=True).values

        if self.entropy_bonus == 0.0:
            if self.distributional:
                n = int(self.ent_support.shape[0])
                ent_target = torch.full(
                    (ent_pred.shape[0], n), 1.0 / n, device=ent_pred.device, dtype=ent_pred.dtype
                )
            else:
                ent_target = torch.zeros(ent_pred.shape[0], 1, device=ent_pred.device, dtype=ent_pred.dtype)
        elif self.distributional:
            # target = discount * (e + Q_ent) ≡ reward=(discount*e), γ=discount.
            ent_target = self._c51_backup(ent_pred, discount * e, discount, self.ent_support)
        else:
            ent_target = discount * (e + ent_pred.min(dim=-1, keepdim=True).values)
        return task_target, ent_target

    def compute_loss(
        self,
        task_pred: torch.Tensor,
        ent_pred: torch.Tensor,
        task_target: torch.Tensor,
        ent_target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample task + entropy loss (no batch reduction)."""
        task_loss = self._stream_loss(task_pred, task_target, self.task_support)
        if self.entropy_bonus == 0.0:
            return task_loss
        return task_loss + self._stream_loss(ent_pred, ent_target, self.ent_support)


def build_dual_stream_critic(
    obs_dim: int,
    act_dim: int,
    *,
    distributional: bool,
    entropy_bonus: float = 1.0,
    num_atoms: int = 101,
    task_v_range: Tuple[float, float] = (-0.5, 5.0),
    ent_v_range: Tuple[float, float] = (-1.0, 1.0),
    activation: type[nn.Module] = nn.SiLU,
) -> DualStreamCritic:
    out_dim = num_atoms if distributional else 1
    module = TwinDualStreamModule(
        fn=lambda: DualHeadCriticTrunk(
            input_dim=obs_dim + act_dim,
            hidden_dim=512,
            task_output_dim=out_dim,
            ent_output_dim=out_dim,
            activation=activation,
        )
    )
    return DualStreamCritic(
        module,
        distributional=distributional,
        entropy_bonus=entropy_bonus,
        task_v_range=task_v_range,
        ent_v_range=ent_v_range,
        num_atoms=num_atoms,
    )


class NormalActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        std_max: float = 1.0,
        std_min: float = 0.001,
        action_init: Literal["zeros", "orthogonal"] = "zeros",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.in_layer = nn.Linear(obs_dim, 384)
        self.in_layer.weight._non_muon = True
        self.trunk = nn.Sequential(
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            nn.RMSNorm(384),
        )
        self.action = nn.Linear(384, act_dim * 2)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_sac_linear)
        
        if action_init == "orthogonal":
            self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
        elif action_init == "zeros":
            # zero-init following FastSAC
            nn.init.constant_(self.action.weight, 0.0) # zero-init the weight
            nn.init.constant_(self.action.bias, 0.0) # zero-init the bias
        else:
            raise ValueError(f"Invalid action_init: {action_init}")

        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.log_std_max = math.log(std_max)
        self.log_std_min = math.log(std_min)

    def forward(self, obs: torch.Tensor, ):
        feat = self.trunk(self.in_layer(obs))
        mean, raw = self.action(feat).chunk(2, dim=-1)
        # log_std = self.log_std_max - F.softplus(raw)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw))
        return mean, torch.exp(log_std)


class SAC(TensorDictModuleBase):

    # keys to select from the batch for training
    train_keys = (
        CMD_KEY, OBS_KEY, ("next", OBS_KEY), ("next", CMD_KEY), ACTION_KEY,
        REWARD_KEY, TERM_KEY, DONE_KEY, ("next", "discount"), "is_init",
    )

    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        obs_transform: Optional[SymmetryTransform] = None,
        act_transform: Optional[SymmetryTransform] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.eff_horizon = 1 / (1 - self.cfg.gamma)

        self.obs_transform = obs_transform.to(device) if obs_transform is not None else None
        self.act_transform = act_transform.to(device) if act_transform is not None else None

        self._distributed = aa.is_distributed()
        if self._distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed training is enabled but torch.distributed is not initialized."
            )

        fake = observation_spec.zero()
        preproc = []
        if CMD_KEY in observation_spec.keys(True, True):
            obs_dim = fake[OBS_KEY].shape[-1] + fake[CMD_KEY].shape[-1]
            preproc.append(CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False))
        else:
            obs_dim = fake[OBS_KEY].shape[-1]
            preproc.append(Mod(nn.Identity(), [OBS_KEY], ["_input"]))
        self.act_dim = action_spec.shape[-1]
        # since entropy grows linearly with the action dimension, we need to scale it down
        self.entropy_scale = 1.0 / math.sqrt(self.act_dim)

        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim).to(device)
        else:
            self.vecnorm_obs = nn.Identity()
        preproc.append(Mod(self.vecnorm_obs, ["_input"], ["_input_normed"]))
        self.preproc = Seq(*preproc).to(device)
        
        if (self.obs_transform is not None) and (self.act_transform is not None):
            self.has_symmetry = True
        else:
            self.has_symmetry = False

        if self.cfg.sym_aug:
            assert self.has_symmetry, "Symmetry augmentation is enabled but no symmetry transform is provided"

        if self.cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(
                self.act_dim, self.cfg.target_entropy_sigma
            )
        else:
            self.target_entropy = -0.5 * float(self.act_dim)

        if self.cfg.distributional:
            if self.cfg.normalize_reward:
                # Std-normalized returns are O(1); fixed atom support (not task-tuned).
                task_v_range = (-0.5, 5.0)
                num_atoms = 101
            else:
                task_v_range = (-1.0, 9.0)
                num_atoms = int((task_v_range[1] - task_v_range[0]) / 0.05) + 1
            # Q_ent ≈ γ/(1-γ) * α H_target * entropy_scale; pad ×2 and mirror about 0.
            e_typ = abs(self.cfg.alpha_init * self.target_entropy * self.entropy_scale)
            half = max(2.0 * (self.cfg.gamma / max(1e-8, 1.0 - self.cfg.gamma)) * e_typ, 1e-3)
            ent_v_range = (-half, half)
            self.Q = build_dual_stream_critic(
                obs_dim,
                self.act_dim,
                distributional=True,
                num_atoms=num_atoms,
                task_v_range=task_v_range,
                ent_v_range=ent_v_range,
                entropy_bonus=self.cfg.entropy_bonus,
            ).to(device)
        else:
            self.Q = build_dual_stream_critic(
                obs_dim,
                self.act_dim,
                distributional=False,
                entropy_bonus=self.cfg.entropy_bonus,
            ).to(device)

        self.DistClass = IndependentNormal
        self.actor = NormalActor(
            obs_dim,
            self.act_dim,
            std_max=1.0,
            std_min=0.001,
            action_init=self.cfg.actor_init,
        ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.actor_target.requires_grad_(False)
        self.actor_target.eval()

        self.alpha = AlphaModule(self.cfg.alpha_init).to(device)
        self.opt_alpha = torch.optim.Adam(self.alpha.parameters(), lr=self.cfg.lr_alpha)
        if self.cfg.muon:
            self.opt_actor = MuonAdamWWrapper(
                [self.actor],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.opt_Q = MuonAdamWWrapper(
                [self.Q],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            self.opt_Q = torch.optim.AdamW(self.Q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        self.global_step = 0

        self.msr = (
            MultiStepReturn(self.cfg.gamma, self.cfg.n_steps).to(device)
            if self.cfg.n_steps > 1
            else None
        )

        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                load_rms=False,
                device=self.device if isinstance(self.device, torch.device) else torch.device(self.device),
                epsilon=float(self.cfg.reward_norm_epsilon),
            )

        # Distributed wiring: wrap *after* deepcopy of targets so target nets
        # stay plain modules, and *after* ``alpha`` / optimizers exist so
        # the initial broadcast includes them. DDP shares the underlying
        # parameter tensors with the wrapped module, so optimizers built from
        # ``self.actor.parameters()`` keep pointing at the same params.
        if self._distributed:
            self._wrap_ddp(local_rank=aa.get_local_rank())
            self._broadcast_parameters()

        _dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._amp_device_type = _dev.type
        self._amp_enabled = bool(self.cfg.use_amp and _dev.type == "cuda")
        # Separate scalers so critic/actor loss scales and Inf/NaN skips stay independent.
        self.grad_scaler_Q = GradScaler(self._amp_device_type, enabled=self._amp_enabled)
        self.grad_scaler_actor = GradScaler(self._amp_device_type, enabled=self._amp_enabled)
        # self.compute_target = torch.compile(
        #     self._compute_target,
        #     mode="reduce-overhead",
        # ) # compiling sometimes give worse performance, use with caution
        self.compute_target = self._compute_target

        def fn(rew: torch.Tensor | TensorDict) -> torch.Tensor:
            if isinstance(rew, TensorDict):
                rew = torch.cat(list(rew.values()), dim=-1)
            rew = rew.sum(-1, keepdim=True)
            if self.cfg.clamp_reward:
                rew = rew.clamp_min(0.0)
            return rew
        self.reward_collate_fn = fn

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return autocast(
            device_type=self._amp_device_type,
            dtype=torch.float16,
        )

    def _wrap_ddp(self, local_rank: int) -> None:
        device = (
            torch.device(self.device)
            if not isinstance(self.device, torch.device)
            else self.device
        )
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": True,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        self.actor = wrap_ddp(self.actor, **ddp_kwargs)
        self.Q = wrap_ddp(self.Q, **ddp_kwargs)
        self.alpha = wrap_ddp(self.alpha, **ddp_kwargs)

    @torch.no_grad()
    def _broadcast_parameters(self) -> None:
        """Make rank-0's parameters/buffers the source of truth at startup.

        Includes the target networks (deepcopied locally, so their initial RNG
        state would otherwise diverge across ranks), :attr:`vecnorm_obs`, and
        :attr:`alpha`.
        """
        if not self._distributed:
            return
        for module in (
            self.vecnorm_obs,
            self.actor,
            self.actor_target,
            self.Q,
            self.Q_target,
            self.alpha,
        ):
            for param in module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    def make_tensordict_primer(self):
        """Register correlated-noise state **before** constructing :class:`SAC` so replay ``fake_tensordict`` matches rollouts."""
        from torchrl.envs import TensorDictPrimer
        from torchrl.data import UnboundedContinuous, BoundedContinuous, Composite

        shape = tuple(self.action_spec.shape)
        dev = torch.device(self.device)
        spec = {
            "prev_noise": UnboundedContinuous(shape, device=dev),
            "rho": BoundedContinuous(low=0.0, high=1.0, shape=[shape[0], 1], device=dev)
        }
        return TensorDictPrimer(
            Composite(spec, shape=[shape[0]], device=dev),
            random=self.cfg.use_correlated,
            reset_key="done",
            expand_specs=False,
        )
    
    @classmethod
    def from_env(cls, cfg: SACConfig, env: _EnvBase, device: torch.device):
        if cfg.sym_aug:
            obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform()
            act_transform = env.action_manager.symmetry_transform()
            if CMD_KEY in env.observation_spec.keys(True, True):
                cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transform()
                obs_transform = SymmetryTransform.cat([cmd_transform, obs_transform])
        else:
            obs_transform = None
            act_transform = None
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
            obs_transform=obs_transform,
            act_transform=act_transform,
        )

    def get_rollout_policy(self, mode: str = "train", critic: bool = False) -> TensorDictModuleBase:
        """Train: optional AR(1) pre-tanh rollout noise; eval/deploy: deterministic squash of the Gaussian mean."""
        policy = SACRolloutPolicy(
            self.preproc if mode == "train" else VecNorm.freeze()(self.preproc),
            self.actor,
            self.DistClass,
            use_correlated=self.cfg.use_correlated,
            Q=self.Q if critic else None,
            reward_normalizer=self.reward_normalizer,
            critic=critic,
        )
        return policy

    def on_stage_start(self, stage: str, env: _EnvBase):
        # we will not create buffer when not training
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
        )
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys = observation_keys - {"prev_noise", "rho"}
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size, fake_rb,
            fake_bootstrap=True,
            observation_keys=list(observation_keys),
        )
        print("Primary buffer:")
        print(self.rb)
        if self.cfg.prior_data is not None:
            self.rb_prior = ReplayBuffer.from_rollout(
                self.cfg.prior_data,
                fake_bootstrap=True,
                observation_keys=list(observation_keys),
            )
            self.rb_prior.compute_return(
                gamma=self.cfg.gamma, reward_collate_fn=self.reward_collate_fn
            )
            print("Prior data buffer:")
            print(self.rb_prior)
        else:
            self.rb_prior = None

        # ``self.Q`` / ``self.actor`` may be DDP-wrapped; strip the wrapper so
        # the state dict's keys match the (plain) target nets.
        self.Q_target.load_state_dict(unwrap_ddp(self.Q).state_dict())
        self.actor_target.load_state_dict(unwrap_ddp(self.actor).state_dict())

    def step(self, tensordict: TensorDict):
        """For off-policy algorithms, which typically update more frequently than
        on-policy algorithms, we do not use a collector to collect stacked transitions.

        Instead, we directly push the collected transition to the replay buffer.
        """

        self.global_step += 1
        td = tensordict.exclude(("next", "stats"), "collector")
        
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(
                reward=self.reward_collate_fn(td[REWARD_KEY]),
                terminated=td[TERM_KEY],
                truncated=td["next", "truncated"],
            )
        self.rb.push(td)

        if self.global_step > self.cfg.warm_up_steps and self.global_step % self.cfg.train_every == 0:
            return self.train_op()
        else:
            return {}

    @ScopedTimer("sac_train")
    @VecNorm.freeze()
    def train_op(self):
        # Sync per-rank running stats *before* any consumer (UTD loop /
        # actor update) reads them. VecNorm updates happen during rollouts
        # (outside ``train_op``) and the reward normalizer was just updated
        # above, so both are at the latest-but-divergent state across ranks.
        with torch.no_grad():
            if self._distributed and self.cfg.vecnorm:
                self.vecnorm_obs.synchronize(mode="broadcast")
            if self._distributed and self.cfg.normalize_reward:
                self.reward_normalizer.synchronize(mode="broadcast")

        infos: dict = {"rb_size": len(self.rb)}

        last_indices = None
        critic_iters = self.cfg.train_every * self.cfg.utd_ratio
        for i in range(critic_iters):
            # batch, last_indices = self.rb.sample_sequential(
            #     batch_size=self.cfg.critic_batch_size,
            #     steps=self.cfg.n_steps,
            #     last_indices=last_indices,
            #     sequential_prob=0.6,
            #     sequential_offset=-1,
            # )
            batch = self.rb.sample(
                batch_size=self.cfg.critic_batch_size,
                steps=self.cfg.n_steps,
                next_obs=True,
            ).to(self.device, non_blocking=True)
            if self.rb_prior is not None:
                batch_prior = self.rb_prior.sample(
                    batch_size=int(self.cfg.critic_batch_size * self.cfg.prior_data_ratio),
                    steps=self.cfg.n_steps,
                    next_obs=True,
                ).to(self.device, non_blocking=True)
            else:
                batch_prior = None
            d = i == critic_iters - 1
            info = self.train_critic(batch, batch_prior=batch_prior, diagnostics=d)
        infos.update(info)

        # actor update is delayed and has fewer iterations
        actor_iters = self.cfg.train_every
        for j in range(actor_iters):
            d = j == actor_iters - 1
            info = self.train_actor(diagnostics=d)
        infos.update(info)

        return dict(sorted(infos.items()))

    @ScopedTimer("train_critic")
    def train_critic(
        self,
        batch: TensorDict,
        batch_prior: TensorDict | None = None,
        diagnostics: bool = False,
    ):
        self.Q.train()
        batch = batch.select(*self.train_keys, inplace=True, strict=False)
        B_online = batch.shape[1]

        if batch_prior is not None:
            batch_prior = batch_prior.select(*self.train_keys, inplace=True, strict=False)
            B_prior = batch_prior.shape[1]
            batch = torch.cat([batch, batch_prior], dim=1)
        else:
            B_prior = 0
        B_eff = B_online + B_prior

        reward = self.reward_collate_fn(batch[REWARD_KEY])

        if self.cfg.debug:
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            # scale by effective horizon
            reward = reward * (1.0 - self.cfg.gamma)

        # maybe concat and normalize the observation
        self.preproc(batch)
        self.preproc(batch["next"])

        if self.cfg.n_steps == 1:
            obs = batch["_input_normed"]
            act = batch[ACTION_KEY]
            next_obs = batch["next", "_input_normed"]
            term = batch[TERM_KEY].float()
            env_disc = batch.get(("next", "discount"))
            if env_disc is None:
                env_disc = torch.ones_like(term)
            discount = self.cfg.gamma * env_disc * (1.0 - term)
            is_init = batch["is_init"]
            term_flat = batch[TERM_KEY]
            if term_flat.dim() > 1 and term_flat.shape[-1] == 1:
                term_flat = term_flat.squeeze(-1)
            terminated = term_flat.bool()
        else:
            assert self.msr is not None
            obs = batch["_input_normed"][0]
            act_n = batch[ACTION_KEY]
            env_disc_ms = batch.get(("next", "discount"))
            if env_disc_ms is not None:
                env_disc_ms = env_disc_ms[: self.msr.n_steps]
            act_n, next_obs, reward, discount, terminated = self.msr(
                actions=act_n,
                next_observations=batch["next", "_input_normed"],
                rewards=reward[:self.msr.n_steps],
                terminated=batch[TERM_KEY],
                done=batch[DONE_KEY],
                env_discount=env_disc_ms,
            )
            act = act_n[:, 0]
            is_init = batch["is_init"][0]

        with self._autocast():
            with ScopedTimer("compute_target"):
                task_target, ent_target = self.compute_target(next_obs, reward, discount)

            # as of torch 2.11, compiling loss computation leads to numerically
            # inconsistent results and degrades performance
            
            if self.cfg.sym_aug:
                # Q(s, a) = Q(s_mirror, a_mirror)
                obs_mirror = self.obs_transform(obs)
                act_mirror = self.act_transform(act)
                obs = torch.cat([obs, obs_mirror], dim=0)
                act = torch.cat([act, act_mirror], dim=0)
                task_target = torch.cat([task_target, task_target], dim=0)
                ent_target = torch.cat([ent_target, ent_target], dim=0)
                terminated = torch.cat([terminated, terminated], dim=0)
                is_init = torch.cat([is_init, is_init], dim=0)

            pred_task, pred_ent = self.Q(obs, act)
            per_sample_q_loss = self.Q.compute_loss(
                pred_task, pred_ent, task_target, ent_target
            )
            valid = (1.0 - is_init.float()).reshape_as(per_sample_q_loss)
            denom = valid.sum().clamp_min(1e-8)
            q_loss = (per_sample_q_loss * valid).sum() / denom

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_Q.scale(q_loss).backward()
            # Must unscale before all-reduce / clip / grad norm: those are only
            # meaningful on the physical (unscaled) gradients; grad_scaler.step
            # still runs Inf/NaN checks afterwards.
            self.grad_scaler_Q.unscale_(self.opt_Q)
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_Q.step(self.opt_Q)
            self.grad_scaler_Q.update()
        else:
            q_loss.backward()
            critic_grad_norm = clip_grad_norm_(self.Q.parameters(), max_norm=self.cfg.max_grad_norm)
            self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        if not diagnostics:
            return

        infos: dict = {
            "critic/q_loss": q_loss.item(),
            "critic/grad_norm": critic_grad_norm.item(),
        }
        with torch.no_grad():
            if self.cfg.n_steps > 1:
                # Online-only policy/action mismatch at t+1 (exclude prior data).
                obs_t1 = batch["_input_normed"][1, :B_online]
                act_t1 = batch[ACTION_KEY][1, :B_online]
                done_t0 = batch[DONE_KEY][0, :B_online].reshape(B_online)
                alive_t1 = ~done_t0.bool()
                if alive_t1.any():
                    policy_act_t1 = self.actor(obs_t1)[0]
                    l2_t1 = torch.linalg.vector_norm(policy_act_t1 - act_t1, dim=-1)
                    infos["critic/action_mismatch_t1"] = l2_t1[alive_t1].mean().item()

            task_pred, ent_pred = self.Q(obs, act)
            q_task = self.Q.expected(task_pred, self.Q.task_support)
            q_ent = self.Q.expected(ent_pred, self.Q.ent_support)
            if self.cfg.distributional:
                q_task_lower = self.Q.expected(task_pred, self.Q.task_support, risk_alpha=0.5)
                q_task_upper = self.Q.expected(task_pred, self.Q.task_support, risk_alpha=-0.5)
            else:
                q_task_lower = q_task_upper = None

            q_soft = self.Q.soft_q(q_task, q_ent)

            # Task stream trained on normalized rewards; denormalize task only.
            if self.reward_normalizer is not None:
                q_task_log = self.reward_normalizer.denormalize_return_values(q_task)
                q_soft_log = self.Q.soft_q(q_task_log, q_ent)
                if q_task_lower is not None:
                    q_task_lower = self.reward_normalizer.denormalize_return_values(q_task_lower)
                    q_task_upper = self.reward_normalizer.denormalize_return_values(q_task_upper)
            else:
                q_task_log = q_task
                q_soft_log = q_soft

            if q_task_lower is not None:
                infos["critic/q_task_lower"] = q_task_lower.mean().item()
                infos["critic/q_task_upper"] = q_task_upper.mean().item()

            q_online = q_soft_log[:B_online]
            infos["critic/q_value"] = q_online.mean().item()
            infos["critic/q_max"] = q_online.max().item()
            infos["critic/q_std"] = q_online.std(dim=-1).mean().item()
            infos["critic/q_soft"] = infos["critic/q_value"]
            infos["critic/q_task"] = q_task_log[:B_online].mean().item()
            infos["critic/q_ent"] = q_ent[:B_online].mean().item()
        
        if B_prior > 0:
            q_prior = q_soft_log[B_online: B_eff]
            infos["critic/prior_q_mean"] = q_prior.mean().item()
            infos["critic/prior_q_max"] = q_prior.max().item()
        
        if terminated.any():
            term_mask = terminated.reshape(q_soft_log.shape[0])
            infos["critic/q_value_terminated"] = q_soft_log[term_mask].mean().item()
            infos["critic/q_loss_terminated"] = per_sample_q_loss[term_mask].mean().item()

        return infos

    @torch.no_grad()
    def _compute_target(
        self, next_obs: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # actions are sampled with uncorrelated noise
        loc, scale = self.actor_target(next_obs)
        dist = self.DistClass(loc, scale)
        next_action = dist.sample()

        next_log_prob = dist.log_prob(next_action)
        alpha = self.alpha()
        # Unscaled entropy bonus for the entropy-stream TD (λ applied in soft Q only).
        ent_bonus = (-alpha * next_log_prob).reshape_as(reward) * self.entropy_scale
        noise = torch.randn_like(next_action).clamp(-3.0, 3.0)
        return unwrap_ddp(self.Q_target).compute_targets(
            next_obs,
            next_action + noise * self.cfg.target_action_noise,
            reward,
            discount,
            ent_bonus,
        )

    @ScopedTimer("train_actor")
    def train_actor(self, diagnostics: bool = False):
        # actor training does not need next observations
        batch = (
            self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1, next_obs=False)
            .to(self.device)
            .select(*self.train_keys, strict=False) # [N,]
        )
        B_online = batch.shape[0]
        if self.rb_prior is not None:
            batch_prior = self.rb_prior.sample(
                batch_size=int(self.cfg.actor_batch_size * self.cfg.prior_data_ratio),
                steps=1,
                next_obs=False,
            ).to(self.device)
            B_prior = batch_prior.shape[0]
            G = batch_prior["G"]
            steps_to_go = batch_prior["steps_to_go"]
            batch_prior = batch_prior.select(*self.train_keys, strict=False)
            batch = torch.cat([batch, batch_prior], dim=0)
        else:
            B_prior = 0

        self.preproc(batch)
        obs = batch["_input_normed"]
        act = batch[ACTION_KEY]
        is_init = batch["is_init"]

        if self.cfg.sym_aug:
            obs_mirror = self.obs_transform(obs)
            act_mirror = self.act_transform(act)
            obs = torch.cat([obs, obs_mirror], dim=0)
            act = torch.cat([act, act_mirror], dim=0)
            is_init = torch.cat([is_init, is_init], dim=0)

        with hold_out_net(self.Q), self._autocast():
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale)
            action_update = dist.rsample((4,))  # [4, N, D]
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            # Actor path: one forward for soft min (avoids dropout mismatch).
            act_k = einops.rearrange(action_update, "k n d -> n k d")
            task_pred, ent_pred = self.Q(obs, act_k)
            q_task_v = self.Q.expected(task_pred, self.Q.task_support)
            q_ent_v = self.Q.expected(ent_pred, self.Q.ent_support)
            q = self.Q.soft_q(q_task_v, q_ent_v, clip=True).squeeze(-1)
            policy_term = -q.mean(dim=1)

            bc_term = torch.zeros_like(policy_term)
            bc_gate = None
            advantage = None
            horizon_mask = None
            if self.rb_prior is not None:
                # Gate on task Q (hard-return scale) vs demo MC; entropy stream is tiny.
                prior_sl = slice(B_online, B_online + B_prior)
                horizon_mask = steps_to_go > self.eff_horizon
                q_prior = q_task_v.detach()[prior_sl].min(dim=-1).values.mean(
                    dim=-1, keepdim=True
                )
                if self.reward_normalizer is not None:
                    q_prior = self.reward_normalizer.denormalize_return_values(q_prior)
                G_log = G * (1.0 - self.cfg.gamma)
                assert q_prior.shape == G_log.shape, f"{q_prior.shape} != {G_log.shape}"
                advantage = q_prior - G_log
                bc_gate = torch.relu(-advantage).clamp_max(1.0) * horizon_mask.float()
                prior_dist = self.DistClass(loc[prior_sl], scale[prior_sl])
                bc = prior_bc_loss(
                    self.cfg.bc_loss,
                    action_pred=action_update[:, prior_sl],
                    action_demo=batch_prior[ACTION_KEY],
                    dist=prior_dist,
                )
                bc_term[prior_sl] = bc * bc_gate.squeeze(-1)

        alpha = self.alpha()
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term) * self.entropy_scale)
            + 0.01 * ((loc/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
            + self.cfg.bc_coef * bc_term
        )
        valid = (1.0 - is_init.float()).reshape_as(actor_loss)
        denom = valid.sum().clamp_min(1e-8)
        actor_loss = (actor_loss * valid).sum() / denom

        q_action_grad_norm: torch.Tensor | None = None
        if diagnostics:
            (grad_q_wrt_a,) = torch.autograd.grad(
                q.sum(),
                action_update,
                retain_graph=True,
                create_graph=False,
            )
            q_action_grad_norm = grad_q_wrt_a.norm(dim=-1).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_actor.scale(actor_loss).backward()
            self.grad_scaler_actor.unscale_(self.opt_actor)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_actor.step(self.opt_actor)
            self.grad_scaler_actor.update()
        else:
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        if not diagnostics:
            return 

        assert q_action_grad_norm is not None
        with torch.no_grad():
            if self.reward_normalizer is not None:
                q_task_log = self.reward_normalizer.denormalize_return_values(q_task_v)
                q_for_log = self.Q.soft_q(q_task_log, q_ent_v, clip=True).squeeze(-1)
            else:
                q_for_log = q
            # prior data may not contain "loc" key
            # mean_change = (dist.loc[: batch.shape[0]] - batch["loc"]).abs().mean()
            infos = {
                "actor/loss": actor_loss.item(),
                "actor/grad_norm": actor_grad_norm.item(),
                "actor/alpha": alpha.item(),
                "actor/entropy": entropy_est.mean().item(),
                # "actor/mean_change": mean_change.item(),
                "actor/q_std": q_for_log.std(dim=1).mean().item(),
                "actor/q_action_grad_norm": q_action_grad_norm.item(),
                "actor/mean_loc": loc.abs().mean().item(),
                "actor/mean_scale": scale.mean().item(),
            }
            if self.rb_prior is not None:
                assert advantage is not None and bc_gate is not None and horizon_mask is not None
                infos["rlpd/bc_term"] = bc_term[B_online: B_online + B_prior].mean().item()
                infos["rlpd/bc_frac"] = bc_gate.float().mean().item()
                if horizon_mask.any():
                    infos["rlpd/online_advantage"] = advantage[horizon_mask].mean().item()
                else:
                    infos["rlpd/online_advantage"] = float("nan")

        if self.has_symmetry:
            with torch.no_grad():
                _obs = obs[:batch.shape[0]]
                mean_mirror_obs = self.actor(self.obs_transform(_obs))[0]
                mean_mirrot_act = self.act_transform(self.actor(_obs)[0])
            infos["actor/symmetry_loss"] = (mean_mirror_obs - mean_mirrot_act).square().mean().item()

        return infos

    def state_dict(self):
        state_dict = OrderedDict()
        # Save the underlying modules so checkpoints are portable between
        # distributed and single-process runs.
        state_dict["Q"] = unwrap_ddp(self.Q).state_dict()
        state_dict["actor"] = unwrap_ddp(self.actor).state_dict()
        state_dict["alpha"] = unwrap_ddp(self.alpha).state_dict()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        if self.reward_normalizer is not None:
            state_dict["reward_normalizer"] = self.reward_normalizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        unwrap_ddp(self.Q).load_state_dict(state_dict["Q"], strict=strict)
        unwrap_ddp(self.actor).load_state_dict(state_dict["actor"], strict=strict)
        if "opt_alpha" in state_dict:
            self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        alpha = unwrap_ddp(self.alpha)
        if "alpha" in state_dict:
            alpha.load_state_dict(state_dict["alpha"], strict=strict)
        elif "log_alpha" in state_dict:
            alpha.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])
        rk = state_dict.get("reward_normalizer")
        if self.reward_normalizer is not None and rk is not None:
            self.reward_normalizer.load_state_dict(rk)


class SACRolloutPolicy(TensorDictModuleBase):
    """Rollout policy for SAC with optional AR(1) pre-tanh noise and Q logging."""

    def __init__(
        self,
        preproc: nn.Module,
        actor: nn.Module,
        DistClass: type[torch.distributions.Distribution],
        *,
        use_correlated: bool = True,
        Q: nn.Module | None = None,
        reward_normalizer: RewardNormalizer | None = None,
        critic: bool = False,
    ):
        super().__init__()
        self.preproc = preproc
        self.actor = actor
        self.DistClass = DistClass
        self.use_correlated = use_correlated
        self.Q = Q
        self.reward_normalizer = reward_normalizer
        self.critic = critic

        in_keys = [OBS_KEY]
        out_keys = [ACTION_KEY, "loc"]
        if self.use_correlated:
            in_keys = in_keys + ["prev_noise", "rho"]
            out_keys = out_keys + ["next", "prev_noise"]
        if self.critic is not None:
            out_keys = out_keys + ["Q_value"]
        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.preproc(tensordict)
        obs = tensordict["_input_normed"]
        loc, scale = self.actor(obs)
        dist = self.DistClass(loc, scale)

        if interaction_type() == InteractionType.MODE:
            sample = loc.clone()
        elif self.use_correlated:
            prev_noise = tensordict["prev_noise"]
            rho = tensordict["rho"]
            noise = (
                rho * prev_noise
                + torch.sqrt((1.0 - rho.square())) * torch.randn_like(loc).clamp(-3.0, 3.0)
            )
            sample = loc + noise * scale
            tensordict["next", "prev_noise"] = noise
        else:
            sample = dist.sample()

        if isinstance(dist, FasterTransformedDistribution):
            for transform in dist.transforms:
                sample = transform(sample)

        if self.critic and self.Q is not None:
            task_pred, ent_pred = self.Q(obs, sample)
            q_task = self.Q.expected(task_pred, self.Q.task_support)
            q_ent = self.Q.expected(ent_pred, self.Q.ent_support)
            if self.reward_normalizer is not None:
                q_task = self.reward_normalizer.denormalize_return_values(q_task)
            qs = self.Q.soft_q(q_task, q_ent).mean(dim=-1)
            tensordict["Q_value"] = qs

        tensordict[ACTION_KEY] = sample
        tensordict["loc"] = loc
        return tensordict
