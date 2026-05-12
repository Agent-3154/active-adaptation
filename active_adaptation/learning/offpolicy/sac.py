import copy
import math
import einops
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
)

from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

from active_adaptation.learning.modules import VecNorm, IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import (
    ValueDistribution,
    expected_from_logits,
    cvar_from_logits,
)
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.distribution import ScaledTanhNormal, FasterTransformedDistribution
from active_adaptation.learning.offpolicy.network import ConditionalBlock
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.dormancy import DormancyTracker
from active_adaptation.utils.profiling import ScopedTimer

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
    _target_: str = "active_adaptation.learning.offpolicy.sac.SAC"
    name: str = "sac"
    train_every: int = 4
    buffer_size: int = 2000
    warm_up_steps: int = 200
    lr: float = 5e-4
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper` (see ``ppo_symaug``).
    muon: bool = False
    weight_decay: float = 0.02
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    # architecture
    actor_init: str = "zeros"
    init_upscale: float = 2.0
    actor_layer_norm: Any = "pre"
    critic_layer_norm: Any = "pre"
    distributional: bool = True
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    sym_aug: bool = False
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # AR(1) pre-tanh exploration noise on rollout only: eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * N(0,I).
    # 0 disables correlation (standard :meth:`ScaledTanhNormal.sample`-equivalent path). Critic/actor still use iid.
    use_correlated: bool = True
    # sac specific
    entropy_bonus: float = 1.0
    alpha_init: float = 4e-3
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common heuristic for tanh-squashed SAC).
    target_entropy_sigma: float | None = 0.15
    soft_bound: float = math.pi

    tau_actor: float = 0.1 # a relatively large value for faster convergence
    tau_Q: float = 0.02  # a relatively large value for faster convergence
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); GradScaler for critic, V head, standalone train_v, and actor (alpha stays fp32).
    use_amp: bool = True
    # Prioritized replay (same API as off-policy ReplayBuffer): None disables PER.
    per_alpha: float | None = None
    per_beta: float = 0.6
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = True
    normalized_G_max: float = 5.0
    reward_norm_epsilon: float = 1e-8

    in_keys: Tuple[str, ...] = (OBS_KEY, ACTION_KEY)


cs.store(name="sac", node=SACConfig, group="algo")
# cs.store(name="dsac", node=SACConfig, group="algo") # distributional SAC


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        activation: type[nn.Module] = nn.SiLU,
        layer_norm: Literal["pre", "post", None] = "pre"
    ):
        super().__init__()
        critic_input_dim = obs_dim + act_dim
        self.critic_1 = nn.Sequential(
            nn.Linear(critic_input_dim, 512), activation(),
            nn.Linear(512, 512), activation(),
            nn.Linear(512, 512), activation(),
            nn.RMSNorm(512),
            nn.Linear(512, 1),
        )
        self.critic_2 = nn.Sequential(
            nn.Linear(critic_input_dim, 512), activation(),
            nn.Linear(512, 512), activation(),
            nn.Linear(512, 512), activation(),
            nn.RMSNorm(512),
            nn.Linear(512, 1),
        )
        for c in (self.critic_1, self.critic_2):
            c[-1].weight._non_muon = True
        self.reset_parameters()
    
    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return torch.cat([q1, q2], dim=-1)
    
    def get_values(
        self,
        obs: torch.Tensor,  # [B, obs_dim]
        act: torch.Tensor,  # [B, act_dim] or [B, K, act_dim] for multiple actions
    ) -> torch.Tensor:
        """Twin Q-head scalars: shape ``[..., 2]`` (broadcast same as :meth:`forward`)."""
        if act.dim() == 2:
            return self(obs, act)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_exp = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, obs.shape[-1])
            act_flat = act.reshape(b * k, act.shape[-1])
            qs = self.forward(obs_exp, act_flat)
            return qs.reshape(b, k, 2)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")
    
    def compute_loss(
        self,
        qs: torch.Tensor,
        q_target: torch.Tensor,
    ) -> torch.Tensor:
        """Twin Q regression to scalar Bellman target (mean over batch)."""
        return (qs - q_target).square().sum(dim=-1).mean()


class TwinDistributionalQNetwork(nn.Module):
    """Twin C51-style critics: logits per atom, shared discrete support (see td3dist / FastSAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        activation: str| type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        if num_atoms < 3:
            raise ValueError("num_atoms must be > 2 for distributional Q.")
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        critic_input_dim = obs_dim + act_dim
    
        def make_critic():
            in_layer = nn.Linear(critic_input_dim, 512)
            in_layer.weight._non_muon = True
            out_layer = nn.Linear(512, num_atoms)
            out_layer.weight._non_muon = True
            critic = nn.Sequential(
                in_layer,
                ConditionalBlock(hidden_dim=512, activation=activation),
                ConditionalBlock(hidden_dim=512, activation=activation),
                nn.RMSNorm(512),
                out_layer,
            )
            critic.apply(_init_sac_linear)
            return critic

        self.critic_1 = make_critic()
        self.critic_2 = make_critic()

        self.register_buffer(
            "q_support",
            torch.linspace(v_min, v_max, num_atoms),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        z1 = self.critic_1(x)
        z2 = self.critic_2(x)
        return torch.cat([z1, z2], dim=-1)

    def get_values(
        self,
        obs: torch.Tensor,  # [B, obs_dim]
        act: torch.Tensor,  # [B, act_dim] or [B, K, act_dim] for multiple actions
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expected Q per twin head: shape ``[..., 2]`` (from logits / categorical support)."""
        if act.dim() == 2:
            return self.expected_values(self(obs, act), risk_alpha)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_exp = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, obs.shape[-1])
            act_flat = act.reshape(b * k, act.shape[-1])
            logits = self.forward(obs_exp, act_flat)
            ev = self.expected_values(logits, risk_alpha)
            return ev.reshape(b, k, 2)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")

    def compute_loss(
        self,
        qs_logits: torch.Tensor,
        target_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of categorical cross-entropies for both twins versus ``target_dist`` (mean over batch)."""
        q1, q2 = qs_logits.chunk(2, dim=-1)
        log_p1 = F.log_softmax(q1, dim=-1).clamp(min=-30.0)
        log_p2 = F.log_softmax(q2, dim=-1).clamp(min=-30.0)
        return - ((target_dist * log_p1).sum(-1) + (target_dist * log_p2).sum(-1))

    def expected_values(
        self,
        logits_pair: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expected Q under softmax for each twin: logits_pair [B, 2 * num_atoms] -> [B, 2]."""
        log1, log2 = logits_pair.chunk(2, dim=-1)
        if risk_alpha is not None:
            e1 = cvar_from_logits(log1, self.q_support, risk_alpha)
            e2 = cvar_from_logits(log2, self.q_support, risk_alpha)
        else:
            e1 = expected_from_logits(log1, self.q_support)
            e2 = expected_from_logits(log2, self.q_support)
        return torch.cat([e1, e2], dim=-1)

    def bellman_projection(
        self,
        next_logits: torch.Tensor,
        rewards: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> torch.Tensor:
        """Categorical projection (Bellman backup onto the fixed support)."""
        return ValueDistribution(next_logits, self.q_support).project(rewards, discount)



class _SACDormancyScope(nn.Module):
    """Modules exercised during SAC rollout + learner forwards (:class:`DormancyTracker` hooks)."""

    def __init__(
        self,
        actor: nn.Module,
        q_online: nn.Module,
    ):
        super().__init__()
        self.actor = actor
        self.Q = q_online


class TanhNormalActor(nn.Module):
    """Policy trunk + Gaussian + tanh squash (same layout as blade_runner SAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_norm: str = None,
        std_max: float = 1.0,
        std_min: float = 0.001,
        action_init: Literal["zeros", "orthogonal"] = "zeros",
        init_upscale: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.in_layer = nn.Linear(obs_dim, 384)
        self.in_layer.weight._non_muon = True
        self.trunk = nn.Sequential(
            ConditionalBlock(hidden_dim=384, condition_dim=0),
            ConditionalBlock(hidden_dim=384, condition_dim=0),
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
        
        self.upscale: torch.Tensor
        self.register_buffer("upscale", torch.ones(act_dim) * init_upscale)
        
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
    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.env = env

        fake = observation_spec.zero()
        obs_dim = fake[OBS_KEY].shape[-1]
        act_dim = action_spec.shape[-1]

        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim, decay=1.0).to(device)
        else:
            self.vecnorm_obs = nn.Identity()
        
        try:
            self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(device)
            self.act_transform = env.action_manager.symmetry_transform().to(device)
            self.has_symmetry = True
        except (NotImplementedError, AttributeError) as e:
            if self.cfg.sym_aug:
                raise ValueError(f"Symmetry augmentation is not supported for this environment: {e}")
            self.has_symmetry = False

        if self.cfg.distributional:
            if self.cfg.normalize_reward:
                v_min = -0.5 # we will not have negative values, but it is a good idea to have a small margin
                v_max = float(self.cfg.normalized_G_max)
                num_atoms = 101
            else:
                v_min, v_max = -1.0, 9.0
                num_atoms = int((v_max - v_min) / 0.05) + 1
            self.Q = TwinDistributionalQNetwork(
                obs_dim,
                act_dim,
                num_atoms=num_atoms,
                v_min=v_min, # we actually do not have negative values, but it is a good idea to have a small margin
                v_max=v_max,
            ).to(device)
        else:
            self.Q = TwinQNetwork(obs_dim, act_dim, layer_norm=self.cfg.critic_layer_norm).to(device)

        # self.DistClass = ScaledTanhNormal
        self.DistClass = lambda loc, scale, upscale: IndependentNormal(loc, scale)
        self.actor = TanhNormalActor(
            obs_dim,
            act_dim,
            layer_norm=self.cfg.actor_layer_norm,
            std_max=1.0,
            std_min=0.001,
            action_init=self.cfg.actor_init,
            init_upscale=self.cfg.init_upscale,
        ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target.requires_grad_(False)
        self.actor_target.requires_grad_(False)

        if self.cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(
                act_dim, self.cfg.target_entropy_sigma
            )
        else:
            self.target_entropy = -float(act_dim)
        self.target_entropy = 0.0
        self.log_alpha = nn.Parameter(torch.tensor(math.log(self.cfg.alpha_init), device=device))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
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

        if env is None:
            raise ValueError("SAC requires env for ReplayBuffer layout (fake_tensordict).")
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
            # .exclude(("next", OBS_KEY))
            .detach()
            # .cpu()
        )
        fake_rb[REWARD_KEY] = fake_rb[REWARD_KEY].sum(-1, keepdim=True)
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        per_kw: dict[str, Any] = {}
        if self.cfg.per_alpha is not None:
            per_kw.update(
                per_alpha=self.cfg.per_alpha,
                per_beta=self.cfg.per_beta,
            )
        self.rb = ReplayBuffer(self.cfg.buffer_size, fake_rb, **per_kw)
        self.msr = (
            MultiStepReturn(self.cfg.gamma, self.cfg.n_steps).to(device)
            if self.cfg.n_steps > 1
            else None
        )

        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                G_max=float(self.cfg.normalized_G_max),
                load_rms=False,
                device=self.device if isinstance(self.device, torch.device) else torch.device(self.device),
                epsilon=float(self.cfg.reward_norm_epsilon),
            )

        scope = _SACDormancyScope(
            self.actor,
            self.Q,
        )
        self._dormancy_tracker = DormancyTracker(scope)

        _dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._amp_device_type = _dev.type
        self._amp_enabled = bool(self.cfg.use_amp and _dev.type == "cuda")
        self.grad_scaler = GradScaler(self._amp_device_type, enabled=self._amp_enabled)

    def _autocast(self):
        return autocast(
            device_type=self._amp_device_type,
            dtype=torch.float16,
            enabled=self._amp_enabled,
        )

    def _flush_dormancy(self, infos: dict) -> None:
        dormancy = self._dormancy_tracker.compute_dormancy(0.02)
        for module_name, value in dormancy.items():
            infos[f"dormancy/{module_name}"] = value
        self._dormancy_tracker.reset()

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

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        """Train: optional AR(1) pre-tanh rollout noise; eval/deploy: deterministic squash of the Gaussian mean."""

        def policy(tensordict: TensorDict):
            obs = self.vecnorm_obs(tensordict[OBS_KEY])
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale, upscale=self.actor.upscale)

            if self.cfg.use_correlated:
                prev_noise = tensordict["prev_noise"]
                rho = tensordict["rho"]
                noise = (
                    rho * prev_noise 
                    + torch.sqrt((1.0 - rho.square())) * torch.randn_like(loc)
                )
                sample = loc + noise * scale
                tensordict["next", "prev_noise"] = noise
                if isinstance(dist, FasterTransformedDistribution):
                    for transform in dist.transforms:
                        sample = transform(sample)
            else:
                sample = dist.sample()

            tensordict[ACTION_KEY] = sample # + 0.04 * torch.randn_like(sample)
            tensordict["loc"] = loc
            return tensordict

        return self._dormancy_tracker.wrap(policy)

    def on_stage_start(self, stage: str):
        self.enable_actor = True
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        self.global_step += self.cfg.train_every

        td = tensordict.exclude(("next", "stats"), "collector")
        # td = td.exclude(("next", OBS_KEY))

        reward = td[REWARD_KEY]
        # KEEP THIS FOR DEBUGGING
        if self.cfg.debug:
            # debug: constant reward scaled by effective horizon
            # the value should converge to 1.0 in this case
            # multi-step return should significantly speed up convergence
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
            neg_rew_ratio = 0.0
        else:
            reward = reward.sum(-1, keepdim=True)
            neg_rew_ratio = (reward <= 0.).float().mean().item()
            reward = reward.clamp_min(0.)
        td[REWARD_KEY] = reward

        bs = td.batch_size
        # StackingCollector stacks steps on batch dim 1: [num_envs, horizon, …].
        for ti in range(int(bs[1])):
            sub = td[:, ti]
            if self.reward_normalizer is not None:
                self.reward_normalizer.update_reward_stats(
                    reward=sub[REWARD_KEY],
                    terminated=sub[TERM_KEY],
                    truncated=sub["next", "truncated"],
                )
            self.rb.push(sub)

        infos: dict = {"rb_size": len(self.rb), "critic/neg_rew_ratio": neg_rew_ratio}
        if self.global_step < self.cfg.warm_up_steps:
            self._flush_dormancy(infos)
            return infos

        with self._dormancy_tracker.track():
            last_indices = None
            iters = self.cfg.train_every * self.cfg.utd_ratio
            for i in range(iters):
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
                ).to(self.device)
                with ScopedTimer("train_critic"):
                    d = i == iters - 1
                    info = self.train_critic(batch, diagnostics=d)
            infos.update(info)

            if self.enable_actor:
                for j in range(self.cfg.train_every):
                    d = j == self.cfg.train_every - 1
                    with ScopedTimer("train_actor"):
                        info = self.train_actor(diagnostics=d)
                infos.update(info)

        # if self.global_step % self.cfg.v_update_every == 0:
        #     for _ in range(self.cfg.v_inner):
        #         infos.update(self.train_v())

        self._flush_dormancy(infos)
        return dict(sorted(infos.items()))

    def train_critic(self, batch: TensorDict, diagnostics: bool = False):
        self.Q.train()
        reward = batch[REWARD_KEY]
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)

        if self.cfg.n_steps == 1:
            obs = batch[OBS_KEY]
            act = batch[ACTION_KEY]
            next_obs = batch["next", OBS_KEY]
            discount = self.cfg.gamma * (1.0 - batch[TERM_KEY].float())
            is_init = batch["is_init"]
            term_flat = batch[TERM_KEY]
            if term_flat.dim() > 1 and term_flat.shape[-1] == 1:
                term_flat = term_flat.squeeze(-1)
            terminated = term_flat.bool()
        else:
            assert self.msr is not None
            batch_done = batch[DONE_KEY][:self.msr.n_steps]
            batch_term = batch[TERM_KEY][:self.msr.n_steps]
            if (next_obs := batch.get(("next", OBS_KEY))) is None:
                assert batch.shape[0] == self.msr.n_steps + 1
                next_obs = torch.where(
                    batch_done,
                    batch[OBS_KEY][:self.msr.n_steps], # repeat the last obs as the terminal obs
                    batch[OBS_KEY][1:self.msr.n_steps+1],
                )
            obs = batch[OBS_KEY][0]
            act = batch[ACTION_KEY][0]
            next_obs, reward, discount, terminated = self.msr(
                next_obs,
                reward[:self.msr.n_steps],
                batch_term,
                batch_done,
            )
            is_init = batch["is_init"][0]

        obs = self.vecnorm_obs(obs)
        next_obs = self.vecnorm_obs(next_obs)

        B_eff = obs.shape[0]

        weight = batch["priority_weight"]
        replay_flat_idx = batch["replay_flat_index"].long()
        if weight.ndim == 2:
            weight = weight[0].contiguous()
            replay_flat_idx = replay_flat_idx[0].contiguous()
        weight = weight.to(device=self.device, dtype=torch.float32)
        ri_base_cpu = replay_flat_idx.detach().cpu() if self.rb.prioritized else None

        importance_weights_base = weight
        importance_weights = weight.clone()

        with self._autocast():
            with torch.no_grad():
                # actions are sampled with uncorrelated noise
                loc, scale = self.actor_target(next_obs)
                dist = self.DistClass(loc, scale, upscale=self.actor.upscale)
                next_action = dist.sample()

                next_log_prob = dist.log_prob(next_action)
                target_action = next_action + torch.randn_like(next_action) * self.cfg.target_action_noise
                alpha = self.log_alpha.exp()
                lp = next_log_prob.reshape_as(reward)

                if self.cfg.distributional:
                    # Fold soft Bellman entropy into rewards, then categorical projection (FastSAC-style).
                    adjusted_reward = reward + discount * self.cfg.entropy_bonus * (-alpha * lp)
                    next_logits = self.Q_target(next_obs, target_action)
                    n1, n2 = next_logits.chunk(2, dim=-1)
                    p1 = self.Q_target.bellman_projection(n1, adjusted_reward, discount)
                    p2 = self.Q_target.bellman_projection(n2, adjusted_reward, discount)
                    z = self.Q_target.q_support.to(
                        device=p1.device, dtype=p1.dtype
                    ).view(1, -1)
                    ev1 = (p1 * z).sum(-1, keepdim=True)
                    ev2 = (p2 * z).sum(-1, keepdim=True)
                    q_target = torch.where(ev1 < ev2, p1, p2)
                else:
                    entropy_bonus = (-alpha * lp).reshape_as(reward)
                    target_qs = self.Q_target(next_obs, target_action)
                    target_q = target_qs.mean(dim=-1, keepdim=True)
                    q_target = reward + discount * (
                        target_q + self.cfg.entropy_bonus * entropy_bonus
                    )

            if self.cfg.sym_aug:
                # Q(s, a) = Q(s_mirror, a_mirror)
                obs_mirror = self.obs_transform(obs)
                act_mirror = self.act_transform(act)
                obs = torch.cat([obs, obs_mirror], dim=0)
                act = torch.cat([act, act_mirror], dim=0)
                q_target = torch.cat([q_target, q_target], dim=0)
                terminated = torch.cat([terminated, terminated], dim=0)
                is_init = torch.cat([is_init, is_init], dim=0)
                importance_weights = torch.cat(
                    [importance_weights_base, importance_weights_base], dim=0
                )

            qs: torch.Tensor = self.Q(obs, act)
            if self.cfg.distributional:
                per_sample_q_loss = self.Q.compute_loss(qs, q_target)
            else:
                per_sample_q_loss = (qs - q_target).square().sum(dim=-1)
            valid = (1.0 - is_init.float()).reshape_as(per_sample_q_loss)
            denom = (importance_weights * valid).sum().clamp_min(1e-8)
            q_loss = (per_sample_q_loss * importance_weights * valid).sum() / denom

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler.scale(q_loss).backward()
            # Must unscale before clip / grad norm: clip_grad_norm_ and the logged norm are only
            # meaningful on the physical (unscaled) gradients; grad_scaler.step still runs Inf/NaN checks.
            self.grad_scaler.unscale_(self.opt_Q)
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler.step(self.opt_Q)
            self.grad_scaler.update()
        else:
            q_loss.backward()
            critic_grad_norm = clip_grad_norm_(self.Q.parameters(), max_norm=self.cfg.max_grad_norm)
            self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        if self.rb.prioritized:
            with torch.no_grad():
                if self.cfg.distributional:
                    prio_src = per_sample_q_loss.detach()[:B_eff].float().cpu()
                else:
                    prio_src = (
                        (
                            qs.detach()[:B_eff] - q_target.detach()[:B_eff]
                        )
                        .abs()
                        .mean(dim=-1)
                        .float()
                        .cpu()
                    )
                self.rb.update_priority(ri_base_cpu, prio_src)

        infos: dict = {"critic/q_loss": q_loss.item()}
        if diagnostics:
            with torch.no_grad():
                logits = self.Q(obs.detach(), act.detach())
                q = self.Q.expected_values(logits)
                q_lower = self.Q.expected_values(logits, risk_alpha=0.5)
                q_upper = self.Q.expected_values(logits, risk_alpha=-0.5)

            q_val_mean = q.mean().item()
            q_val_max = q.max().item()
            q_val_std = q.std(dim=-1).mean().item()

            infos["critic/q_value"] = q_val_mean
            infos["critic/q_lower"] = q_lower.mean().item()
            infos["critic/q_upper"] = q_upper.mean().item()
            infos["critic/q_max"] = q_val_max
            infos["critic/q_std"] = q_val_std
            infos["critic/grad_norm"] = critic_grad_norm.item()
            if terminated.any():
                q_val_terminated = q[terminated.reshape(q.shape[0])]
                infos["critic/q_value_terminated"] = q_val_terminated.mean().item()

        return infos

    def train_actor(self, diagnostics: bool = False):
        batch = self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1).to(
            self.device
        ) # [N,]

        weight = batch["priority_weight"]
        if weight.ndim == 2:
            weight = weight[0].contiguous()
        weight = weight.to(device=self.device, dtype=torch.float32)
        importance_weights_base = weight
        importance_weights = weight.clone()

        obs = batch[OBS_KEY]
        obs = self.vecnorm_obs(obs)
        act = batch[ACTION_KEY]
        is_init = batch["is_init"]

        if self.cfg.sym_aug:
            obs_mirror = self.obs_transform(obs)
            act_mirror = self.act_transform(act)
            obs = torch.cat([obs, obs_mirror], dim=0)
            act = torch.cat([act, act_mirror], dim=0)
            is_init = torch.cat([is_init, is_init], dim=0)
            importance_weights = torch.cat(
                [importance_weights_base, importance_weights_base], dim=0
            )

        with hold_out_net(self.Q), self._autocast():
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale, upscale=self.actor.upscale)
            action_update = dist.rsample((4,))  # [4, N, D]
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            q = self.Q.get_values(
                obs,
                einops.rearrange(action_update, "k n d -> n k d"),
            ).mean(dim=-1)
            policy_term = -q.mean(dim=1)

        alpha = self.log_alpha.exp()
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term))
            + 0.01 * ((loc/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
        )
        valid = (1.0 - is_init.float()).reshape_as(actor_loss)
        denom = (importance_weights * valid).sum().clamp_min(1e-8)
        actor_loss = (actor_loss * importance_weights * valid).sum() / denom

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
            self.grad_scaler.scale(actor_loss).backward()
            self.grad_scaler.unscale_(self.opt_actor)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler.step(self.opt_actor)
            self.grad_scaler.update()
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
        mean_change = (
            (dist.loc[: batch.shape[0]].detach() - batch["loc"]).abs().mean()
        )
        infos = {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/alpha": alpha.detach().item(),
            "actor/entropy": entropy_est.mean().item(),
            "actor/mean_change": mean_change.item(),
            "actor/q_std": q.std(dim=1).mean().item(),
            "actor/q_action_grad_norm": q_action_grad_norm.item(),
            "actor/mean_loc": loc.abs().mean().item(),
            "actor/mean_scale": scale.mean().item(),
        }

        actor_diagnostics = {}
        if isinstance(dist, ScaledTanhNormal):
            eps = 0.05
            with torch.no_grad():
                tanh_grad = 1.0 - (action_update.detach() / dist.upscale).square()
                action_saturation = (1.0 - action_update.detach().abs() / dist.upscale < eps)
                mean_squashed = torch.tanh(dist.loc.detach() / dist.upscale) * dist.upscale
                mean_saturation = (1.0 - mean_squashed.abs() / dist.upscale < eps)
                # mean saturation per action dimension
                dim_saturation = mean_saturation.float().mean(dim=0)
            actor_diagnostics = {
                "actor/action_saturation": action_saturation.float().mean().item(),
                "actor/mean_saturation": mean_saturation.float().mean().item(),
                "actor/max_saturation": dim_saturation.max().item(),
                "actor/tanh_grad": tanh_grad.mean().item(),
                "actor/upscale": dist.upscale.mean().item(),
            }
            # self.actor.upscale.add_((dim_saturation > 0.15).float() * 5e-4)
        
        if self.has_symmetry:
            with torch.no_grad():
                _obs = obs[:batch.shape[0]]
                mean_mirror_obs = self.actor(self.obs_transform(_obs))[0]
                mean_mirrot_act = self.act_transform(self.actor(_obs)[0])
            infos["actor/symmetry_loss"] = (mean_mirror_obs - mean_mirrot_act).square().mean().item()

        infos.update(actor_diagnostics)
        return infos

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["Q"] = self.Q.state_dict()
        state_dict["actor"] = self.actor.state_dict()
        state_dict["opt_alpha"] = self.opt_alpha.state_dict()
        state_dict["log_alpha"] = self.log_alpha.detach()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.Q.load_state_dict(state_dict["Q"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        self.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])

