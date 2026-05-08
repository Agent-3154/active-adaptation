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

from active_adaptation.learning.modules import ResidualMLP, MLP, VecNorm, SimbaMLP, IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    GAE,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import (
    ValueDistribution,
    expected_q_from_logits,
)
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn, SACLoss
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.distribution import (
    ScaledTanhNormal,
    ScaledSymlogNormal,
    FasterTransformedDistribution
)
from active_adaptation.learning.offpolicy.network import ConditionalBlock
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.dormancy import DormancyTracker

cs = ConfigStore.instance()


clip_grad_norm_ = nn.utils.clip_grad_norm_


def gaussian_target_entropy(act_dim: int, sigma: float) -> float:
    """Differential entropy of independent \\mathcal N(0, \\sigma^2) in \\mathbb R^d (FlashSAC-style).

    H = (d/2) * log(2 * pi * e * sigma^2). Used as SAC log-alpha target when ``target_entropy_sigma`` is set.
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
    # BC-style anchor on replay actions; curbs Q exploitation (:class:`SACLoss`).
    actor_behavior_coef: float = 0.0
    # sac specific
    entropy_bonus: float = 1.0
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common heuristic for tanh-squashed SAC).
    alpha_init: float = 4e-3
    target_entropy_sigma: float | None = 0.15
    soft_bound: float = math.pi

    tau_actor: float = 0.1 # a relatively large value for faster convergence
    tau_Q: float = 0.02  # a relatively large value for faster convergence
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0
    v_update_every: int = 32
    v_trace_steps: int = 32  # on-policy GAE horizon from replay ring (like blade_runner last())
    v_inner: int = 2
    gae_lambda: float = 0.95

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); GradScaler for critic, V head, standalone train_v, and actor (alpha stays fp32).
    use_amp: bool = True
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = False
    normalized_G_max: float = 5.0
    reward_norm_epsilon: float = 1e-8

    in_keys: Tuple[str, ...] = (OBS_KEY, ACTION_KEY)


cs.store(name="sac", node=SACConfig, group="algo")


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
        simba_mlp: bool = False,
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
            if simba_mlp:
                in_layer = nn.Linear(critic_input_dim, 512)
                in_layer.weight._non_muon = True
                out_layer = nn.Linear(512, num_atoms)
                out_layer.weight._non_muon = True
                return nn.Sequential(
                    in_layer,
                    SimbaMLP(512, 2, activation),
                    nn.LayerNorm(512),
                    out_layer,
                )
            else:
                out_layer = nn.Linear(512, num_atoms)
                out_layer.weight._non_muon = True
                return nn.Sequential(
                    nn.Linear(critic_input_dim, 512),
                    ConditionalBlock(hidden_dim=512),
                    ConditionalBlock(hidden_dim=512),
                    nn.RMSNorm(512),
                    out_layer,
                )

        self.critic_1 = make_critic()
        self.critic_2 = make_critic()

        self.register_buffer(
            "q_support",
            torch.linspace(v_min, v_max, num_atoms),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        z1 = self.critic_1(x)
        z2 = self.critic_2(x)
        return torch.cat([z1, z2], dim=-1)

    def get_values(
        self,
        obs: torch.Tensor,  # [B, obs_dim]
        act: torch.Tensor,  # [B, act_dim] or [B, K, act_dim] for multiple actions
    ) -> torch.Tensor:
        """Expected Q per twin head: shape ``[..., 2]`` (from logits / categorical support)."""
        if act.dim() == 2:
            return self.expected_values(self(obs, act))
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_exp = obs.unsqueeze(1).expand(b, k, obs.shape[-1]).reshape(b * k, obs.shape[-1])
            act_flat = act.reshape(b * k, act.shape[-1])
            logits = self.forward(obs_exp, act_flat)
            ev = self.expected_values(logits)
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
        return -(
            (target_dist * log_p1).sum(-1) + (target_dist * log_p2).sum(-1)
        ).mean()

    def expected_values(self, logits_pair: torch.Tensor) -> torch.Tensor:
        """Expected Q under softmax for each twin: logits_pair [B, 2 * num_atoms] -> [B, 2]."""
        log1, log2 = logits_pair.chunk(2, dim=-1)
        e1 = expected_q_from_logits(log1, self.q_support)
        e2 = expected_q_from_logits(log2, self.q_support)
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

        # self.trunk = MLP(
        #     [obs_dim, 256, 256, 256],
        #     nn.SiLU,
        #     layer_norm=layer_norm,
        #     first_non_muon=True,
        # )
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 384),
            ConditionalBlock(hidden_dim=384, condition_dim=0),
            ConditionalBlock(hidden_dim=384, condition_dim=0),
            nn.LayerNorm(384),
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

    def forward(self, obs: torch.Tensor):
        feat = self.trunk(obs)
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
                simba_mlp=False
            ).to(device)
            self.V = None  # unused; keeps optim / checkpoint layout stable
            self.V_quantile = 0.7
        else:
            self.Q = TwinQNetwork(obs_dim, act_dim, layer_norm=self.cfg.critic_layer_norm).to(device)
            self.V = nn.Sequential(
                MLP([obs_dim, 512, 512], nn.SiLU),
                nn.Linear(512, 1),
            ).to(device)
            self.V.apply(_init_sac_linear)
            self.V_quantile = 0.7

        self.gae = GAE(self.cfg.gamma, self.cfg.gae_lambda).to(device)
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
        
        if self.V is not None:
            self.opt_V = torch.optim.Adam(self.V.parameters(), lr=self.cfg.lr)

        self.global_step = 0

        if env is None:
            raise ValueError("SAC requires env for ReplayBuffer layout (fake_tensordict).")
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
            .detach()
            # .cpu()
        )
        fake_rb[REWARD_KEY] = fake_rb[REWARD_KEY].sum(-1, keepdim=True)
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        self.rb = ReplayBuffer(self.cfg.buffer_size, fake_rb)
        self.msr = (
            MultiStepReturn(self.cfg.gamma, self.cfg.n_steps).to(device)
            if self.cfg.n_steps > 1
            else None
        )
        self.sac_actor_loss = SACLoss(behavior_coef=self.cfg.actor_behavior_coef)

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

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        self.global_step += self.cfg.train_every

        td = tensordict.exclude(("next", "stats"), "collector")
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
        if len(bs) >= 2:
            for ti in range(int(bs[1])):
                sub = td[:, ti]
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update_reward_stats(
                        reward=sub[REWARD_KEY],
                        terminated=sub[TERM_KEY],
                        truncated=sub["next", "truncated"],
                    )
                self.rb.push(sub)
        else:
            if self.reward_normalizer is not None:
                self.reward_normalizer.update_reward_stats(
                    reward=td[REWARD_KEY],
                    terminated=td[TERM_KEY],
                    truncated=td["next", "truncated"],
                )
            self.rb.push(td)

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
                info = self.train_critic(
                    batch, diagnostics=(i == iters - 1)
                )
            infos.update(info)

            if self.enable_actor:
                for j in range(self.cfg.train_every):
                    info = self.train_actor(
                        diagnostics=(j == self.cfg.train_every - 1)
                    )
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
        else:
            assert self.msr is not None
            obs = batch[OBS_KEY][0]
            act = batch[ACTION_KEY][0]
            next_obs, reward, discount = self.msr(
                batch["next", OBS_KEY],
                batch[ACTION_KEY],
                reward,
                batch[TERM_KEY],
                batch[DONE_KEY],
            )

        obs = self.vecnorm_obs(obs)
        next_obs = self.vecnorm_obs(next_obs)

        with self._autocast():
            with torch.no_grad():
                # actions are sampled with uncorrelated noise
                loc, scale = self.actor_target(next_obs)
                dist = self.DistClass(loc, scale, upscale=self.actor.upscale)
                next_action = dist.sample()

                next_log_prob = dist.log_prob(next_action)
                target_action = next_action + torch.randn_like(next_action) * self.cfg.target_action_noise
                alpha = self.log_alpha.exp()
                lp = next_log_prob
                if lp.dim() == 1:
                    lp = lp.unsqueeze(-1)
                if lp.shape != reward.shape:
                    lp = lp.reshape_as(reward)

                if self.cfg.distributional:
                    assert isinstance(self.Q, TwinDistributionalQNetwork)
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
                    entropy_bonus = -alpha * lp
                    if entropy_bonus.shape != reward.shape:
                        entropy_bonus = entropy_bonus.reshape_as(reward)
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

            qs: torch.Tensor = self.Q(obs, act)
            q_loss = self.Q.compute_loss(qs, q_target)

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

        infos: dict = {"critic/q_loss": q_loss.item()}
        if diagnostics:
            with torch.no_grad():
                q_h = self.Q.get_values(obs.detach(), act.detach())
            q_val_mean = q_h.mean().item()
            q_val_max = q_h.max().item()
            q_val_std = q_h.std(dim=-1).mean().item()
            infos.update(
                {
                    "critic/q_value": q_val_mean,
                    "critic/q_max": q_val_max,
                    "critic/q_std": q_val_std,
                    "critic/grad_norm": critic_grad_norm.item(),
                }
            )

        # Optional: use expectile regression to estimate the value
        if self.V is not None:
            with self._autocast():
                v_pred = self.V(obs)
                q_pred = qs.detach().mean(dim=-1, keepdim=True)
                assert q_pred.shape == v_pred.shape
                v_err = q_pred - v_pred
                vf_sign = (v_err < 0).float()
                vf_weight = (1 - vf_sign) * self.V_quantile + vf_sign * (
                    1 - self.V_quantile
                )
                vf_loss = (vf_weight * (v_err**2)).mean()

            self.opt_V.zero_grad(set_to_none=True)
            if self._amp_enabled:
                self.grad_scaler.scale(vf_loss).backward()
                self.grad_scaler.step(self.opt_V)
                self.grad_scaler.update()
            else:
                vf_loss.backward()
                self.opt_V.step()

            if diagnostics:
                infos.update(
                    {
                        "critic/v_loss": vf_loss.item(),
                        "critic/v_value": v_pred.mean().item(),
                        "critic/v_err": v_err.mean().item(),
                    }
                )
        return infos

    def train_actor(self, diagnostics: bool = False):
        batch = self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1).to(
            self.device
        ) # [N,]

        obs = batch[OBS_KEY]
        obs = self.vecnorm_obs(obs)
        act = batch[ACTION_KEY]

        if self.cfg.sym_aug:
            obs_mirror = self.obs_transform(obs)
            act_mirror = self.act_transform(act)
            obs = torch.cat([obs, obs_mirror], dim=0)
            act = torch.cat([act, act_mirror], dim=0)

        with hold_out_net(self.Q), self._autocast():
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale, upscale=self.actor.upscale)
            action_update = dist.rsample((4,))  # [4, N, D]
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            q = self.Q.get_values(
                obs, einops.rearrange(action_update, "k n d -> n k d")
            ).mean(dim=-1)
            policy_term = -q.mean(dim=1)

        alpha = self.log_alpha.exp()
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term))
            + 0.01 * ((loc/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
        ).mean()

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

    def train_v(self):
        """On-policy-style V update: last `v_trace_steps` ring-buffer rows + GAE (ppo.common layout [N, T, …])."""
        if len(self.rb) <= self.cfg.v_trace_steps:
            return {}
        batch = self.rb.last(steps=self.cfg.v_trace_steps).to(self.device)

        reward = batch[REWARD_KEY]
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)

        # Ring buffer layout: [T, N, …]. GAE expects [N, T, …].
        obs_tn = batch[OBS_KEY]
        next_obs_tn = batch["next", OBS_KEY]
        T, N = obs_tn.shape[:2]
        flat = T * N

        obs_tn = self.vecnorm_obs(obs_tn)
        next_obs_tn = self.vecnorm_obs(next_obs_tn)
        with self._autocast():
            vals_tn = (
                self.V(obs_tn.reshape(flat, obs_tn.shape[-1])).reshape(T, N, 1)
            )
            next_vals_tn = (
                self.V(next_obs_tn.reshape(flat, next_obs_tn.shape[-1])).reshape(
                    T, N, 1
                )
            )

        r_nt = reward.transpose(0, 1)
        term_nt = batch[TERM_KEY].transpose(0, 1).float()
        done_nt = batch[DONE_KEY].transpose(0, 1).float()
        val_nt = vals_tn.transpose(0, 1).float()
        next_val_nt = next_vals_tn.transpose(0, 1).float()

        with torch.no_grad():
            _, ret = self.gae(r_nt, term_nt, done_nt, val_nt, next_val_nt)

        pred_nt = vals_tn.transpose(0, 1)
        with self._autocast():
            v_loss = F.mse_loss(pred_nt, ret)

        self.opt_V.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler.scale(v_loss).backward()
            self.grad_scaler.step(self.opt_V)
            self.grad_scaler.update()
        else:
            v_loss.backward()
            self.opt_V.step()

        return {
            "critic/v_loss": v_loss.item(),
            "critic/v_value": pred_nt.mean().item(),
        }

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["Q"] = self.Q.state_dict()
        # state_dict["V"] = self.V.state_dict()
        state_dict["actor"] = self.actor.state_dict()
        # do not store opt states as they make the ckpt very large
        # state_dict["opt_actor"] = self.opt_actor.state_dict()
        # state_dict["opt_Q"] = self.opt_Q.state_dict()
        # state_dict["opt_V"] = self.opt_V.state_dict()
        state_dict["opt_alpha"] = self.opt_alpha.state_dict()
        state_dict["log_alpha"] = self.log_alpha.detach()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.Q.load_state_dict(state_dict["Q"], strict=strict)
        # self.V.load_state_dict(state_dict["V"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        # reuse the same state dict for target networks
        self.Q_target.load_state_dict(state_dict["Q"], strict=strict)
        self.actor_target.load_state_dict(state_dict["actor"], strict=strict)
        # do not store opt states as they make the ckpt very large
        # self.opt_actor.load_state_dict(state_dict["opt_actor"])
        # self.opt_Q.load_state_dict(state_dict["opt_Q"])
        # self.opt_V.load_state_dict(state_dict["opt_V"])
        self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        self.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])

