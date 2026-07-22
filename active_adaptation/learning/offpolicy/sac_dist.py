from __future__ import annotations

import copy
import math
import einops
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.distributed as dist
import torch.nn as nn
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
from active_adaptation.learning.modules import VecNorm, IndependentNormal, CatTensors
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
from active_adaptation.learning.offpolicy.distributional import C51Critic
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import (
    unwrap_ddp,
    wrap_ddp,
)
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform

# Shared actor / critic trunks / rollout policy with scalar SAC.
from active_adaptation.learning.offpolicy.sac import (
    AlphaModule,
    CriticTrunk,
    NormalActor,
    SACRolloutPolicy,
    SimpleDoubleCritic,
    gaussian_target_entropy,
)

cs = ConfigStore.instance()


clip_grad_norm_ = nn.utils.clip_grad_norm_


@dataclass
class SACConfig:
    """Soft Actor-Critic with twin **distributional (C51)** critics (``train_offpolicy`` API)."""

    _target_: str = "active_adaptation.learning.offpolicy.sac_dist.SACConfig"
    name: str = "sac_dist"
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
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    sym_aug: bool = False
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # AR(1) pre-tanh exploration noise on rollout only: eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * N(0,I).
    use_correlated: bool = True
    # sac specific
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
    clamp_reward: bool = False
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8

    # path to prior data for RLPD
    prior_data: str | None = None
    prior_data_ratio: float = 0.4

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)

    def get_class(self):
        return SAC

cs.store(name="sac_dist", node=SACConfig, group="algo")


def TwinC51Critic(
    obs_dim: int,
    act_dim: int,
    num_atoms: int,
    v_min: float,
    v_max: float,
    activation: type[nn.Module] = nn.SiLU,
):
    module = SimpleDoubleCritic(
        fn=lambda: CriticTrunk(
            input_dim=obs_dim + act_dim,
            hidden_dim=512,
            output_dim=num_atoms,
            activation=activation,
        )
    )
    return C51Critic(
        module=module,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
    )


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

        if self.cfg.normalize_reward:
            # Std-normalized returns are O(1); fixed atom support (not task-tuned).
            v_min, v_max, num_atoms = -0.5, 5.0, 101
        else:
            v_min, v_max = -1.0, 9.0
            num_atoms = int((v_max - v_min) / 0.05) + 1
        self.Q = TwinC51Critic(
            obs_dim,
            self.act_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
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

        if self.cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(
                self.act_dim, self.cfg.target_entropy_sigma
            )
        else:
            self.target_entropy = - 0.5 * float(self.act_dim)
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
            obs_transform = env.observation_groups[OBS_KEY].symmetry_transform()
            act_transform = env.action_manager.symmetry_transform()
            if CMD_KEY in env.observation_spec.keys(True, True):
                cmd_transform = env.observation_groups[CMD_KEY].symmetry_transform()
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
            self.preproc,
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
            self.rb_prior.compute_return(gamma=self.cfg.gamma, reward_collate_fn=self.reward_collate_fn)
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
    
    def maybe_reset(self, next_tensordict: TensorDict):
        done = next_tensordict["done"]
        if done.any():
            pass
        return next_tensordict

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
                q_target = self.compute_target(next_obs, reward, discount)

            # as of torch 2.11, compiling loss computation leads to numerically
            # inconsistent results and degrades performance
            
            if self.cfg.sym_aug:
                # Q(s, a) = Q(s_mirror, a_mirror)
                obs_mirror = self.obs_transform(obs)
                act_mirror = self.act_transform(act)
                obs = torch.cat([obs, obs_mirror], dim=0)
                act = torch.cat([act, act_mirror], dim=0)
                q_target = torch.cat([q_target, q_target], dim=0)
                terminated = torch.cat([terminated, terminated], dim=0)
                is_init = torch.cat([is_init, is_init], dim=0)

            pred = self.Q(obs, act)
            per_sample_q_loss = self.Q.compute_loss(pred, q_target)
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

            logits = self.Q(obs, act)
            q = self.Q.expected_values(logits)
            q_lower = self.Q.expected_values(logits, risk_alpha=0.5)
            q_upper = self.Q.expected_values(logits, risk_alpha=-0.5)
            infos.update(self.Q.support_saturation_stats(logits))

            # Q is trained on normalized rewards when reward_normalizer is active;
            # map logs to PPO effective-horizon scale (``* S * (1 - gamma)``).
            if self.reward_normalizer is not None:
                q = self.reward_normalizer.denormalize_return_values(q)
                q_lower = self.reward_normalizer.denormalize_return_values(q_lower)
                q_upper = self.reward_normalizer.denormalize_return_values(q_upper)

            infos["critic/q_lower"] = q_lower.mean().item()
            infos["critic/q_upper"] = q_upper.mean().item()

            # online statistics
            q_val_mean = q[:B_online].mean().item()
            q_val_max = q[:B_online].max().item()
            q_val_std = q[:B_online].std(dim=-1).mean().item()

        infos["critic/q_value"] = q_val_mean
        infos["critic/q_max"] = q_val_max
        infos["critic/q_std"] = q_val_std
        
        if B_prior > 0:
            q_prior = q[B_online: B_eff]
            infos["critic/prior_q_value"] = q_prior.mean().item()
            infos["critic/prior_q_max"] = q_prior.max().item()
        
        if terminated.any():
            q_val_terminated = q[terminated.reshape(q.shape[0])]
            infos["critic/q_value_terminated"] = q_val_terminated.mean().item()
            infos["critic/q_loss_terminated"] = per_sample_q_loss[terminated.reshape(q.shape[0])].mean().item()

        return infos

    @torch.no_grad()
    def _compute_target(self, next_obs: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        # actions are sampled with uncorrelated noise
        loc, scale = self.actor_target(next_obs)
        dist = self.DistClass(loc, scale)
        next_action = dist.sample()

        next_log_prob = dist.log_prob(next_action)
        alpha = self.alpha()
        lp = next_log_prob.reshape_as(reward)

        entropy_bonus = (-alpha * lp).reshape_as(reward) * self.entropy_scale
        adjusted_reward = reward + discount * self.cfg.entropy_bonus * entropy_bonus
        noise = torch.randn_like(next_action).clamp(-3.0, 3.0)
        q_target = self.Q_target.compute_target(
            next_obs,
            next_action + noise * self.cfg.target_action_noise,
            adjusted_reward,
            discount,
        )
        return q_target

    @ScopedTimer("train_actor")
    def train_actor(self, diagnostics: bool = False):
        # actor training does not need next observations
        batch = (
            self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1, next_obs=False)
            .to(self.device)
            .select(*self.train_keys, strict=False) # [N,]
        )
        if self.rb_prior is not None:
            batch_prior = self.rb_prior.sample(
                batch_size=int(self.cfg.actor_batch_size * self.cfg.prior_data_ratio),
                steps=1,
                next_obs=False,
                term_obs=True,
            ).to(self.device)
            G = batch_prior["G"]
            steps_to_go = batch_prior["steps_to_go"]
            term_obs = batch_prior["term_obs"]
            batch_prior = batch_prior.select(*self.train_keys, strict=False)
            batch = torch.cat([batch, batch_prior], dim=0)

        self.preproc(batch)
        obs = batch["_input_normed"]
        act = batch[ACTION_KEY]
        is_init = batch["is_init"]
        prior_obs = None
        prior_count = 0
        if self.rb_prior is not None:
            prior_count = batch_prior.shape[0]
            prior_obs = obs[-prior_count:]

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
            q = self.Q.get_values(
                obs,
                einops.rearrange(action_update, "k n d -> n k d"),
            ).mean(dim=-1)
            policy_term = -q.mean(dim=1)

        alpha = self.alpha()
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term) * self.entropy_scale)
            + 0.01 * ((loc/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
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
            q_for_log = q
            if self.reward_normalizer is not None:
                q_for_log = self.reward_normalizer.denormalize_return_values(q_for_log)
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
                # Online π vs prior MC return-to-go (no entropy): positive ⇒ π beats demo.
                # baseline = G + γ^H Q(s_term, a~π) with G/Q in the critic's reward scale.
                assert prior_obs is not None
                loc_p, scale_p = self.actor(prior_obs)
                dist_p = self.DistClass(loc_p, scale_p)
                a_p = dist_p.rsample((4,))
                online_q = self.Q.get_values(
                    prior_obs,
                    einops.rearrange(a_p, "k n d -> n k d"),
                ).mean(dim=-1).mean(dim=1)

                term_td = term_obs.clone()
                self.preproc(term_td)
                term_in = term_td["_input_normed"]
                loc_t, scale_t = self.actor(term_in)
                a_t = self.DistClass(loc_t, scale_t).rsample()
                q_term = self.Q.get_values(term_in, a_t).mean(dim=-1)

                if self.reward_normalizer is not None:
                    G_scaled = self.reward_normalizer.normalize_rewards(G)
                else:
                    G_scaled = G * (1.0 - self.cfg.gamma)
                H = steps_to_go.to(dtype=online_q.dtype).squeeze(-1)
                baseline = G_scaled.squeeze(-1) + torch.pow(float(self.cfg.gamma), H) * q_term
                advantage = online_q - baseline
                infos["actor/online_advantage"] = advantage.mean().item()

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

