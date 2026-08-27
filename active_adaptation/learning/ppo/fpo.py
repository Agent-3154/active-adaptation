"""Flow Policy Optimization (FPO++) with CFM-loss ratio updates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union
import warnings

import torch
import torch.distributed as distr
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase
from tensordict.nn import TensorDictSequential as Seq
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import Composite, TensorSpec

import active_adaptation as aa
from active_adaptation.learning.modules import CatTensors, MLP, VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    Critic,
    GAE,
    make_batch,
)
from active_adaptation.learning.utils.distributed import check_parameters
from active_adaptation.learning.utils.dormancy import DormancyTracker
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform
from tensordict.nn.probabilistic import interaction_type, InteractionType

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


def clamp_ste(
    x: torch.Tensor, min_value: float | None = None, max_value: float | None = None
) -> torch.Tensor:
    """Clamp in forward pass and preserve identity gradient."""
    clamped = x.clamp(min=min_value, max=max_value)
    return x + (clamped - x).detach()


def fpo_surrogate_loss(
    ratio: torch.Tensor,
    adv: torch.Tensor,
    clip_param: float,
    mode: str,
) -> torch.Tensor:
    """Loss to minimize for PPO/SPO/ASPO trust-region objectives."""
    if adv.shape != ratio.shape:
        adv = adv.expand_as(ratio)

    if mode == "ppo":
        surr = adv * ratio
        surr_clipped = adv * ratio.clamp(1.0 - clip_param, 1.0 + clip_param)
        return -torch.min(surr, surr_clipped).mean()

    spo_obj = ratio * adv - adv.abs() / (2.0 * clip_param) * (ratio - 1.0).square()
    if mode == "spo":
        return -spo_obj.mean()
    if mode == "aspo":
        surr = adv * ratio
        surr_clipped = adv * ratio.clamp(1.0 - clip_param, 1.0 + clip_param)
        ppo_obj = torch.min(surr, surr_clipped)
        mixed_obj = torch.where(adv >= 0.0, ppo_obj, spo_obj)
        return -mixed_obj.mean()
    raise ValueError(f"Unknown trust_region_mode: {mode}")


class FlowMatchingActor(nn.Module):
    """Flow-matching actor that returns actions and rollout CFM bookkeeping."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation: type[nn.Module],
        hidden_dims: Tuple[int, ...],
        *,
        timestep_embed_dim: int,
        sampling_steps: int,
        n_samples_per_action: int,
        action_perturb_std: float,
        cfm_loss_t_inverse_cdf_beta: float,
        cfm_loss_reduction: str,
    ):
        super().__init__()
        if timestep_embed_dim % 2 != 0:
            raise ValueError("timestep_embed_dim must be even")
        self.action_dim = action_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.sampling_steps = sampling_steps
        self.n_samples_per_action = n_samples_per_action
        self.action_perturb_std = action_perturb_std
        self.cfm_loss_t_inverse_cdf_beta = cfm_loss_t_inverse_cdf_beta
        if cfm_loss_reduction not in ("mean", "sum", "sqrt"):
            raise ValueError(
                f"Invalid cfm_loss_reduction={cfm_loss_reduction}; expected one of "
                "('mean', 'sum', 'sqrt')"
            )
        self.cfm_loss_reduction = cfm_loss_reduction

        self.backbone = MLP(
            num_units=[obs_dim + timestep_embed_dim + action_dim, *hidden_dims],
            activation=activation,
            first_non_muon=True,
        )
        self.velocity_head = nn.LazyLinear(action_dim)
        self.velocity_head.weight._non_muon = True
        # self.integrate_flow = torch.compile(self.integrate_flow, mode="reduce-overhead")
    
    def forward(
        self,
        obs: torch.Tensor, # [*, obs_dim]
        x_t: torch.Tensor, # [*, action_dim]
        t: torch.Tensor, # [*, 1]
    ):
        freqs = 2 ** torch.arange(self.timestep_embed_dim // 2, device=obs.device)
        scaled_t = t * freqs 
        embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
        inp = torch.cat([obs, embedded_t, x_t], dim=-1)
        features = self.backbone(inp)
        velocity = self.velocity_head(features)
        return velocity
    
    def act(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = obs.shape[0]
        device = obs.device
        if interaction_type() in (InteractionType.MODE, InteractionType.DETERMINISTIC):
            x_0 = torch.zeros(N, self.action_dim, device=obs.device)
            perturb = torch.zeros_like(x_0)
        else:
            x_0 = torch.randn(N, self.action_dim, device=obs.device)
            perturb = torch.randn_like(x_0) * self.action_perturb_std
        x_1 = self.integrate_flow(obs, x_0)

        eps = torch.randn(N, self.n_samples_per_action, self.action_dim, device=device)
        t = torch.rand(N, self.n_samples_per_action, 1, device=device)
        beta = self.cfm_loss_t_inverse_cdf_beta
        # Scale to [0.005, 0.995] to avoid boundary instabilities at t=0 and t=1
        t = 0.005 + 0.995 * (1.0 - (1.0 - t) ** (1.0 / beta))
        cfm_loss = self.compute_cfm_loss(obs, x_1, eps, t)

        action = x_1 + perturb
        # cfm_loss has shape [N, self.n_samples_per_action]
        # eps has shape [N, self.n_samples_per_action, self.action_dim]
        # t has shape [N, self.n_samples_per_action, 1]
        return action, cfm_loss, eps, t
    
    def integrate_flow(
        self,
        obs: torch.Tensor, # [N, obs_dim]
        x_0: torch.Tensor, # [N, action_dim]
    ) -> torch.Tensor: # [N, action_dim]
        """Following common flow matching literature, x_0 is noise and x_1 is the target.
        We integrate the flow from x_0 to x_1 using the velocity field.
        Note that this is different from the original FPO++ paper. TODO: is this correct?
        """
        dt = 1.0 / self.sampling_steps
        x_t = x_0
        for i in range(self.sampling_steps):
            t = torch.full((obs.shape[0], 1), i * dt, device=obs.device)
            velocity = self(obs, x_t, t)
            x_t = x_t + velocity * dt
        return x_t
    
    def compute_cfm_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor: # [B, n_samples_per_action]
        """Following common flow matching literature, x_0 is noise and x_1 is the target.
        We integrate the flow from x_0 to x_1 using the velocity field.
        Note that this is different from the original FPO++ paper. TODO: is this correct?
        """
        N = act.shape[0]
        obs = obs.reshape(N, 1, -1).expand(N, self.n_samples_per_action, -1)
        x_t = t * act.reshape(N, 1, self.action_dim) + (1 - t) * eps
        velocity = self(obs, x_t, t)
        target_velocity = act.reshape(N, 1, self.action_dim) - eps
        
        if self.cfm_loss_reduction == "mean":
            loss = torch.mean((velocity - target_velocity) ** 2, dim=-1)
        elif self.cfm_loss_reduction == "sum":
            loss = torch.sum((velocity - target_velocity) ** 2, dim=-1)
        else:  # "sqrt"
            squared_errors = (velocity - target_velocity) ** 2
            denom = squared_errors.shape[-1] ** 0.5
            loss = torch.sum(squared_errors, dim=-1) / denom
        return loss


@dataclass
class FPOConfig:
    _target_: str = "active_adaptation.learning.ppo.fpo.FPOConfig"
    name: str = "fpo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    desired_kl: Union[float, None] = 1e-4
    clip_param: float = 0.2
    value_loss_coef: float = 1.0

    clamp_reward: bool = False
    activation: str = "Mish"
    muon: bool = False
    symaug: bool = False

    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 256)
    timestep_embed_dim: int = 8
    sampling_steps: int = 64
    n_samples_per_action: int = 16
    action_perturb_std: float = 0.02
    cfm_loss_t_inverse_cdf_beta: float = 1.0
    cfm_loss_reduction: str = "sqrt" # "mean", "sum", "sqrt"
    trust_region_mode: str = "aspo" # "ppo", "spo", "aspo"
    normalize_advantage: bool = True
    normalize_advantage_per_minibatch: bool = False
    advantage_clamp: Tuple[float, float] = (100.0, 100.0)
    cfm_diff_clamp_max: float = 10.0
    cfm_loss_clamp: float = 20.0
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0

    compile: bool = False
    use_ddp: bool = True
    debug: bool = False
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)

    def get_class(self):
        return FPOPolicy


cs = ConfigStore.instance()
cs.store("fpo", node=FPOConfig, group="algo")


def vecnorm_sync_(module: nn.Module):
    if isinstance(module, VecNorm):
        module.synchronize(mode="broadcast")


class FPOPolicy(TensorDictModuleBase):
    """Flow Policy Optimization with CFM-loss ratio and PPO-style value updates."""

    def __init__(
        self,
        cfg: FPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        cmd_transform: Optional[SymmetryTransform] = None,
        obs_transform: Optional[SymmetryTransform] = None,
        act_transform: Optional[SymmetryTransform] = None,
    ):
        del reward_spec
        super().__init__()
        self.cfg = cfg
        if self.cfg.debug and self.cfg.compile:
            raise ValueError("Debug mode and compile mode cannot be enabled together")
        if self.cfg.trust_region_mode not in ("ppo", "spo", "aspo"):
            raise ValueError(
                f"Invalid trust_region_mode={self.cfg.trust_region_mode}; expected one of "
                "('ppo', 'spo', 'aspo')"
            )
        self.device = device
        self.max_grad_norm = 1.0
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.desired_kl = self.cfg.desired_kl

        fake_input = observation_spec.zero().to(self.device)
        self.cmd_transform = (
            cmd_transform.to(self.device) if cmd_transform is not None else None
        )
        self.obs_transform = (
            obs_transform.to(self.device) if obs_transform is not None else None
        )
        self.act_transform = (
            act_transform.to(self.device) if act_transform is not None else None
        )

        self.training_keys = [
            "cfm_loss",
            "cfm_loss_eps",
            "cfm_loss_t",
            "adv",
            "ret",
            "is_init",
        ]
        if CMD_KEY in observation_spec.keys(True, True):
            self.training_keys += [CMD_KEY, OBS_KEY, ACTION_KEY]
            inp_dim = fake_input[CMD_KEY].shape[-1] + fake_input[OBS_KEY].shape[-1]
            self.vecnorm = Seq(
                CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False),
                Mod(VecNorm((inp_dim,)), ["_input"], ["_obs_normed"]),
            ).to(self.device)
        else:
            self.training_keys += [OBS_KEY, ACTION_KEY]
            inp_dim = fake_input[OBS_KEY].shape[-1]
            self.vecnorm = Mod(VecNorm((inp_dim,)), [OBS_KEY], ["_obs_normed"]).to(
                self.device
            )

        self.action_dim = action_spec.shape[-1]
        activation = getattr(nn, self.cfg.activation)
        
        self.flow_actor_impl = FlowMatchingActor(
            obs_dim=inp_dim,
            action_dim=self.action_dim,
            activation=activation,
            hidden_dims=self.cfg.actor_hidden_dims,
            timestep_embed_dim=self.cfg.timestep_embed_dim,
            sampling_steps=self.cfg.sampling_steps,
            n_samples_per_action=self.cfg.n_samples_per_action,
            action_perturb_std=self.cfg.action_perturb_std,
            cfm_loss_t_inverse_cdf_beta=self.cfg.cfm_loss_t_inverse_cdf_beta,
            cfm_loss_reduction=self.cfg.cfm_loss_reduction,
        ).to(self.device)

        self.actor_rollout = Mod(
            self.flow_actor_impl.act,
            ["_obs_normed"],
            [ACTION_KEY, "cfm_loss", "cfm_loss_eps", "cfm_loss_t"],
        )

        critic_mlp = MLP(
            num_units=[inp_dim, *self.cfg.critic_hidden_dims],
            activation=activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            Mod(critic_mlp, ["_obs_normed"], ["_critic_feature"]),
            Mod(Critic(1), ["_critic_feature"], ["state_value"]),
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor_rollout(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)

        self.flow_actor_impl.apply(init_)
        self.critic.apply(init_)
        self._rollout_dormancy_tracker: Union[DormancyTracker, None] = None

    @classmethod
    def from_env(cls, cfg: FPOConfig, env: _EnvBase, device: str):
        observation_spec = env.observation_spec
        action_spec = env.action_spec
        reward_spec = env.reward_spec
        if CMD_KEY in observation_spec.keys(True, True):
            cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transform()
        else:
            cmd_transform = None
        obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform()
        act_transform = env.action_manager.symmetry_transform()
        return cls(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            device=device,
            cmd_transform=cmd_transform,
            obs_transform=obs_transform,
            act_transform=act_transform,
        )

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict, device: str):
        del state_dict, device
        raise NotImplementedError("from_state_dict is not implemented yet")

    def on_stage_start(self, stage: str, env: _EnvBase):
        del env
        if stage not in ("train", ""):
            return
        if aa.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor, device_ids=[aa.get_local_rank()])
                self.critic = DDP(self.critic, device_ids=[aa.get_local_rank()])
            else:
                for param in self.flow_actor_impl.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp
        self.world_size = aa.get_world_size()

        if self.cfg.muon:
            self.opt = MuonAdamWWrapper(
                [self.flow_actor_impl, self.critic],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.flow_actor_impl.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )

        self.update = self._update
        if self.cfg.compile and not aa.is_distributed():
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if self._rollout_dormancy_tracker is not None:
            self._rollout_dormancy_tracker.close()
            self._rollout_dormancy_tracker = None
        modules = [self.vecnorm, self.actor_rollout]
        if critic:
            modules.append(self.critic)
        policy = Seq(*modules)

        if self.cfg.compile:
            policy = torch.compile(policy)
        if self.cfg.debug:
            tracker = DormancyTracker(policy)
            policy.forward = tracker.wrap(policy.forward)
            self._rollout_dormancy_tracker = tracker
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats").to(self.device, non_blocking=True)
        infos = []

        self.vecnorm.to(self.device, non_blocking=True)
        self.flow_actor_impl.to(self.device)
        self.critic.to(self.device)

        with ScopedTimer("compute_advantage"):
            self.compute_advantage(
                tensordict, self.critic, "adv", "ret", self.cfg.clamp_reward
            )
            adv_unnormalized = tensordict["adv"]
            adv_mean = adv_unnormalized.mean()
            adv_std = adv_unnormalized.std()
            adv = tensordict["adv"]
            adv = (adv - adv_mean) / adv_std.clamp_min(1e-7)
            tensordict["adv"] = adv
            pos_clamp, neg_clamp = self.cfg.advantage_clamp
            tensordict["adv"] = tensordict["adv"].clamp(-neg_clamp, pos_clamp)

        td = tensordict.select(*self.training_keys)
        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(td, self.cfg.num_minibatches):
                if self.cfg.symaug:
                    minibatch = self._augment_symmetry(minibatch)
                infos.append(self.update(minibatch))

                # if self.desired_kl is not None:
                #     kl = infos[-1]["actor/kl_x1"]
                #     actor_lr = self.opt.param_groups[0]["lr"]
                #     if kl > self.desired_kl * 2.0:
                #         actor_lr = max(1e-5, actor_lr / 1.5)
                #     elif kl < self.desired_kl / 2.0 and kl > 0.0:
                #         actor_lr = min(1e-2, actor_lr * 1.5)
                #     self.opt.param_groups[0]["lr"] = actor_lr

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        # Same signal as unclipped surrogate mass: larger is usually "more policy improvement".
        infos["actor/policy_gain"] = infos["actor/weighted_ratio"]
        # Return-distribution diagnostics for value-target scale drift across training.
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/value_max"] = tensordict["ret"].max().item()
        # Raw (pre-normalization) advantage spread helps detect noisy reward/critic regimes.
        infos["critic/adv_std"] = adv_std.item()
        infos["critic/adv_mean"] = adv_mean.item()
        infos["critic/adv_pos"] = (adv >= 0.0).float().mean().item()
        reward_aggregated = tensordict["next", "reward_aggregated"]
        # Fraction of non-positive rewards; useful for sparse/penalty-heavy tasks.
        infos["critic/neg_rew_ratio"] = (reward_aggregated <= 0.0).float().mean().item()

        if self.cfg.debug and self._rollout_dormancy_tracker is not None:
            dormancy = self._rollout_dormancy_tracker.compute_dormancy()
            for module_name, value in dormancy.items():
                infos[f"dormancy/{module_name}"] = value
            self._rollout_dormancy_tracker.reset()

        if aa.is_distributed():
            self.vecnorm.apply(vecnorm_sync_)
            if self.cfg.debug:
                infos["actor/diff"] = check_parameters(self.flow_actor_impl)
                infos["critic/diff"] = check_parameters(self.critic)
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def compute_value(self, tensordict: TensorDict):
        return self.critic(tensordict)

    @torch.no_grad()
    def compute_advantage(
        self,
        tensordict: TensorDict,
        critic: Mod,
        adv_key: str = "adv",
        ret_key: str = "ret",
        clamp_reward: bool = True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as td_flat:
                critic(self.vecnorm(td_flat))
                critic(self.vecnorm(td_flat["next"]))

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

        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)
        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _augment_symmetry(self, tensordict: TensorDict) -> TensorDict:
        if self.obs_transform is None or self.act_transform is None:
            return tensordict
        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry["cfm_loss_eps"] = self.act_transform(tensordict["cfm_loss_eps"])
        symmetry["cfm_loss_t"] = tensordict["cfm_loss_t"]
        symmetry["cfm_loss"] = tensordict["cfm_loss"]
        if self.cmd_transform is not None and CMD_KEY in tensordict.keys(True, True):
            symmetry[CMD_KEY] = self.cmd_transform(tensordict[CMD_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        return torch.cat([tensordict, symmetry])

    @ScopedTimer("fpo_update")
    def _update(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
        valid = (~tensordict["is_init"]).float()
        valid_cnt = valid.sum().clamp_min(1.0)

        action_data = tensordict[ACTION_KEY]
        old_cfm_loss = tensordict["cfm_loss"] # [B, n_samples_per_action]
        old_cfm_eps = tensordict["cfm_loss_eps"] # [B, n_samples_per_action, action_dim]
        old_cfm_t = tensordict["cfm_loss_t"] # [B, n_samples_per_action, 1]

        cfm_loss = self.flow_actor_impl.compute_cfm_loss(
            tensordict["_obs_normed"],
            action_data,
            old_cfm_eps,
            old_cfm_t,
        ) # [B, n_samples_per_action]

        if self.cfg.cfm_loss_clamp > 0.0:
            old_cfm_loss = old_cfm_loss.clamp(max=self.cfg.cfm_loss_clamp)
            cfm_loss = cfm_loss.clamp(max=self.cfg.cfm_loss_clamp)

        adv = tensordict["adv"]
        if self.cfg.normalize_advantage_per_minibatch and self.cfg.normalize_advantage:
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-7)
        pos_clamp, neg_clamp = self.cfg.advantage_clamp
        adv = adv.clamp(-neg_clamp, pos_clamp)

        if self.cfg.cfm_loss_clamp_negative_advantages:
            negative_adv = adv < 0.0
            cfm_loss = torch.where(
                negative_adv.expand_as(cfm_loss),
                cfm_loss.clamp(max=self.cfg.cfm_loss_clamp_negative_advantages_max),
                cfm_loss,
            )

        # old_cfm_loss, cfm_loss, log_ratio, ratio: [B, n_samples_per_action]
        assert old_cfm_loss.shape == cfm_loss.shape
        log_ratio = old_cfm_loss - cfm_loss
        log_ratio = clamp_ste(log_ratio, max_value=self.cfg.cfm_diff_clamp_max)
        ratio = torch.exp(log_ratio)

        policy_loss = fpo_surrogate_loss(
            ratio=ratio,
            adv=adv,
            clip_param=self.cfg.clip_param,
            mode=self.cfg.trust_region_mode,
        )

        ret = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(ret, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt
        loss = policy_loss + self.cfg.value_loss_coef * value_loss

        self.opt.zero_grad()
        loss.backward()

        if self.should_reduce_grads:
            for param in self.flow_actor_impl.parameters():
                if param.grad is not None:
                    distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                    param.grad /= self.world_size
            for param in self.critic.parameters():
                if param.grad is not None:
                    distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                    param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.flow_actor_impl.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.opt.step()

        with torch.no_grad():
            explained_var = 1.0 - F.mse_loss(values, ret) / ret.var().clamp_min(1e-7)
            # PPO-style trust-region saturation rate.
            clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
            # First-order KL proxy based on r-1-log r.
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            # Importance-weighted advantage mass under CFM-derived ratio.
            weighted_ratio = (ratio * adv.expand_as(ratio)).mean()
            is_positive_adv = (adv >= 0.0).expand_as(cfm_loss)
            cfm_loss_pos = cfm_loss[is_positive_adv].mean()
            cfm_loss_neg = cfm_loss[~is_positive_adv].mean()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/weighted_ratio": weighted_ratio.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/approx_kl": approx_kl,
            "actor/cfm_loss": cfm_loss.mean().detach(),
            "actor/cfm_loss_pos": cfm_loss_pos.detach(),
            "actor/cfm_loss_neg": cfm_loss_neg.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()

        if self.cmd_transform is not None:
            state_dict["cmd_transform"] = self.cmd_transform.state_dict()
        if self.obs_transform is not None:
            state_dict["obs_transform"] = self.obs_transform.state_dict()
        if self.act_transform is not None:
            state_dict["act_transform"] = self.act_transform.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            module_state = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(module_state, strict=strict)
                succeed_keys.append(name)
            except Exception as exc:
                warnings.warn(f"Failed to load state dict for {name}: {str(exc)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys
