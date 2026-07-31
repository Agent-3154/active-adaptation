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
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import warnings
import torch.utils._pytree as pytree

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import hold_out_net

from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Any, List, TYPE_CHECKING
from collections import OrderedDict
import numpy as np

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

from active_adaptation.learning.modules import (
    VecNorm,
    IndependentNormal,
    MLP,
    CatTensors,
)
from active_adaptation.learning.ppo.common import (
    ppo_clipped_loss,
    spo_loss,
    resolve_clip_param,
    CMD_KEY,
    OBS_KEY,
    ACTION_KEY,
    REWARD_KEY,
    TERM_KEY,
    DONE_KEY,
    GAE,
    make_batch,
    Actor,
    Critic,
)
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import check_parameters, unwrap_ddp
from active_adaptation.learning.utils.dormancy import DormancyTracker
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform

import active_adaptation as aa
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_symaug.PPOConfig"
    name: str = "ppo_symaug"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    desired_kl: Union[float, None] = None
    # Scalar ε → [1-ε, 1+ε]. Length-2 list/tuple [eps_neg, eps_pos] →
    # [1-eps_neg, 1+eps_pos]. Typed as Any: Hydra/OmegaConf cannot express
    # Union[float, Sequence[float]], and YAML lists become ListConfig.
    clip_param: Any = (0.2, 0.2)
    entropy_coef: float = 0.002
    pred_std: bool = False

    clamp_reward: bool = False
    
    actor_num_units: Tuple[int, ...] = (256, 256, 256)
    critic_num_units: Tuple[int, ...] = (512, 256, 256)
    activation: str = "Mish"
    spo: bool = False # use Simple Policy Optimization Loss
    muon: bool = False # use Muon optimizer
    aux_coef: float = 0.0 # loss coefficient for auxiliary prediction loss
    
    compile: bool = False
    use_ddp: bool = True
    debug: bool = False # enable correctness checkers

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY,) # CMD_KEY is optional. One can embed the command into the observation.

    def get_class(self):
        return PPOPolicy


cs = ConfigStore.instance()
cs.store("ppo_symaug", node=PPOConfig, group="algo")
cs.store("ppo_symaug_large", node=PPOConfig(actor_num_units=(512, 512, 512), critic_num_units=(512, 512, 512)), group="algo")


def vecnorm_sync_(module: nn.Module):
    if isinstance(module, VecNorm):
        module.synchronize(mode="broadcast")


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: Composite, 
        action_spec: Composite, 
        reward_spec: TensorSpec,
        device,
        *,
        cmd_transform: Optional[SymmetryTransform] = None,
        obs_transform: Optional[SymmetryTransform] = None,
        act_transform: Optional[SymmetryTransform] = None,
    ):
        super().__init__()
        self.cfg = cfg
        if self.cfg.debug and self.cfg.compile:
            raise ValueError("Debug mode and compile mode cannot be enabled together")
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = resolve_clip_param(self.cfg.clip_param)
        self.actor_loss_fn = spo_loss if self.cfg.spo else ppo_clipped_loss
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)  

        fake_input = observation_spec.zero().to(self.device)
        self.cmd_transform = cmd_transform.to(self.device) if cmd_transform is not None else None
        self.obs_transform = obs_transform.to(self.device)
        self.act_transform = act_transform.to(self.device)
        
        # the keys needed for `_update`
        self.training_keys = ["action_log_prob", "adv", "ret", "is_init"]
        if CMD_KEY in observation_spec.keys(True, True):
            self.training_keys += [CMD_KEY, OBS_KEY, ACTION_KEY]
            inp_dim = fake_input[CMD_KEY].shape[-1] + fake_input[OBS_KEY].shape[-1]
            self.vecnorm = Seq(
                CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False),
                Mod(VecNorm((inp_dim,), decay=1.0), ["_input"], ["_obs_normed"]),
            ).to(self.device)
        else:
            self.training_keys += [OBS_KEY, ACTION_KEY]
            inp_dim = fake_input[OBS_KEY].shape[-1]
            self.vecnorm = Mod(VecNorm((inp_dim,), decay=1.0), [OBS_KEY], ["_obs_normed"]).to(self.device)
        self.action_dim = action_spec.shape[-1]

        Activation = getattr(nn, self.cfg.activation)
        actor_mlp = MLP(
            num_units=[inp_dim, *self.cfg.actor_num_units],
            activation=Activation,
            first_non_muon=True,
        )
        actor_modules = [
            Mod(actor_mlp, ["_obs_normed"], ["_actor_feature"]),
            Mod(Actor(self.action_dim, predict_std=self.cfg.pred_std), ["_actor_feature"], ["loc", "scale"])
        ]
        if self.cfg.aux_coef > 0.0:
            actor_modules.append(Mod(nn.LazyLinear(1), ["_actor_feature"], ["aux_pred"]))
        
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=Seq(*actor_modules),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        critic_mlp = MLP(
            num_units=[inp_dim, *self.cfg.critic_num_units],
            activation=Activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            Mod(critic_mlp, ["_obs_normed"], ["_critic_feature"]),
            Mod(Critic(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.)
            elif isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        self._rollout_dormancy_tracker: Union[DormancyTracker, None] = None
        self.obs_func_keys: List[str]
        self.obs_split: List[int]
        self._obs_importance_interval: int = 8
        self._obs_importance_step: int = 0
        # Running EMA of return std for scale-invariant value loss normalization.
        # Initialized conservatively at 1.0 so early training uses a larger (not smaller)
        # value gradient — the EMA ramps up to the true scale within the first few rollouts.
        self.ret_std_ema: float = 1.0

    @classmethod
    def from_env(cls, cfg: PPOConfig, env: _EnvBase, device: str):
        observation_spec = env.observation_spec
        action_spec = env.action_spec
        reward_spec = env.reward_spec
        obs_func_keys = list(env.observation_groups[OBS_KEY].keys())
        obs_split = env.observation_groups[OBS_KEY].split
        # CMD_KEY precedes OBS_KEY in the observation
        if CMD_KEY in observation_spec.keys(True, True):
            cmd_transform = env.observation_groups[CMD_KEY].symmetry_transform()
            obs_func_keys = list(env.observation_groups[CMD_KEY].keys()) + obs_func_keys
            obs_split = env.observation_groups[CMD_KEY].split + obs_split
        else:
            cmd_transform = None
        obs_transform = env.observation_groups[OBS_KEY].symmetry_transform()
        act_transform = env.action_manager.symmetry_transform()
        policy = cls(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            device=device,
            cmd_transform=cmd_transform,
            obs_transform=obs_transform,
            act_transform=act_transform,
        )
        policy.obs_func_keys = obs_func_keys
        policy.obs_split = obs_split
        return policy

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict, device: str):
        pass
        return cls(...)

    def on_stage_start(self, stage: str, env: _EnvBase):
        if not stage in ("train", ""):
            return
        if aa.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor, device_ids=[aa.get_local_rank()])
                self.critic = DDP(self.critic, device_ids=[aa.get_local_rank()])
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp
        self.world_size = aa.get_world_size()

        if self.cfg.muon:
            self.opt = MuonAdamWWrapper(
                [self.actor, self.critic],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01
            )

        self.update = self._update
        if self.cfg.compile and not aa.is_distributed():
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str="train", critic: bool = False):
        if self._rollout_dormancy_tracker is not None:
            self._rollout_dormancy_tracker.close()
            self._rollout_dormancy_tracker = None
        # VecNorm is frozen in eval mode to avoid unexpected updates
        vecnorm = self.vecnorm if mode == "train" else VecNorm.freeze()(self.vecnorm)
        if critic:
            policy = Seq(vecnorm, self.actor, self.critic)
        else:
            policy = Seq(vecnorm, self.actor)
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
        self.actor.to(self.device)
        self.critic.to(self.device)

        with ScopedTimer("compute_advantage"):
            self.compute_advantage(tensordict, self.critic, "adv", "ret", self.cfg.clamp_reward)
            action = tensordict[ACTION_KEY]
            adv_unnormalized = tensordict["adv"]
            log_probs_before = tensordict["action_log_prob"]
            adv = tensordict["adv"]
            adv_mean = adv.mean()
            adv_std = adv.std()
            adv = (adv - adv_mean) / adv_std.clamp_min(1e-7)
            tensordict["adv"] = adv
        
        # Update EMA of return std from the full rollout before the PPO epoch loop,
        # so every minibatch update for this rollout uses the same normalization factor.
        # In distributed training, synchronize the local ret_std across ranks first so
        # all ranks update their EMA with the same global value — matching the scale of
        # gradients that DDP will average.
        ret_std_t = tensordict["ret"].std()
        if aa.is_distributed():
            distr.all_reduce(ret_std_t, op=distr.ReduceOp.SUM)
            ret_std_t = ret_std_t / aa.get_world_size()
        m = 0.99
        self.ret_std_ema = m * self.ret_std_ema + (1.0 - m) * ret_std_t.item()

        td = tensordict.select(*self.training_keys)
        for epoch in range(self.cfg.ppo_epochs):
            compute_diagnostics = epoch == self.cfg.ppo_epochs - 1
            batch = make_batch(td, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch = self._augment_symmetry(minibatch)
                info = self.update(minibatch, compute_diagnostics)
                if compute_diagnostics:
                    infos.append(info)
                
                if self.desired_kl is not None: # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr
        
        with torch.no_grad():
            tensordict_ = self.actor(tensordict.copy())
            dist = IndependentNormal(tensordict_["loc"], tensordict_["scale"])
            log_probs_after = dist.log_prob(action)
            log_ratio = (log_probs_after - log_probs_before).reshape_as(adv_unnormalized)
            # log π_new/π_old · A: first-order signal of whether the post-update policy
            # shifts log-prob in the direction favored by the (unnormalized) advantage.
            policy_gain = log_ratio * adv_unnormalized
            # r(θ) · A with r = exp(log_ratio) = π_new/π_old; same weighted term as in
            # the unclipped PPO surrogate, useful to monitor IS-weighted advantage mass.
            weighted_ratio = log_ratio.exp() * adv_unnormalized
            actor_effective_rank = effective_rank(tensordict_["_actor_feature"])
            critic_effective_rank = effective_rank(tensordict_["_critic_feature"])
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/policy_gain"] = policy_gain.mean().item()
        infos["actor/weighted_ratio"] = weighted_ratio.mean().item()
        infos["actor/effective_rank"] = actor_effective_rank.item()
        infos["critic/effective_rank"] = critic_effective_rank.item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/value_max"] = tensordict["ret"].max().item()
        reward_aggregated = tensordict["next", "reward_aggregated"]
        infos["critic/neg_rew_ratio"] = (reward_aggregated <= 0.).float().mean().item()
        infos["critic/adv_mean"] = adv_mean.item()
        infos["critic/adv_std"] = adv_std.item()

        if self.cfg.debug and self._rollout_dormancy_tracker is not None:
            dormancy = self._rollout_dormancy_tracker.compute_dormancy()
            for module_name, value in dormancy.items():
                infos[f"dormancy/{module_name}"] = value
            self._rollout_dormancy_tracker.reset()
        
        if aa.is_distributed():
            self.vecnorm.apply(vecnorm_sync_)
            if self.cfg.debug:
                actor_diff = check_parameters(self.actor)
                critic_diff = check_parameters(self.critic)
                infos["actor/diff"] = actor_diff
                infos["critic/diff"] = critic_diff
        
        self._obs_importance_step += 1
        if (
            self._obs_importance_step % self._obs_importance_interval == 0 and
            aa.is_main_process()
        ):
            value_grad, policy_grad = self.compute_grad_diagnostics(tensordict)
            if value_grad is not None:
                infos["obs_importance"] = plot_obs_importance(
                    value_grad, policy_grad, self.obs_func_keys, self.obs_split,
                )
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def compute_value(self, tensordict: TensorDict):
        return self.critic(tensordict)
    
    @torch.no_grad()
    def compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: Mod, 
        adv_key: str="adv",
        ret_key: str="ret",
        clamp_reward: bool = True,  # avoid suicide due to negative rewards
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(self.vecnorm(tensordict_flat))
                critic(self.vecnorm(tensordict_flat["next"]))

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if isinstance(rewards, TensorDict):
            rewards = torch.concat(list(rewards.values()), dim=-1)
        rewards = rewards.sum(-1, keepdim=True)
        tensordict["next", "reward_aggregated"] = rewards
        if clamp_reward:
            rewards = rewards.clamp_min(0.0)
        # scale according to the effective horizon
        rewards = rewards * (1. - self.gae.gamma)

        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    @ScopedTimer("compute_grad_diagnostics")
    def compute_grad_diagnostics(
        self,
        tensordict: TensorDict,
        *,
        max_samples: int = 256,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Per-observation-dimension gradient magnitudes for critic and policy.

        Returns CPU tensors ``(value_grad, policy_grad)`` of shape ``[obs_dim]``,
        or ``(None, None)`` when the minibatch has no valid transitions.

        Careful about GPU memory: works on a small *cloned* subsample (never
        mutates the rollout batch), unwraps DDP, freezes params via
        ``hold_out_net``, and drops the autograd graph before returning.
        """
        keys = list(self.training_keys)
        flat = tensordict.select(*keys).reshape(-1)
        valid_all = (~flat["is_init"]).reshape(-1)
        n_valid = int(valid_all.sum().item())
        if n_valid == 0:
            return None, None

        # Subsample valid rows only, then clone so we never alias the collector buffer.
        valid_idx = valid_all.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() > max_samples:
            valid_idx = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_samples]]
        td = flat[valid_idx].clone()
        del flat, valid_all, valid_idx

        n = td.shape[0]
        weight = td["is_init"].new_ones(n, dtype=torch.float32) / float(n)
        adv = td["adv"].detach().reshape(n)
        actions = td[ACTION_KEY].detach()

        # Bypass DDP hooks: forward the underlying modules only.
        critic = unwrap_ddp(self.critic)
        actor_body = unwrap_ddp(self.actor).module[0]
        vecnorm = self.vecnorm

        with hold_out_net(critic), hold_out_net(vecnorm):
            td_value = vecnorm(td.copy())
            obs_normed = td_value["_obs_normed"].detach().clone().requires_grad_(True)
            td_value["_obs_normed"] = obs_normed
            values = critic(td_value)["state_value"].reshape(n)
            value_obj = (values * weight).sum()
            (grad_value,) = torch.autograd.grad(
                value_obj, obs_normed, retain_graph=False, create_graph=False
            )
            grad_value_per_dim = grad_value.detach().abs().mean(0).cpu()
            del obs_normed, td_value, values, value_obj, grad_value

        with hold_out_net(actor_body), hold_out_net(vecnorm):
            td_policy = vecnorm(td.copy())
            obs_normed = td_policy["_obs_normed"].detach().clone().requires_grad_(True)
            td_policy["_obs_normed"] = obs_normed
            actor_body(td_policy)
            dist = IndependentNormal(td_policy["loc"], td_policy["scale"])
            log_probs = dist.log_prob(actions).reshape(n)
            policy_obj = (log_probs * adv * weight).sum()
            (grad_policy,) = torch.autograd.grad(
                policy_obj, obs_normed, retain_graph=False, create_graph=False
            )
            grad_policy_per_dim = grad_policy.detach().abs().mean(0).cpu()
            del obs_normed, td_policy, dist, log_probs, policy_obj, grad_policy

        del td, weight, adv, actions
        # Clear any param .grad that a DDP-wrapped path may have touched.
        self.opt.zero_grad(set_to_none=True)
        return grad_value_per_dim, grad_policy_per_dim

    def _augment_symmetry(self, tensordict: TensorDict) -> TensorDict:
        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        if self.cmd_transform is not None:
            symmetry[CMD_KEY] = self.cmd_transform(tensordict[CMD_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        return torch.cat([tensordict, symmetry])

    @ScopedTimer("ppo_update")
    def _update(self, tensordict: TensorDict, compute_diagnostics: bool = False):
        bsize = tensordict.shape[0] // 2

        self.vecnorm(tensordict)
        
        valid = (~tensordict["is_init"]).float()
        valid_cnt = valid.sum()
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        # dist: IndependentNormal = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"] # [bsize, 1]
        ret = tensordict["ret"] # [bsize, 1]
        log_ratio = (log_probs - log_probs_data).reshape_as(adv) # [bsize, 1]
        ratio = torch.exp(log_ratio)
        eps_neg, eps_pos = self.clip_param
        ratio_det = ratio.detach()
        clamped_pos = (ratio_det > 1.0 + eps_pos)
        clamped_neg = (ratio_det < 1.0 - eps_neg)
        clamped = (clamped_pos | clamped_neg).reshape_as(ret)

        policy_loss = self.actor_loss_fn(ratio, adv, self.clip_param)
        entropy_loss = - self.entropy_coef * entropy

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(ret, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        if self.cfg.aux_coef > 0.0:
            aux_weight = clamped.float() * valid
            aux_loss = (tensordict["aux_pred"].reshape_as(ret) - ret).square() * aux_weight
            aux_loss = aux_loss.sum() / aux_weight.sum().clamp_min(1.0)
            loss += self.cfg.aux_coef * aux_loss / max(self.ret_std_ema, 1.0) ** 2
        else:
            aux_loss = ret.new_zeros(())
        self.opt.zero_grad()
        loss.backward()

        if self.should_reduce_grads:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        if not compute_diagnostics:
            return

        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, ret) / ret.var()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
            actor_feature_norm = torch.norm(tensordict["_actor_feature"], dim=-1).mean()
            critic_feature_norm = torch.norm(tensordict["_critic_feature"], dim=-1).mean()

        
        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_pos": clamped_pos.float().mean(),
            "actor/clamp_neg": clamped_neg.float().mean(),
            "actor/approx_kl": approx_kl,
            "actor/aux_loss": aux_loss,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "actor/feature_norm": actor_feature_norm.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
            "critic/feature_norm": critic_feature_norm.detach(),
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        
        if self.cmd_transform is not None:
            state_dict["cmd_transform"] = self.cmd_transform.state_dict()
        state_dict["obs_transform"] = self.obs_transform.state_dict()
        state_dict["act_transform"] = self.act_transform.state_dict()
        state_dict["ret_std_ema"] = self.ret_std_ema
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        self.ret_std_ema = state_dict.get("ret_std_ema", 1.0)
        return failed_keys


def effective_rank(X: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Effective rank (entropy of normalized squared singular values) for a matrix X of shape [n, d].
    Uses p_i = σ_i² / Σσ_j² so p is the proportion of variance in each principal direction.
    Lower values indicate loss of expressivity (variance concentrated in few dimensions).
    """
    X = X.reshape(-1, X.shape[-1])
    if X.numel() == 0 or X.shape[0] < 2 or X.shape[1] < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    S = torch.linalg.svdvals(X)
    S = S[S > eps]
    if S.numel() == 0:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    S2 = S.square()
    p = S2 / S2.sum().clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum()
    return entropy.exp()


import matplotlib
matplotlib.use("Agg")  # must be before pyplot
import matplotlib.pyplot as plt
import wandb

@ScopedTimer("plot_obs_importance")
def plot_obs_importance(
    value_grad: torch.Tensor,
    policy_grad: torch.Tensor,
    obs_func_keys: list[str],
    obs_split: list[int],
):
    """Bar chart of per-obs-dim |grads|, colored by observation component.

    ``obs_labels[i]`` is the ObsGroup term name for dimension ``i`` (repeated
    for multi-dim terms). Value and policy gradients share the same color map
    and are shown in stacked subplots.
    """

    value_grad = value_grad.numpy()
    policy_grad = policy_grad.numpy()
    n = len(value_grad)

    # Preserve first-seen order of observation components.
    cmap = plt.get_cmap("tab20" if len(obs_func_keys) > 10 else "tab10")
    colors = []
    for i in range(len(obs_func_keys)):
        colors.extend([cmap(i)] * obs_split[i])

    x = np.arange(n)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(10, n * 0.18), 6),
        dpi=120,
        sharex=True,
        constrained_layout=True,
    )
    series = [
        (axes[0], value_grad, r"$|\partial V / \partial \mathrm{obs}|$"),
        (axes[1], policy_grad, r"$|\partial (\log\pi \cdot A) / \partial \mathrm{obs}|$"),
    ]
    for ax, grads, title in series:
        ax.bar(x, grads, width=0.9, color=colors, edgecolor="none")
        ax.set_ylabel("mean |gradient|")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[1].set_xticks(np.cumsum([0] + obs_split)[:-1])
    axes[1].set_xticklabels(obs_func_keys, rotation=45, ha="right", fontsize=8)

    image = wandb.Image(fig)
    plt.close(fig)
    return image
