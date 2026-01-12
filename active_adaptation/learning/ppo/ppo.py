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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.utils._pytree as pytree
import warnings

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import (
    VecNorm, 
    IndependentNormal, 
    SymmetryWrapper,
)
from .common import *
from .ppo_base import PPOBase

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

# @dataclass
# class PPOConfig:
#     _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
#     name: str = "ppo"
#     train_every: int = 32
#     ppo_epochs: int = 4
#     num_minibatches: int = 4
#     lr: float = 5e-4
#     desired_kl: Union[float, None] = None
#     clip_param: float = 0.2
#     entropy_coef: float = 0.002
#     layer_norm: Union[str, None] = "before"

#     # symmetry options
#     symnet: bool = False # use symmetry wrapper to wrap the policy and critic
#     symaug: bool = False # use symmetry augmentation

#     compile: bool = False
#     use_ddp: bool = True

#     checkpoint_path: Union[str, None] = None
#     in_keys: Tuple[str, ...] = (OBS_KEY,)

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
    name: str = "ppo"
    train_every: int = 24
    ppo_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 1e-3
    desired_kl: Union[float, None] = 0.01
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    layer_norm: Union[str, None] = "before"

    # symmetry options
    symnet: bool = False # use symmetry wrapper to wrap the policy and critic
    symaug: bool = False # use symmetry augmentation

    compile: bool = False
    use_ddp: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")


class PPOPolicy(PPOBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device

        self.max_grad_norm = 1.0
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        vecnorm_policy = VecNorm(
            input_shape=observation_spec[OBS_KEY].shape[-1:],
            stats_shape=observation_spec[OBS_KEY].shape[-1:],
            decay=1.0
        )
        
        vecnorm_priv = VecNorm(
            input_shape=observation_spec[OBS_PRIV_KEY].shape[-1:],
            stats_shape=observation_spec[OBS_PRIV_KEY].shape[-1:],
            decay=1.0
        )

        self.vecnorm = Seq(
            Mod(vecnorm_policy, [OBS_KEY], ["_obs_normed"]),
            Mod(vecnorm_priv, [OBS_PRIV_KEY], ["_priv_normed"])
        ).to(self.device)

        self.action_dim = env.action_manager.action_dim
        
        actor_module = Seq(
            Mod(make_mlp([512, 256, 128]), ["_obs_normed"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
        )
        
        self.critic = Seq(
            CatTensors(["_obs_normed", "_priv_normed"], "_critic_input"),
            Mod(make_mlp([512, 256, 128]), ["_critic_input"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        if self.cfg.symnet:
            self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
            self.act_transform = env.action_manager.symmetry_transform().to(self.device)
            actor_module = SymmetryWrapper(
                actor_module,
                Mod(self.obs_transform, ["_obs_normed"], ["_obs_normed"]),
                Seq(
                    Mod(self.act_transform, ["loc"], ["loc"]),
                    Mod(self.act_transform.permutation(), ["scale"], ["scale"]),
                )
            )
            self.critic = SymmetryWrapper(
                self.critic,
                Mod(self.obs_transform, ["_obs_normed"], ["_obs_normed"]),
                Mod(nn.Identity(), ["state_value"], ["state_value"])
            )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            self.world_size = active_adaptation.get_world_size()
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        
        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

        self.update = self._update
        if self.cfg.compile and not active_adaptation.is_distributed():
            self.update = torch.compile(self.update, fullgraph=True)

    def get_rollout_policy(self, mode: str="train"):
        policy = Seq(self.vecnorm, self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy, fullgraph=True)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats")
        valid_ratio = (~tensordict["is_init"]).sum() / tensordict.numel()

        infos = []
        self.vecnorm(tensordict)
        self.vecnorm(tensordict["next"])
        self.compute_advantage(tensordict, self.critic, "adv", "ret")
        
        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"]
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))

                if self.cfg.desired_kl is not None: # adaptive learning rate
                    kl = infos[-1]["actor/approx_kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.cfg.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.cfg.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-3, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr
        
        with torch.no_grad():
            tensordict_ = self.actor(tensordict.copy())
            dist = IndependentNormal(tensordict_["loc"], tensordict_["scale"])
            log_probs_after = dist.log_prob(action)
            pg_loss_after = log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            pg_loss_before = log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized
        
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        infos["critic/valid_ratio"] = valid_ratio.item()

        if active_adaptation.is_distributed():
            loc_diffs, scale_diffs = check_vecnorm_divergence(self.vecnorm[0].module)
            if active_adaptation.is_main_process():
                infos["vecnorm/loc_diff_max"] = max(loc_diffs)
                infos["vecnorm/scale_diff_max"] = max(scale_diffs)
                infos["vecnorm/loc_diff_mean"] = sum(loc_diffs) / len(loc_diffs)
                infos["vecnorm/scale_diff_mean"] = sum(scale_diffs) / len(scale_diffs)
            self.vecnorm[0].module.synchronize(mode="broadcast")
        return dict(sorted(infos.items()))

    def _update(self, tensordict: TensorDict):
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        
        valid = (~tensordict["is_init"])
        valid_cnt = valid.sum()

        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.cfg.clip_param, 1.+self.cfg.clip_param)
        policy_loss = - (torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
        entropy_loss = - self.cfg.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt
        
        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed() and not self.cfg.use_ddp:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad.data, op=distr.ReduceOp.SUM)
                param.grad.data /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad.data, op=distr.ReduceOp.SUM)
                param.grad.data /= self.world_size
        
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
        }
        with torch.no_grad():
            info["critic/explained_var"] = 1 - value_loss / b_returns[valid].var()
            info["actor/clamp_ratio"] = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
            info["actor/approx_kl"] = ((ratio - 1.0) - log_ratio).mean()
        return info


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)


def check_vecnorm_divergence(vecnorm: VecNorm):
    WORLD_SIZE = active_adaptation.get_world_size()
    
    loc, scale = vecnorm._compute()
    gather_loc = [torch.empty_like(loc) for _ in range(WORLD_SIZE)]
    gather_scale = [torch.empty_like(scale) for _ in range(WORLD_SIZE)]
    distr.all_gather(gather_loc, loc)
    distr.all_gather(gather_scale, scale)
    
    loc_diffs = []
    scale_diffs = []
    for i in range(WORLD_SIZE):
        loc_diff = torch.abs(gather_loc[i] - loc).sum().item()
        scale_diff = torch.abs(gather_scale[i] - scale).sum().item()
        loc_diffs.append(loc_diff)
        scale_diffs.append(scale_diff)
    return loc_diffs, scale_diffs
    
    