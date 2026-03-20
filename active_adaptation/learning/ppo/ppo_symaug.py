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
import warnings
import torch.utils._pytree as pytree

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
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

from active_adaptation.learning.modules import VecNorm, IndependentNormal
from active_adaptation.learning.ppo.common import (
    ppo_clipped_loss,
    spo_loss,
    normalize,
    OBS_KEY,
    OBS_PRIV_KEY,
    OBS_HIST_KEY,
    ACTION_KEY,
    REWARD_KEY,
    TERM_KEY,
    DONE_KEY,
    CMD_KEY,
    GAE,
    make_batch,
    make_mlp,
    Actor,
)
from active_adaptation.learning.utils.opt import OptimizerGroup
from active_adaptation.learning.utils.distributed import check_parameters

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_symaug.PPOPolicy"
    name: str = "ppo_symaug"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 5e-4
    desired_kl: Union[float, None] = None
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    spo: bool = False # use Simple Policy Optimization Loss
    muon: bool = False # use Muon optimizer
    aux_coef: float = 0.0 # loss coefficient for auxiliary prediction loss
    
    compile: bool = False
    use_ddp: bool = True
    debug: bool = False # enable correctness checkers

    in_keys: Tuple[str, ...] = (OBS_KEY,)

cs = ConfigStore.instance()
cs.store("ppo_symaug", node=PPOConfig, group="algo")


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: Composite, 
        action_spec: Composite, 
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.actor_loss_fn = spo_loss if self.cfg.spo else ppo_clipped_loss
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)  

        fake_input = observation_spec.zero()
        
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
        self.act_transform = env.action_manager.symmetry_transform().to(self.device)
        self.action_dim = env.action_manager.action_dim

        self.vecnorm = Mod(
            VecNorm(
                input_shape=observation_spec[OBS_KEY].shape[-1:],
                stats_shape=observation_spec[OBS_KEY].shape[-1:],
                decay=1.0
            ),
            in_keys=[OBS_KEY],
            out_keys=["_obs_normed"]
        ).to(self.device)
        
        actor_modules = [
            Mod(make_mlp([256, 256, 256]), ["_obs_normed"], ["_actor_feature"]),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
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
        
        self.critic = Seq(
            Mod(make_mlp([256, 256, 256]), ["_obs_normed"], ["_critic_feature"]),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
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

        if active_adaptation.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        self.world_size = active_adaptation.get_world_size()
            
        def is_matrix_shaped(param: torch.Tensor) -> bool:
            return param.dim() >= 2
        
        if self.cfg.muon:
            muon = torch.optim.Muon([
                {"params": [p for p in self.actor.parameters() if is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if is_matrix_shaped(p)]},
            ], lr=cfg.lr, adjust_lr_fn="match_rms_adamw", weight_decay=0.01)

            adamw = torch.optim.AdamW([
                {"params": [p for p in self.actor.parameters() if not is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if not is_matrix_shaped(p)]},
            ], lr=cfg.lr, weight_decay=0.01)
            self.opt = OptimizerGroup([muon, adamw])
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=cfg.lr,
                weight_decay=0.01
            )

        self.update = self._update
        if self.cfg.compile and not active_adaptation.is_distributed():
            # TODO: compile for multi-gpu training?
            self.update = torch.compile(self.update, fullgraph=True)
            # self.update = CudaGraphModule(self.update)
    
    def on_stage_start(self, stage: str):
        pass

    def get_rollout_policy(self, mode: str="train", critic: bool = False):
        if critic:
            policy = Seq(self.vecnorm, self.actor, self.critic)
        else:
            policy = Seq(self.vecnorm, self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy, fullgraph=True)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"

        tensordict = tensordict.exclude("stats")
        infos = []
        self.compute_advantage(tensordict, self.critic, "adv", "ret")
        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"]
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))
                
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
            pg_loss_after = log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            pg_loss_before = log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized
            actor_feature = tensordict_["_actor_feature"].reshape(-1, tensordict_["_actor_feature"].shape[-1]) # [N*T, D]
            critic_feature = tensordict_["_critic_feature"].reshape(-1, tensordict_["_critic_feature"].shape[-1]) # [N*T, D]
            actor_effective_rank = effective_rank(actor_feature)
            critic_effective_rank = effective_rank(critic_feature)
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()
        infos["actor/effective_rank"] = actor_effective_rank.item()
        infos["critic/effective_rank"] = critic_effective_rank.item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        if active_adaptation.is_distributed():
            self.vecnorm.module.synchronize(mode="broadcast")
            if self.cfg.debug:
                actor_diff = check_parameters(self.actor)
                critic_diff = check_parameters(self.critic)
                infos["actor/diff"] = actor_diff
                infos["critic/diff"] = critic_diff
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
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(self.vecnorm(tensordict_flat))
                critic(self.vecnorm(tensordict_flat["next"]))

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict["loc"], tensordict["scale"]

        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)

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
        clamped = ((ratio.detach() - 1.0).abs() > self.clip_param).reshape_as(ret)
        
        policy_loss = self.actor_loss_fn(ratio, adv, self.clip_param)
        entropy_loss = - self.entropy_coef * entropy

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(ret, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        if self.cfg.aux_coef > 0.0:
            aux_loss = (tensordict["aux_pred"].reshape_as(ret) - ret).square() * clamped
            aux_loss = aux_loss.sum() / clamped.sum().clamp_min(1.0)
            loss += self.cfg.aux_coef * aux_loss
        else:
            aux_loss = torch.tensor(0.0)
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed() and not self.cfg.use_ddp:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, ret) / ret.var()
            clipfrac = clamped.float().mean()
            loc, scale = dist.loc[:bsize], dist.scale[:bsize]
            kl = torch.sum(
                torch.log(scale) - torch.log(scale_old)
                + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
                - 0.5,
                axis=-1,
            ).mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
            actor_feature_norm = torch.norm(tensordict["_actor_feature"], dim=-1).mean()
            critic_feature_norm = torch.norm(tensordict["_critic_feature"], dim=-1).mean()
        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
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
        return failed_keys


def effective_rank(X: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Effective rank (entropy of normalized squared singular values) for a matrix X of shape [n, d].
    Uses p_i = σ_i² / Σσ_j² so p is the proportion of variance in each principal direction.
    Lower values indicate loss of expressivity (variance concentrated in few dimensions).
    """
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


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
