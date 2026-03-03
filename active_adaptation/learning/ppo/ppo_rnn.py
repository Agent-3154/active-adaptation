# MIT License
#
# Copyright (c) 2023 Botian Xu
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

from dataclasses import dataclass, field
from typing import Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
import einops

from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq, 
)

from torchrl.data import Composite, TensorSpec, UnboundedContinuous
from torchrl.envs import CatTensors, TensorDictPrimer
from torchrl.modules import ProbabilisticActor

from active_adaptation.learning.modules import GRUCore, IndependentNormal, VecNorm
from .common import *
from .ppo_base import PPOBase

@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_rnn.PPORNNPolicy"
    name: str = "ppo_rnn"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    seq_len: int = train_every
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    hidden_size: int = 128

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo_gru", node=PPOConfig, group="algo")


class GRUEncoder(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int):
        super().__init__()
        self.inp = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.LayerNorm(hidden_size), nn.Mish(),
        )
        self.gru = GRUCore(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.LazyLinear(hidden_size),
        )
    
    def forward(self, x: torch.Tensor, hx: torch.Tensor, is_init: torch.Tensor):
        x = self.inp(x)
        x, next_hx = self.gru(x, hx, is_init)
        x = self.out(x)
        return x, next_hx


class PPORNNPolicy(PPOBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.action_dim = env.action_manager.action_dim
        self.observation_shape = observation_spec[OBS_KEY].shape[-1:]

        fake_input = observation_spec.zero()
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["actor_hx"] = torch.zeros(fake_input.shape[0], 128)
            fake_input["critic_hx"] = torch.zeros(fake_input.shape[0], 128)

        vecnorm = VecNorm(
            input_shape=observation_spec[OBS_KEY].shape[-1:],
            stats_shape=observation_spec[OBS_KEY].shape[-1:],
            decay=1.0
        )
        self.vecnorm = Mod(vecnorm, [OBS_KEY], ["_obs_normed"]).to(self.device)
        actor_module = Seq(
            Mod(
                GRUEncoder(self.observation_spec[0], 128),
                ["_obs_normed", "actor_hx", "is_init"],
                ["_actor_feature", ("next", "actor_hx")]
            ),
            Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
        )
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = Seq(
            Mod(
                GRUEncoder(self.observation_spec[0], 128),
                ["_obs_normed", "critic_hx", "is_init"],
                ["_critic_feature", ("next", "critic_hx")]
            ),
            Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)
        # self.critic = Seq(
        #     Mod(make_mlp([256, 256, 256]), ["_obs_normed"], ["_critic_feature"]),
        #     Mod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        # ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )

    def get_rollout_policy(self, mode: str):
        return Seq(self.vecnorm, self.actor)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"actor_hx": UnboundedContinuous((num_envs, 128), device=self.device),
            "critic_hx": UnboundedContinuous((num_envs, 128), device=self.device)},
            reset_key="done",
            expand_specs=False
        )

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")

        infos = []
        self.vecnorm(tensordict)
        self.vecnorm(tensordict["next"])
        self.compute_advantage(tensordict, self.critic, "adv", "ret")

        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"]
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.seq_len)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
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
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 2.)
        self.opt.step()

        info = {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/grad_norm": actor_grad_norm,
            "critic/grad_norm": critic_grad_norm,
            "critic/value_loss": value_loss,
        }
        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var()
            clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean()
            info["actor/clamp_ratio"] = clipfrac
            info["critic/explained_var"] = explained_var
        return info

