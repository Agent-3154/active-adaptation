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
import functools

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod,
    TensorDictSequential as Seq
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import *


@torch.no_grad()
def grad_norm(parameters):
    norms = torch._foreach_norm(list(parameters), ord=2)
    total_norm = torch.linalg.norm(torch.stack(norms), ord=2)
    return total_norm


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_go2.PPOPolicy"
    name: str = "ppo_go2"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    symaug: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, "extero")


cs = ConfigStore.instance()
cs.store("ppo_go2", node=PPOConfig, group="algo")


class ResidualFC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.LazyLinear(dim)
        self.act = nn.Mish()
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.ln(self.act(self.linear(x)) + x)


class MixedEncoder(nn.Module):
    def __init__(self, proprio_shape: torch.Size, extero_shape: torch.Size):
        super().__init__()
        self.proprio_shape = proprio_shape
        self.extero_shape = extero_shape
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256)
        )
        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    nn.Conv2d(extero_shape[0], 8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                    nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
                    nn.Flatten(),
                ),
                data_dim=3,
            ),
            nn.LazyLinear(32),
            nn.Mish(),
            nn.LayerNorm(32),
            nn.LazyLinear(256)
        )
        self.out = nn.Mish()

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        cnn_feature = self.cnn_encoder(cnn_inp)
        mlp_feature = self.mlp_encoder(mlp_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        feature = mlp_feature + cnn_feature
        return self.out(feature)


import einops
from active_adaptation.learning.modules.pos_emb import PositionEncodingND
class CrossAttnEncoder(nn.Module):
    def __init__(self, proprio_shape: torch.Size, extero_shape: torch.Size):
        super().__init__()
        self.proprio_shape = proprio_shape
        self.extero_shape = extero_shape
        assert len(proprio_shape) == 1
        assert len(extero_shape) == 3
        
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256),
        )
        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    nn.Conv2d(extero_shape[0], 8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                ),
                data_dim=3,
            ),
        )

        with torch.no_grad():
            shape = self.cnn_encoder(torch.zeros(1, *extero_shape)).shape[-2:]
        self.pos_enc = PositionEncodingND(shape)
        
        self.embed_dim = 64
        self.propri_proj = nn.LazyLinear(self.embed_dim)
        self.extero_proj = nn.LazyLinear(self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=2,
            batch_first=True,
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.Mish(),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )
        self.out = nn.LazyLinear(256)

    def forward(self, propri, extero, mask_cnn):
        propri_feature = self.mlp_encoder(propri)
        propri_feature = einops.rearrange(propri_feature, "... (m c) -> ... m c", m=1)
        propri_feature = self.propri_proj(propri_feature)
        propri_slots = propri_feature.shape[-2]

        extero_feature = self.cnn_encoder(extero)
        extero_feature = self.pos_enc(extero_feature)
        extero_feature = einops.rearrange(extero_feature, "... c h w -> ... (h w) c")
        extero_feature = self.extero_proj(extero_feature)
        extero_slots = extero_feature.shape[-2]

        Q = propri_feature
        K = extero_feature
        V = extero_feature
        attn_output, _ = self.attn(Q, K, V, need_weights=False)
        feature = propri_feature + attn_output
        
        # if mask_cnn is not None:
        #     key_padding_mask = torch.zeros(*Q.shape[:-2], propri_slots + extero_slots, dtype=bool, device=Q.device)
        #     key_padding_mask[..., propri_slots:] = ~mask_cnn
        #     attn_output, attn_output_weights = self.attn(Q, K, V, key_padding_mask=key_padding_mask)
        # else:
        #     attn_output, attn_output_weights = self.attn(Q, K, V)
        # feature = propri_feature + attn_output
        # feature = feature + self.mlp(feature)
        feature = self.out(attn_output.flatten(-2))
        return feature
        
        

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
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)

        self.obs_transform = env.observation_funcs["policy"].symmetry_transform().to(self.device)
        self.extero_transform = env.observation_funcs["extero"].symmetry_transform().to(self.device)
        self.act_transform = env.action_manager.symmetry_transform().to(self.device)
        self.action_dim = env.action_manager.action_dim
        
        fake_input = observation_spec.zero()
        self.vecnorm_proprio = VecNorm(
            input_shape=observation_spec[OBS_KEY].shape[-1:],
            stats_shape=observation_spec[OBS_KEY].shape[-1:],
            decay=1.0
        ).to(self.device)

        self.vecnorm_extero = VecNorm(
            input_shape=observation_spec["extero"].shape[-2:],
            stats_shape=observation_spec["extero"].shape[-2:],
            decay=1.0
        ).to(self.device)

        self.vecnorm = Seq(
            Mod(self.vecnorm_proprio, [OBS_KEY], ["policy_normed"]),
            Mod(self.vecnorm_extero, ["extero"], ["extero_normed"]),
        ).to(self.device)
        
        encoder_cls = CrossAttnEncoder
        _actor = nn.Sequential(ResidualFC(256), Actor(self.action_dim))
        self.actor_encoder = encoder_cls(
            observation_spec[OBS_KEY].shape[-1:],
            observation_spec["extero"].shape[-3:]
        )
        actor_module = Seq(
            Mod(self.actor_encoder, ["policy_normed", "extero_normed", "cnn_mask"], ["actor_feature"]),
            Mod(_actor, ["actor_feature"], ["loc", "scale"]),
            Mod(nn.LazyLinear(1), ["actor_feature"], ["aux_pred"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(ResidualFC(256), nn.LazyLinear(1))
        self.critic_encoder = encoder_cls(
            observation_spec[OBS_KEY].shape[-1:],
            observation_spec["extero"].shape[-3:]
        )
        self.critic = Seq(
            Mod(self.critic_encoder, ["policy_normed", "extero_normed", "cnn_mask"], ["critic_feature"]),
            Mod(_critic, ["critic_feature"], ["state_value"])
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)
    
    def get_rollout_policy(self, mode: str="train", critic: bool = False):
        if critic:
            policy = Seq(self.vecnorm, self.actor, self.critic)
        else:
            policy = Seq(self.vecnorm, self.actor)
        return policy
    
    def on_stage_start(self, stage: str):
        pass

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"

        tensordict = tensordict.exclude("stats")
        infos = []

        self.compute_advantage(tensordict, self.critic, "adv", "ret")
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)
        train_in_keys = (
            ["policy", "extero"]
            + ["action", "action_log_prob"]
            + ["adv", "ret", "is_init"]
        )

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict.select(*train_in_keys, strict=False), self.cfg.num_minibatches)
            for minibatch in batch:
                if self.cfg.symaug:
                    mirrored = minibatch.empty()
                    mirrored["policy"] = self.obs_transform(minibatch["policy"])
                    mirrored["extero"] = self.extero_transform(minibatch["extero"])
                    mirrored["action"] = self.act_transform(minibatch["action"])
                    mirrored["action_log_prob"] = minibatch["action_log_prob"]
                    mirrored["adv"] = minibatch["adv"]
                    mirrored["ret"] = minibatch["ret"]
                    mirrored["is_init"] = minibatch["is_init"]
                    minibatch = torch.cat([minibatch, mirrored], dim=0)
                infos.append(TensorDict(self._update(minibatch), []))
        
        with torch.no_grad(), torch.device(self.device), tensordict.view(-1) as tensordict_flat:
            self.vecnorm(tensordict_flat)
            ones = torch.ones(*tensordict_flat.shape, 1, dtype=torch.bool)
            zeros = torch.zeros(*tensordict_flat.shape, 1, dtype=torch.bool)
            a = self.critic(tensordict_flat.replace(cnn_mask=zeros))
            b = self.critic(tensordict_flat.replace(cnn_mask=ones))
            value_diff = F.mse_loss(a["state_value"], b["state_value"])
            a = self.actor.get_dist(tensordict_flat.replace(cnn_mask=zeros))
            b = self.actor.get_dist(tensordict_flat.replace(cnn_mask=ones))
            policy_diff = F.mse_loss(a.mean, b.mean)

        infos = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["actor/policy_diff"] = policy_diff.item()
        infos["critic/value_diff"] = value_diff.item()
        infos["actor/feature_norm"] = tensordict["actor_feature"].norm(dim=-1).mean().item()
        infos["critic/feature_norm"] = tensordict["critic_feature"].norm(dim=-1).mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def compute_value(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
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

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp(min=0.0)
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        discount = tensordict["next", "discount"]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["action_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * (~tensordict["is_init"]))
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss * (~tensordict["is_init"])).mean()
        
        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        actor_cnn_grad_norm = grad_norm(self.actor_encoder.cnn_encoder.parameters())
        critic_cnn_grad_norm = grad_norm(self.critic_encoder.cnn_encoder.parameters())
        self.opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        info = {
            "actor/policy_loss": policy_loss,
            "actor/entropy": entropy,
            "actor/noise_std": tensordict["scale"].mean(),
            "actor/grad_norm": actor_grad_norm,
            'actor/approx_kl': ((ratio - 1) - log_ratio).mean(),
            "critic/value_loss": value_loss,
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }
        if hasattr(self.actor_encoder, "cnn_encoder"):
            info["actor/cnn_grad_norm"] = actor_cnn_grad_norm
        if hasattr(self.critic_encoder, "cnn_encoder"):
            info["critic/cnn_grad_norm"] = critic_cnn_grad_norm
        return {
            
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys

