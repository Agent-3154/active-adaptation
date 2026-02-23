import copy
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.distributions as D
import torch.nn as nn
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
    set_composite_lp_aggregate,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    GAE,
    make_batch,
    make_mlp,
    soft_copy_,
)
from active_adaptation.learning.utils.valuenorm import ValueNorm1, ValueNormFake
from active_adaptation.learning.ppo.ppo_base import PPOBase



torch.set_float32_matmul_precision("high")

OBJECT_KEY = "object"
REF_JPOS_KEY = "ref_joint_pos_"
PRIV_FEATURE_KEY = "priv_feature"
PRIV_PRED_KEY = "priv_pred"


@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_roa.PPOROA"
    name: str = "ppo_roa"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    residual_action: bool = True
    finetune_adapt_module: bool = True

    lr: float = 3e-4
    desired_kl: float | None = 0.01

    entropy_coef_start: float = 0.004
    entropy_coef_end: float = 0.004
    entropy_decay_iters: int = 3000
    init_noise_scale: float = 1.0
    load_noise_scale: float | None = None

    clip_neg_reward: bool = False
    normalize_before_sum: bool = False

    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    latent_dim: int = 256
    max_grad_norm: float = 1.0

    phase: str = "train"
    vecnorm: Union[str, None] = None
    checkpoint_path: Union[str, None] = None
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY)


cs = ConfigStore.instance()
cs.store(
    "ppo_roa_train",
    node=PPOConfig(
        phase="train", vecnorm="train", entropy_coef_start=0.004, entropy_coef_end=0.001
    ),
    group="algo",
)
cs.store(
    "ppo_roa_adapt",
    node=PPOConfig(
        phase="adapt", vecnorm="eval", entropy_coef_start=0.0, entropy_coef_end=0.0
    ),
    group="algo",
)
cs.store(
    "ppo_roa_finetune",
    node=PPOConfig(
        phase="finetune",
        vecnorm="eval",
        entropy_coef_start=0.002,
        entropy_coef_end=0.0001,
    ),
    group="algo",
)


class PPOROA(PPOBase):
    train_in_keys = [
        CMD_KEY,
        OBS_KEY,
        OBS_PRIV_KEY,
        ACTION_KEY,
        "adv",
        "ret",
        "is_init",
        "sample_log_prob",
        "step_count",
    ]

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "adapt", "finetune"]

        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=self.cfg.gamma, lmbda=self.cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales /= self.reward_scales.sum()
        value_norm_cls = ValueNorm1 if self.cfg.value_norm else ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)

        object.__setattr__(self, "env", env)

        self.action_dim = env.action_manager.action_dim
        self.joint_names = env.action_manager.joint_names

        self.cmd_key = CMD_KEY if CMD_KEY in observation_spec.keys(True, True) else "command_"

        self._build_vecnorm_modules(observation_spec)

        fake_input = observation_spec.zero()

        encoder_priv_in_keys = [self.norm_map[OBS_PRIV_KEY]]
        adapt_module_in_keys = [self.norm_map[OBS_KEY], self.norm_map[self.cmd_key]]
        critic_in_keys = [
            self.norm_map[OBS_PRIV_KEY],
            self.norm_map[OBS_KEY],
            self.norm_map[self.cmd_key],
        ]
        if OBJECT_KEY in observation_spec.keys(True, True):
            encoder_priv_in_keys.append(OBJECT_KEY)
            adapt_module_in_keys.append(OBJECT_KEY)
            critic_in_keys.append(OBJECT_KEY)

        latent_dim = self.cfg.latent_dim
        self.encoder_priv = Seq(
            CatTensors(encoder_priv_in_keys, "_encoder_priv_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp([latent_dim], norm=self.cfg.layer_norm),
                    nn.LazyLinear(latent_dim),
                ),
                "_encoder_priv_inp",
                PRIV_FEATURE_KEY,
            ),
            selected_out_keys=[PRIV_FEATURE_KEY],
        ).to(self.device)

        self.adapt_module = Seq(
            CatTensors(adapt_module_in_keys, "_adapt_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp([latent_dim, latent_dim], norm=self.cfg.layer_norm),
                    nn.LazyLinear(latent_dim),
                ),
                "_adapt_inp",
                [PRIV_PRED_KEY],
            ),
            selected_out_keys=[PRIV_PRED_KEY],
        ).to(self.device)

        if self.cfg.phase == "train" and self.cfg.residual_action and REF_JPOS_KEY in observation_spec:

            class RefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return (ref_jpos + action,)

            residual_module = Mod(RefJointPos(), [REF_JPOS_KEY, "loc"], ["loc"])
        else:
            residual_module = None

        def build_actor(in_keys: List[str], residual: Mod | None = None) -> ProbabilisticActor:
            actor_modules = [
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(
                    make_mlp([512, 256, 256], norm=self.cfg.layer_norm),
                    ["_actor_inp"],
                    ["_actor_feature"],
                ),
                Mod(
                    ActorROA(
                        self.action_dim,
                        init_noise_scale=self.cfg.init_noise_scale,
                        load_noise_scale=self.cfg.load_noise_scale,
                    ),
                    ["_actor_feature"],
                    ["loc", "scale"],
                ),
            ]
            if residual is not None:
                actor_modules.append(residual)
            return ProbabilisticActor(
                module=Seq(*actor_modules),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(self.device)

        self.dist_cls = IndependentNormal
        self.dist_keys = ["loc", "scale"]

        self.actor = build_actor(
            [self.norm_map[self.cmd_key], self.norm_map[OBS_KEY], PRIV_FEATURE_KEY],
            residual=residual_module,
        )
        self.actor_adapt = build_actor(
            [self.norm_map[self.cmd_key], self.norm_map[OBS_KEY], PRIV_PRED_KEY]
        )

        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False),
            Mod(
                nn.Sequential(
                    make_mlp([512, 256, 128], norm=self.cfg.layer_norm),
                    nn.LazyLinear(num_reward_groups),
                ),
                ["_critic_input"],
                ["state_value"],
            ),
        ).to(self.device)

        self.vecnorm(fake_input)
        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.adapt_module(fake_input)
        self.actor_adapt(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)

        if aa.is_distributed():
            self._wrap_ddp(local_rank=aa.get_local_rank())

        self.lr_policy = self.cfg.lr
        if self.cfg.phase == "train":
            policy_params = [
                {"params": self.actor.parameters()},
                {"params": self.encoder_priv.parameters()},
            ]
        else:
            policy_params = [{"params": self.actor_adapt.parameters()}]

        self.opt_policy = torch.optim.Adam(policy_params, lr=self.lr_policy)
        self.opt_critic = torch.optim.Adam([{"params": self.critic.parameters()}], lr=self.cfg.lr)
        self.opt_adapt = torch.optim.Adam([{"params": self.adapt_module.parameters()}], lr=self.cfg.lr)

        if self.cfg.phase == "train" and self.cfg.residual_action:
            self.opt_adapt_actor = torch.optim.Adam(
                [{"params": self.actor_adapt.parameters()}], lr=self.cfg.lr
            )

        self.num_updates = 0

    def _build_vecnorm_modules(self, observation_spec: CompositeSpec):
        modules = []
        self.norm_map = {}
        self.vecnorms = nn.ModuleDict()

        keys_to_norm = [self.cmd_key, OBS_KEY, OBS_PRIV_KEY]
        for key in keys_to_norm:
            if key not in observation_spec.keys(True, True):
                continue
            shape = observation_spec[key].shape[-1:]
            vecnorm = VecNorm(input_shape=shape, stats_shape=shape, decay=1.0)
            self.vecnorms[key] = vecnorm
            modules.append(Mod(vecnorm, [key], [key])) # inplace norm
            self.norm_map[key] = key

        self.vecnorm = Seq(*modules).to(self.device)
    
    def compute_value(self, tensordict):
        self.vecnorm(tensordict)
        return self.critic(tensordict)

    def _wrap_ddp(self, local_rank: int):
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

        class DDPWithAttr(DDP):
            def __getattr__(self, name: str):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if self.module is not None and hasattr(self.module, name):
                        return getattr(self.module, name)
                    raise

        def wrap_td_module(module):
            return DDPWithAttr(module, **ddp_kwargs)

        self.actor = wrap_td_module(self.actor)
        self.actor_adapt = wrap_td_module(self.actor_adapt)
        self.encoder_priv = wrap_td_module(self.encoder_priv)
        self.critic = wrap_td_module(self.critic)
        self.adapt_module = wrap_td_module(self.adapt_module)
        if hasattr(self, "adapt_ema"):
            self.adapt_ema = self.adapt_ema.to(self.device)

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done")

    def on_stage_start(self, stage: str):
        return None

    def _get_current_iter(self) -> int:
        return int(getattr(self.env, "current_iter", 0))

    def get_rollout_policy(self, mode: str = "train"):
        modules = [self.vecnorm]

        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.adapt_module, self.actor]
        elif self.cfg.phase == "adapt":
            modules += [self.adapt_module, self.actor_adapt]
        elif self.cfg.phase == "finetune":
            modules += [self.adapt_module, self.actor_adapt]

        if mode == "deploy":
            modules[-1] = modules[-1].module[0]
            modules.append(MeanAction())
            out_keys = [ACTION_KEY]
        else:
            out_keys = ["sample_log_prob", ACTION_KEY] + self.dist_keys
            if self.cfg.phase == "finetune":
                out_keys.append(PRIV_PRED_KEY)

        return Seq(*modules, selected_out_keys=out_keys)

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")

        info = {}
        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))

        self.num_updates += 1

        if aa.is_distributed():
            for m in [self.value_norm]:
                for p in m.parameters():
                    dist.all_reduce(p, op=dist.ReduceOp.AVG)
                for b in m.buffers():
                    dist.all_reduce(b, op=dist.ReduceOp.AVG)

            for name, vecnorm in self.vecnorms.items():
                loc_diffs, scale_diffs = check_vecnorm_divergence(vecnorm)
                if aa.is_main_process():
                    info[f"vecnorm/{name}/loc_diff_max"] = max(loc_diffs)
                    info[f"vecnorm/{name}/scale_diff_max"] = max(scale_diffs)
                    info[f"vecnorm/{name}/loc_diff_mean"] = sum(loc_diffs) / len(loc_diffs)
                    info[f"vecnorm/{name}/scale_diff_mean"] = sum(scale_diffs) / len(scale_diffs)
                vecnorm.synchronize(mode="broadcast")

        action_std = self._get_actor_std(self.actor if self.cfg.phase == "train" else self.actor_adapt)
        if action_std is not None:
            for joint_name, std in zip(self.joint_names, action_std):
                info[f"actor_std/{joint_name}"] = std
            info["actor_std/mean"] = action_std.mean()

        return info

    def _get_actor_std(self, actor_module):
        module = actor_module.module if isinstance(actor_module, DDP) else actor_module
        for _, p in module.named_parameters():
            if p.ndim == 1 and p.shape[0] == self.action_dim:
                return p.detach()
        return None

    def train_policy(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)

        current_iter = self._get_current_iter()
        progress = float(np.clip(current_iter / self.cfg.entropy_decay_iters, 0.0, 1.0))
        self.entropy_coef = self.cfg.entropy_coef_start + (
            self.cfg.entropy_coef_end - self.cfg.entropy_coef_start
        ) * progress

        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                info = self._update_ppo(minibatch)
                infos.append(info)

                if self.desired_kl is not None:
                    kl = info["actor/kl"]
                    if aa.is_distributed():
                        dist.all_reduce(kl, op=dist.ReduceOp.AVG)
                    if kl > self.desired_kl * 2.0:
                        self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr_policy = min(1e-2, self.lr_policy * 1.5)

                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = self.lr_policy

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr_policy
        infos["actor/entropy_coef"] = self.entropy_coef

        ret = tensordict["ret"]
        ret_mean = ret.mean(dim=(0, 1))
        ret_std = ret.std(dim=(0, 1))
        for i, group_name in enumerate(self.reward_groups):
            infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
            infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
            infos[f"critic/{group_name}.neg_rew_ratio"] = (
                (tensordict[REWARD_KEY][:, :, i] <= 0.0).float().mean().item()
            )
        return dict(sorted(infos.items()))

    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder_priv(tensordict)

        for _ in range(2):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                self.adapt_module(minibatch)
                priv_loss = self.adapt_loss_fn(minibatch[PRIV_PRED_KEY], minibatch[PRIV_FEATURE_KEY])
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                self.opt_adapt.zero_grad()
                priv_loss.backward()
                opt_adapt_grad_norm = nn.utils.clip_grad_norm_(
                    self.adapt_module.parameters(), self.cfg.max_grad_norm
                )
                self.opt_adapt.step()

                info = {
                    "adapt/priv_loss": priv_loss,
                    "adapt/grad_norm": opt_adapt_grad_norm,
                    "adapt/priv_feature_norm": minibatch[PRIV_FEATURE_KEY].norm(p=2, dim=-1).mean(),
                    "adapt/priv_pred_norm": minibatch[PRIV_PRED_KEY].norm(p=2, dim=-1).mean(),
                }

                if self.cfg.phase == "train" and self.cfg.residual_action:
                    with torch.no_grad():
                        dist_teacher = self.actor.get_dist(minibatch)

                    minibatch[PRIV_PRED_KEY] = minibatch[PRIV_FEATURE_KEY].detach()
                    dist_student = self.actor_adapt.get_dist(minibatch)

                    adapt_loss = (dist_teacher.mean - dist_student.mean).square().mean()
                    self.opt_adapt_actor.zero_grad()
                    adapt_loss.backward()
                    self.opt_adapt_actor.step()
                    info["adapt/adapt_loss"] = adapt_loss

                infos.append(TensorDict(info, []))

        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        return {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: Mod,
        adv_key: str = "adv",
        ret_key: str = "ret",
        update_value_norm: bool = True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.0)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        def _global_mean_std(x, mask):
            if aa.is_distributed():
                local_count = mask.sum()
                local_sum = (x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                local_sum_sq = (x * x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                expand_count = local_count.float().expand_as(local_sum)

                stats = torch.stack([local_sum, local_sum_sq, expand_count])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                global_sum, global_sum_sq, global_count = stats
                global_count.clamp_min_(1)

                mean = global_sum / global_count
                var = (global_sum_sq / global_count) - (mean * mean)
                std = var.clamp_min(0.0).sqrt()
            else:
                mean = x.mean(dim=(0, 1))
                std = x.std(dim=(0, 1))
            return mean, std

        mask = ~tensordict["is_init"].squeeze(-1)

        if self.cfg.normalize_before_sum:
            mean, std = _global_mean_std(adv, mask)
            adv_norm = (adv - mean) / (std + 1e-5)
            adv_norm *= self.reward_scales
            adv_final = adv_norm.sum(dim=2, keepdim=True)
        else:
            adv *= self.reward_scales
            adv_sum = adv.sum(dim=2, keepdim=True)
            mean, std = _global_mean_std(adv_sum, mask)
            adv_final = (adv_sum - mean) / (std + 1e-5)

        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv_final)
        tensordict.set(ret_key, ret)
        tensordict["adv_before_norm"] = adv
        return tensordict

    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)

        if self.cfg.phase == "train":
            self.encoder_priv(tensordict)
            actor = self.actor
        elif self.cfg.phase == "finetune":
            if self.cfg.finetune_adapt_module:
                self.adapt_module(tensordict)
            actor = self.actor_adapt
        else:
            raise ValueError(f"Invalid phase: {self.cfg.phase}")

        dist_now: D.Independent = actor.get_dist(tensordict)
        with set_composite_lp_aggregate(True):
            log_probs = dist_now.log_prob(tensordict[ACTION_KEY])
        entropy = dist_now.entropy().mean()

        valid = (tensordict["step_count"] > (1 if self.cfg.phase == "train" else 5)).squeeze(-1)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = -self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = value_loss[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()

        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        if self.cfg.phase == "train":
            priv_grad_norm = nn.utils.clip_grad_norm_(
                self.encoder_priv.parameters(), self.cfg.max_grad_norm
            )
        else:
            priv_grad_norm = torch.zeros(1, device=self.device)
        self.opt_policy.step()
        self.opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            dist_old = self.dist_cls(**dist_kwargs_old)
            kl = D.kl_divergence(dist_old, dist_now).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/priv_grad_norm": priv_grad_norm,
            "actor/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    def state_dict(self):
        if self.cfg.phase == "train":
            if not self.cfg.residual_action:
                for src, dst in zip(self.actor.parameters(), self.actor_adapt.parameters()):
                    dst.data.copy_(src.data)
        if self.cfg.phase in ["train", "adapt"]:
            self.adapt_ema.load_state_dict(self.adapt_module.state_dict())

        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
        state_dict["last_iter"] = self._get_current_iter()
        state_dict["lr_policy"] = self.lr_policy
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module.module.load_state_dict(_state_dict, strict=strict)
                else:
                    module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        start_iter = state_dict.get("last_iter", 0)
        if self.cfg.phase != state_dict.get("last_phase", self.cfg.phase):
            start_iter = 0
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy

        return failed_keys


def check_vecnorm_divergence(vecnorm: VecNorm):
    world_size = aa.get_world_size()

    loc, scale = vecnorm._compute()
    gather_loc = [torch.empty_like(loc) for _ in range(world_size)]
    gather_scale = [torch.empty_like(scale) for _ in range(world_size)]
    dist.all_gather(gather_loc, loc)
    dist.all_gather(gather_scale, scale)

    loc_diffs = []
    scale_diffs = []
    for i in range(world_size):
        loc_diff = torch.abs(gather_loc[i] - loc).sum().item()
        scale_diff = torch.abs(gather_scale[i] - scale).sum().item()
        loc_diffs.append(loc_diff)
        scale_diffs.append(scale_diff)
    return loc_diffs, scale_diffs


class MeanAction(TensorDictModuleBase):
    in_keys = ["loc"]
    out_keys = [ACTION_KEY]

    def forward(self, td):
        td[ACTION_KEY] = td["loc"]
        return td


class ActorROA(nn.Module):
    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        load_noise_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)
