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
"""
Teacher–Student PPO (two stages).

Key groups (config)
-------------------
- ``teacher_keys``: observations available to the teacher / critic.
- ``student_keys``: observations available to the student.
- ``privileged_keys = teacher_keys \\ student_keys``: encoded by the teacher MLP
  and distilled into by the student GRU.
- Shared keys (intersection) go to the actor directly (normed).
- Student-only keys (e.g. ``command_student``) feed the GRU but not the actor.

Architecture
------------
- Teacher encoder: MLP on privileged keys → ``_priv_feature``.
- Student encoder: GRU over all ``student_keys`` history → ``_priv_pred``.
- Shared actor: ``[shared_normed…, _priv_feat]``.
- Critic: all ``teacher_keys`` (privileged, not distilled features).

Stages
------
- **teacher**: PPO on the teacher path; distill student GRU → teacher features (BPTT).
- **student**: DAgger-style rollouts with the student GRU; distill only (no PPO).

Symmetry augmentation applies only during the teacher PPO update.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import einops
import torch.utils._pytree as pytree

from torchrl.data import Composite, TensorSpec, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Dict, List, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

from active_adaptation.learning.modules import (
    VecNorm,
    IndependentNormal,
    MLP,
    CatTensors,
)
from active_adaptation.learning.modules.rnn import set_recurrent_mode, recurrent_mode
from active_adaptation.learning.ppo.common import (
    hard_copy_,
    ppo_clipped_loss,
    resolve_clip_param,
    OBS_KEY,
    OBS_PRIV_KEY,
    ACTION_KEY,
    REWARD_KEY,
    TERM_KEY,
    DONE_KEY,
    GAE,
    make_batch,
    make_mlp,
    Actor,
    Critic,
)
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform

GRU_HIDDEN = 128


def _normed_key(key: str) -> str:
    return f"_{key}_normed"


@dataclass
class PPOTSCfg:
    _target_: str = "active_adaptation.learning.ppo.ppo_teacher_student.PPOTSCfg"
    
    name: str = "ppo_ts"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: Any = (0.2, 0.2)
    entropy_coef: float = 0.002
    pred_std: bool = False
    clamp_reward: bool = False

    actor_num_units: Tuple[int, ...] = (256, 256, 256)
    critic_num_units: Tuple[int, ...] = (512, 256, 256)
    # Hidden widths for the teacher privileged MLP; final dim is ``priv_feat_dim``.
    # Student uses a fixed FACET-style GRU (128-d hidden).
    encoder_num_units: Tuple[int, ...] = (128,)
    priv_feat_dim: int = 64
    distill_epochs: int = 2
    activation: str = "Mish"

    # Symmetry aug only used in teacher PPO updates (ignored in student stage).
    symaug: bool = True
    muon: bool = False  # Muon for teacher PPO (encoder + actor + critic)
    compile: bool = False

    stage: str = "teacher"  # "teacher" or "student"

    # ``privileged_keys = teacher_keys \\ student_keys`` (order preserved from teacher).
    # Examples:
    #   teacher=(policy, priv), student=(policy,)
    #   teacher=(command_teacher, policy), student=(command_student, policy)
    teacher_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY)
    student_keys: Tuple[str, ...] = (OBS_KEY,)
    # Filled in ``__post_init__`` (Hydra-safe tuple, order-preserving unique).
    in_keys: Tuple[str, ...] = ()

    def __post_init__(self):
        self.teacher_keys = tuple(self.teacher_keys)
        self.student_keys = tuple(self.student_keys)
        self.in_keys = tuple(set(self.teacher_keys + self.student_keys))

    def get_class(self):
        return PPOTeacherStudentPolicy


cs = ConfigStore.instance()
cs.store(name="ppo_teacher", node=PPOTSCfg(stage="teacher"), group="algo")
cs.store(
    name="ppo_student",
    node=PPOTSCfg(stage="student", symaug=False),
    group="algo",
)


class GRU(nn.Module):
    """Step / sequence GRU cell with episode resets (from ``ppo_facet``)."""

    def __init__(self, input_size, hidden_size, burn_in: bool = False) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if recurrent_mode():
            N, T = x.shape[:2]
            hx = hx[:, 0]
            output = []
            reset = 1.0 - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)
        else:
            hx = self.gru(x, hx * (1.0 - is_init.float()))
            output = self.ln(hx)
            return output, hx


class GRUModule(nn.Module):
    """MLP → GRU → residual out head (FACET adapt module)."""

    def __init__(self, dim: int, split=None):
        super().__init__()
        self.split = split
        self.mlp = make_mlp([GRU_HIDDEN, GRU_HIDDEN])
        self.gru = GRU(GRU_HIDDEN, hidden_size=GRU_HIDDEN)
        self.out = nn.LazyLinear(dim)

    def forward(self, x, is_init, hx):
        out1 = self.mlp(x)
        out2, hx = self.gru(out1, is_init, hx)
        out3 = self.out(out2 + out1)
        if self.split is None:
            out = (out3,)
        else:
            out = torch.split(out3, self.split, dim=-1)
        return out + (hx.contiguous(),)


class PPOTeacherStudentPolicy(TensorDictModuleBase):
    """Two-stage teacher–student PPO with key-driven GRU feature distillation."""

    def __init__(
        self,
        cfg: PPOTSCfg,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        obs_transforms: Optional[Dict[str, SymmetryTransform]] = None,
        act_transform: Optional[SymmetryTransform] = None,
    ):
        super().__init__()
        self.cfg = cfg
        if self.cfg.stage not in ("teacher", "student"):
            raise ValueError(f"Invalid stage: {self.cfg.stage!r}")
        self.device = device
        self.observation_spec = observation_spec

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = resolve_clip_param(self.cfg.clip_param)
        self.actor_loss_fn = ppo_clipped_loss
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.distill_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)

        student_set = set(self.cfg.student_keys)
        self.teacher_keys = tuple(self.cfg.teacher_keys)
        self.student_keys = tuple(self.cfg.student_keys)
        # Privileged: in teacher but not student (teacher order).
        self.privileged_keys = tuple(
            k for k in self.teacher_keys if k not in student_set
        )
        # Shared: in both (teacher order).
        self.shared_keys = tuple(k for k in self.teacher_keys if k in student_set)
        self.in_keys = tuple(self.cfg.in_keys)

        if not self.privileged_keys:
            raise ValueError(
                "teacher_keys must contain at least one privileged key not in "
                f"student_keys; got teacher={self.teacher_keys}, "
                f"student={self.student_keys}"
            )
        if not self.student_keys:
            raise ValueError("student_keys must be non-empty")

        fake_input = observation_spec.zero().to(self.device)
        for key in self.in_keys:
            if key not in observation_spec.keys(True, True):
                raise KeyError(
                    f"Expected observation key {key!r} in observation_spec; "
                    f"got {list(observation_spec.keys(True, True))}"
                )

        self.obs_transforms: Dict[str, SymmetryTransform] = {}
        if obs_transforms is not None:
            for key, transform in obs_transforms.items():
                self.obs_transforms[key] = transform.to(self.device)
        self.act_transform = (
            act_transform.to(self.device) if act_transform is not None else None
        )

        key_dims = {key: fake_input[key].shape[-1] for key in self.in_keys}
        self.action_dim = action_spec.shape[-1]
        priv_feat_dim = self.cfg.priv_feat_dim
        priv_inp_dim = sum(key_dims[k] for k in self.privileged_keys)
        shared_dim = sum(key_dims[k] for k in self.shared_keys)
        actor_inp_dim = shared_dim + priv_feat_dim
        critic_inp_dim = sum(key_dims[k] for k in self.teacher_keys)

        Activation = getattr(nn, self.cfg.activation)

        # VecNorm every observation group we consume.
        self.vecnorm = Seq(
            *[
                Mod(
                    VecNorm((key_dims[key],)),
                    [key],
                    [_normed_key(key)],
                )
                for key in self.in_keys
            ]
        ).to(self.device)

        # Teacher: MLP over privileged observations → distill target.
        teacher_normed = [_normed_key(k) for k in self.privileged_keys]
        self.encoder_teacher = Seq(
            CatTensors(teacher_normed, "_priv_enc_inp", sort=False),
            Mod(
                nn.Sequential(
                    MLP(
                        num_units=[priv_inp_dim, *self.cfg.encoder_num_units],
                        activation=Activation,
                        first_non_muon=True,
                    ),
                    nn.Linear(self.cfg.encoder_num_units[-1], priv_feat_dim),
                ),
                ["_priv_enc_inp"],
                ["_priv_feature"],
            ),
        ).to(self.device)

        # Student: GRU over all student-visible keys → predict privileged features.
        student_normed = [_normed_key(k) for k in self.student_keys]
        self.encoder_student = Seq(
            CatTensors(student_normed, "_student_enc_inp", sort=False),
            Mod(
                GRUModule(priv_feat_dim, split=None),
                ["_student_enc_inp", "is_init", "adapt_hx"],
                ["_priv_pred", ("next", "adapt_hx")],
            ),
        ).to(self.device)

        self.from_teacher = Mod(nn.Identity(), ["_priv_feature"], ["_priv"])
        self.from_student = Mod(nn.Identity(), ["_priv_pred"], ["_priv"])
        
        def make_actor() -> ProbabilisticActor:
            action_head = nn.Sequential(
                MLP(num_units=[actor_inp_dim, *self.cfg.actor_num_units], activation=Activation),
                Actor(self.action_dim, predict_std=self.cfg.pred_std),
            )
            return ProbabilisticActor(
                module=Seq(
                    CatTensors(
                        [_normed_key(k) for k in self.shared_keys] + ["_priv"],
                        "_actor_inp",
                        sort=False,
                    ),
                    Mod(action_head, ["_actor_inp"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(self.device)

        self.actor_teacher = make_actor()
        self.actor_student = make_actor()

        # Privileged critic: raw teacher keys (not distilled features).
        critic_normed = [_normed_key(k) for k in self.teacher_keys]
        critic_mlp = MLP(
            num_units=[critic_inp_dim, *self.cfg.critic_num_units],
            activation=Activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            CatTensors(
                critic_normed,
                "_critic_inp",
                del_keys=False,
                sort=False,
            ),
            Mod(critic_mlp, ["_critic_inp"], ["_critic_feature"]),
            Mod(Critic(1), ["_critic_feature"], ["state_value"]),
        ).to(self.device)

        self.training_keys = [
            "action_log_prob",
            "adv",
            "ret",
            "is_init",
            ACTION_KEY,
            *self.in_keys,
        ]

        # Lazy init / shape check (GRU needs is_init + adapt_hx).
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["adapt_hx"] = torch.zeros(fake_input.shape[0], GRU_HIDDEN)

        self.vecnorm(fake_input)
        self.actor_teacher(self.from_teacher(self.encoder_teacher(fake_input.copy())))
        self.actor_student(self.from_student(self.encoder_student(fake_input.copy())))
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.0)

        self.encoder_teacher.apply(init_)
        self.encoder_student.apply(init_)
        self.actor_teacher.apply(init_)
        self.critic.apply(init_)

        self.opt_ppo: Optional[torch.optim.Optimizer] = None
        self.opt_distill: Optional[torch.optim.Optimizer] = None
        self.update = self._update

    @classmethod
    def from_env(cls, cfg: PPOTSCfg, env: _EnvBase, device: str):
        observation_spec = env.observation_spec
        action_spec = env.action_spec
        reward_spec = env.reward_spec
        obs_transforms = {
            key: env.observation_groups[key].symmetry_transform()
            for key in cfg.in_keys
        }
        act_transform = env.action_manager.symmetry_transform()
        return cls(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            device=device,
            obs_transforms=obs_transforms,
            act_transform=act_transform,
        )

    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        spec = Unbounded((num_envs, GRU_HIDDEN), device=self.device)
        return TensorDictPrimer({"adapt_hx": spec}, reset_key="done", expand_specs=False)

    def on_stage_start(self, _stage: str, _env: _EnvBase):
        # One stage per run for now; ``cfg.stage`` selects teacher vs student.
        if self.cfg.stage == "teacher":
            if self.cfg.muon:
                self.opt_ppo = MuonAdamWWrapper(
                    [self.encoder_teacher, self.actor_teacher, self.critic],
                    lr=self.cfg.lr,
                    weight_decay=0.01,
                )
            else:
                self.opt_ppo = torch.optim.AdamW(
                    [
                        {"params": self.encoder_teacher.parameters()},
                        {"params": self.actor_teacher.parameters()},
                        {"params": self.critic.parameters()},
                    ],
                    lr=self.cfg.lr,
                    weight_decay=0.01,
                )
            # Student GRU distill stays on AdamW (recurrent / LazyLinear mix).
            self.opt_distill = torch.optim.AdamW(
                [{"params": self.encoder_student.parameters()}],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
        elif self.cfg.stage == "student":
            # Shared actor / critic / teacher encoder already carry teacher weights
            # from the checkpoint. Only the student GRU is updated (DAgger).
            self.opt_ppo = None
            self.opt_distill = torch.optim.AdamW(
                [{"params": self.encoder_student.parameters()}],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
            self.critic.requires_grad_(False)
            # copy to initialize student actor
            hard_copy_(self.actor_teacher, self.actor_student)
        else:
            raise ValueError(f"Invalid stage: {self.cfg.stage}")

        self.update = self._update
        if self.cfg.compile:
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        modules = [self.vecnorm if self.cfg.stage == "teacher" else VecNorm.freeze()(self.vecnorm)]
        if self.cfg.stage == "teacher":
            modules += [self.encoder_teacher, self.from_teacher, self.actor_teacher]
        elif self.cfg.stage == "student":
            # Collect with student GRU (DAgger); carries adapt_hx via primer.
            modules += [self.encoder_student, self.from_student, self.actor_student]
        else:
            raise ValueError(f"Invalid stage: {self.cfg.stage}")
        if critic:
            modules.append(self.critic)
        policy = Seq(*modules)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats").to(self.device, non_blocking=True)
        info = {}

        if self.cfg.stage == "teacher":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_distillation(tensordict.copy()))
        elif self.cfg.stage == "student":
            info.update(self.train_distillation(tensordict.copy()))
        else:
            raise ValueError(f"Invalid stage: {self.cfg.stage}")
        return dict(sorted(info.items()))

    def train_policy(self, tensordict: TensorDict):
        self.encoder_teacher.requires_grad_(True)
        self.actor_teacher.requires_grad_(True)
        self.critic.requires_grad_(True)

        infos = []
        with ScopedTimer("compute_advantage"):
            self.compute_advantage(
                tensordict, self.critic, "adv", "ret", self.cfg.clamp_reward
            )
            adv = tensordict["adv"]
            adv_mean = adv.mean()
            adv_std = adv.std()
            tensordict["adv"] = (adv - adv_mean) / adv_std.clamp_min(1e-7)

        td = tensordict.select(*self.training_keys)
        for _epoch in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(td, self.cfg.num_minibatches):
                if self.cfg.symaug:
                    minibatch = self._augment_symmetry(minibatch)
                infos.append(self.update(minibatch))

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/adv_mean"] = adv_mean.item()
        infos["critic/adv_std"] = adv_std.item()
        reward_aggregated = tensordict["next", "reward_aggregated"]
        infos["critic/neg_rew_ratio"] = (reward_aggregated <= 0.0).float().mean().item()
        return infos

    @set_recurrent_mode(True)
    def train_distillation(self, tensordict: TensorDict):
        """MSE: student GRU features ≈ teacher privileged features (BPTT)."""
        self.encoder_teacher.requires_grad_(False)
        self.actor_teacher.requires_grad_(False)

        infos = []
        self.vecnorm(tensordict)
        with torch.no_grad():
            self.encoder_teacher(tensordict)
            self.actor_teacher(self.from_teacher(tensordict))
            teacher_action = tensordict.pop(ACTION_KEY)
            tensordict["action_teacher"] = teacher_action

        for _epoch in range(self.cfg.distill_epochs):
            for minibatch in make_batch(
                tensordict, self.cfg.num_minibatches, self.cfg.train_every
            ):
                self.encoder_student(minibatch)
                self.actor_teacher(self.from_student(minibatch))

                valid = (~minibatch["is_init"]).float()
                valid_cnt = valid.sum().clamp_min(1.0)
                
                feat_loss = F.mse_loss(
                    minibatch["_priv_pred"], minibatch["_priv_feature"], reduction="none"
                )
                feat_loss = (feat_loss.mean(dim=-1, keepdim=True) * valid).sum() / valid_cnt
                
                action_loss = F.mse_loss(
                    minibatch[ACTION_KEY], minibatch["action_teacher"], reduction="none"
                )
                action_loss = (action_loss.mean(dim=-1, keepdim=True) * valid).sum() / valid_cnt
                
                loss = feat_loss + action_loss * 0.0
                self.opt_distill.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.encoder_student.parameters(), self.max_grad_norm
                )
                self.opt_distill.step()
                infos.append(
                    {
                        "distill/feat_loss": feat_loss.detach(),
                        "distill/action_loss": action_loss.detach(),
                        "distill/grad_norm": grad_norm.detach()
                        if torch.is_tensor(grad_norm)
                        else torch.tensor(grad_norm),
                    }
                )

        return pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)

    @torch.no_grad()
    def compute_value(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
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
        rewards = rewards * (1.0 - self.gae.gamma)

        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)
        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _augment_symmetry(self, tensordict: TensorDict) -> TensorDict:
        symmetry = tensordict.empty()
        if self.act_transform is not None:
            symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        else:
            symmetry[ACTION_KEY] = tensordict[ACTION_KEY]
        for key in self.in_keys:
            transform = self.obs_transforms.get(key)
            if transform is not None:
                symmetry[key] = transform(tensordict[key])
            else:
                symmetry[key] = tensordict[key]
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        return torch.cat([tensordict, symmetry])

    @ScopedTimer("ppo_update")
    def _update(self, tensordict: TensorDict):
        assert self.cfg.stage == "teacher"
        bsize = tensordict.shape[0] // 2 if self.cfg.symaug else tensordict.shape[0]

        valid = (~tensordict["is_init"]).float()
        valid_cnt = valid.sum().clamp_min(1.0)
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]

        self.vecnorm(tensordict)
        self.encoder_teacher(tensordict)
        self.actor_teacher(self.from_teacher(tensordict))
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        ret = tensordict["ret"]
        log_ratio = (log_probs - log_probs_data).reshape_as(adv)
        ratio = torch.exp(log_ratio)
        eps_neg, eps_pos = self.clip_param
        ratio_det = ratio.detach()
        clamped_pos = ratio_det > 1.0 + eps_pos
        clamped_neg = ratio_det < 1.0 - eps_neg

        policy_loss = self.actor_loss_fn(ratio, adv, self.clip_param)
        entropy_loss = -self.entropy_coef * entropy

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(ret, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        self.opt_ppo.zero_grad(set_to_none=True)
        loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(
            self.encoder_teacher.parameters(), self.max_grad_norm
        )
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor_teacher.parameters(), self.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.opt_ppo.step()

        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, ret) / ret.var().clamp_min(1e-7)
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            if self.cfg.symaug and self.act_transform is not None:
                symmetry_loss = F.mse_loss(
                    dist.mean[bsize:], self.act_transform(dist.mean[:bsize])
                )
            else:
                symmetry_loss = ret.new_zeros(())

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/encoder_grad_norm": encoder_grad_norm,
            "actor/clamp_pos": clamped_pos.float().mean(),
            "actor/clamp_neg": clamped_neg.float().mean(),
            "actor/approx_kl": approx_kl,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        for key, transform in self.obs_transforms.items():
            state_dict[f"obs_transform/{key}"] = transform.state_dict()
        if self.act_transform is not None:
            state_dict["act_transform"] = self.act_transform.state_dict()
        state_dict["last_stage"] = self.cfg.stage
        state_dict["teacher_keys"] = self.teacher_keys
        state_dict["student_keys"] = self.student_keys
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
