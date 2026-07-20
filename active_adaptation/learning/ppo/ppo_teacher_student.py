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
Teacher–Student PPO (two stages, no GRU).

Architecture
------------
- Separate command encoders for ``command_teacher`` / ``command_student``.
- Shared policy VecNorm + actor MLP + action head (input: ``[_cmd_feat, _obs_normed]``).
- Critic always sees privileged teacher inputs (``command_teacher`` + ``policy``).

Stages
------
- **teacher**: PPO on the teacher path; distill student cmd encoder → teacher cmd features.
- **student**: DAgger-style rollouts with the student path; distill only (no PPO on student).

Symmetry augmentation applies only during the teacher PPO update.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from typing import Tuple, Optional, Any, TYPE_CHECKING
from collections import OrderedDict

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
    resolve_clip_param,
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
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform

CMD_TEACHER_KEY = "command_teacher"
CMD_STUDENT_KEY = "command_student"


@dataclass
class PPOTSCfg:
    _target_: str = (
        "active_adaptation.learning.ppo.ppo_teacher_student.PPOTeacherStudentPolicy"
    )
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
    # Hidden widths for cmd encoders; final output dim is ``cmd_feat_dim``.
    encoder_num_units: Tuple[int, ...] = (128,)
    cmd_feat_dim: int = 64
    distill_epochs: int = 2
    activation: str = "Mish"

    # Symmetry aug only used in teacher PPO updates (ignored in student stage).
    symaug: bool = True
    compile: bool = False

    stage: str = "teacher"  # "teacher" or "student"

    teacher_keys: Tuple[str, ...] = (CMD_TEACHER_KEY, OBS_KEY)
    student_keys: Tuple[str, ...] = (CMD_STUDENT_KEY, OBS_KEY)
    in_keys: Tuple[str, ...] = (CMD_TEACHER_KEY, CMD_STUDENT_KEY, OBS_KEY)


cs = ConfigStore.instance()
cs.store(name="ppo_teacher", node=PPOTSCfg(stage="teacher"), group="algo")
cs.store(
    name="ppo_student",
    node=PPOTSCfg(stage="student", symaug=False),
    group="algo",
)


class PPOTeacherStudentPolicy(TensorDictModuleBase):
    """Two-stage teacher–student PPO with shared actor and feature distillation."""

    def __init__(
        self,
        cfg: PPOTSCfg,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        cmd_teacher_transform: Optional[SymmetryTransform] = None,
        cmd_student_transform: Optional[SymmetryTransform] = None,
        obs_transform: Optional[SymmetryTransform] = None,
        act_transform: Optional[SymmetryTransform] = None,
    ):
        super().__init__()
        self.cfg = PPOTSCfg(**cfg) if not isinstance(cfg, PPOTSCfg) else cfg
        if self.cfg.stage not in ("teacher", "student"):
            raise ValueError(f"Invalid stage: {self.cfg.stage!r}")
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = resolve_clip_param(self.cfg.clip_param)
        self.actor_loss_fn = ppo_clipped_loss
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.distill_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero().to(self.device)
        for key in self.cfg.in_keys:
            if key not in observation_spec.keys(True, True):
                raise KeyError(
                    f"Expected observation key {key!r} in observation_spec; "
                    f"got {list(observation_spec.keys(True, True))}"
                )

        self.cmd_teacher_transform = (
            cmd_teacher_transform.to(self.device) if cmd_teacher_transform is not None else None
        )
        self.cmd_student_transform = (
            cmd_student_transform.to(self.device) if cmd_student_transform is not None else None
        )
        self.obs_transform = obs_transform.to(self.device) if obs_transform is not None else None
        self.act_transform = act_transform.to(self.device) if act_transform is not None else None

        cmd_teacher_dim = fake_input[CMD_TEACHER_KEY].shape[-1]
        cmd_student_dim = fake_input[CMD_STUDENT_KEY].shape[-1]
        obs_dim = fake_input[OBS_KEY].shape[-1]
        self.action_dim = action_spec.shape[-1]
        cmd_feat_dim = self.cfg.cmd_feat_dim
        actor_inp_dim = cmd_feat_dim + obs_dim
        critic_inp_dim = cmd_teacher_dim + obs_dim

        Activation = getattr(nn, self.cfg.activation)

        self.vecnorm = Seq(
            Mod(
                VecNorm((cmd_teacher_dim,)),
                [CMD_TEACHER_KEY],
                ["_cmd_teacher_normed"],
            ),
            Mod(
                VecNorm((cmd_student_dim,)),
                [CMD_STUDENT_KEY],
                ["_cmd_student_normed"],
            ),
            Mod(VecNorm((obs_dim,)), [OBS_KEY], ["_obs_normed"]),
        ).to(self.device)

        # Teacher / student cmd encoders → same feature width for MSE distill.
        self.encoder_teacher = Mod(
            MLP(
                num_units=[cmd_teacher_dim, *self.cfg.encoder_num_units, cmd_feat_dim],
                activation=Activation,
                first_non_muon=True,
            ),
            ["_cmd_teacher_normed"],
            ["_cmd_feature"],
        ).to(self.device)
        self.encoder_student = Mod(
            MLP(
                num_units=[cmd_student_dim, *self.cfg.encoder_num_units, cmd_feat_dim],
                activation=Activation,
                first_non_muon=True,
            ),
            ["_cmd_student_normed"],
            ["_cmd_pred"],
        ).to(self.device)

        # Remap encoder outputs onto the shared actor input key.
        self._teacher_to_actor = Mod(nn.Identity(), ["_cmd_feature"], ["_cmd_feat"])
        self._student_to_actor = Mod(nn.Identity(), ["_cmd_pred"], ["_cmd_feat"])

        actor_mlp = MLP(
            num_units=[actor_inp_dim, *self.cfg.actor_num_units],
            activation=Activation,
            first_non_muon=True,
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=Seq(
                CatTensors(
                    ["_cmd_feat", "_obs_normed"],
                    "_actor_inp",
                    del_keys=False,
                    sort=False,
                ),
                Mod(actor_mlp, ["_actor_inp"], ["_actor_feature"]),
                Mod(
                    Actor(self.action_dim, predict_std=self.cfg.pred_std),
                    ["_actor_feature"],
                    ["loc", "scale"],
                ),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        # Privileged critic: teacher command + policy (not distilled features).
        critic_mlp = MLP(
            num_units=[critic_inp_dim, *self.cfg.critic_num_units],
            activation=Activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            CatTensors(
                ["_cmd_teacher_normed", "_obs_normed"],
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
            CMD_TEACHER_KEY,
            CMD_STUDENT_KEY,
            OBS_KEY,
            ACTION_KEY,
        ]

        # Lazy init / shape check.
        self.vecnorm(fake_input)
        self.encoder_teacher(fake_input)
        self._teacher_to_actor(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        self.encoder_student(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.0)

        self.encoder_teacher.apply(init_)
        self.encoder_student.apply(init_)
        self.actor.apply(init_)
        self.critic.apply(init_)

        self.opt_ppo: Optional[torch.optim.Optimizer] = None
        self.opt_distill: Optional[torch.optim.Optimizer] = None
        self.update = self._update

    @classmethod
    def from_env(cls, cfg: PPOTSCfg, env: _EnvBase, device: str):
        observation_spec = env.observation_spec
        action_spec = env.action_spec
        reward_spec = env.reward_spec
        cmd_teacher_transform = env.observation_groups[CMD_TEACHER_KEY].symmetry_transform()
        cmd_student_transform = env.observation_groups[CMD_STUDENT_KEY].symmetry_transform()
        obs_transform = env.observation_groups[OBS_KEY].symmetry_transform()
        act_transform = env.action_manager.symmetry_transform()
        return cls(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            device=device,
            cmd_teacher_transform=cmd_teacher_transform,
            cmd_student_transform=cmd_student_transform,
            obs_transform=obs_transform,
            act_transform=act_transform,
        )

    def on_stage_start(self, _stage: str, _env: _EnvBase):
        # One stage per run for now; ``cfg.stage`` selects teacher vs student.
        if self.cfg.stage == "teacher":
            self.opt_ppo = torch.optim.AdamW(
                [
                    {"params": self.encoder_teacher.parameters()},
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
            self.opt_distill = torch.optim.AdamW(
                [{"params": self.encoder_student.parameters()}],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
        elif self.cfg.stage == "student":
            # Shared actor / critic / teacher encoder already carry teacher weights
            # from the checkpoint. Only the student encoder is updated (DAgger).
            self.opt_ppo = None
            self.opt_distill = torch.optim.AdamW(
                [{"params": self.encoder_student.parameters()}],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
            self.encoder_teacher.requires_grad_(False)
            self.actor.requires_grad_(False)
            self.critic.requires_grad_(False)
        else:
            raise ValueError(f"Invalid stage: {self.cfg.stage}")

        self.update = self._update
        if self.cfg.compile:
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        modules = [self.vecnorm]
        if self.cfg.stage == "teacher":
            modules += [self.encoder_teacher, self._teacher_to_actor, self.actor]
        elif self.cfg.stage == "student":
            # Collect with student modules (DAgger).
            modules += [self.encoder_student, self._student_to_actor, self.actor]
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

    def train_distillation(self, tensordict: TensorDict):
        """MSE: student cmd features ≈ teacher cmd features (stop-grad teacher)."""
        infos = []
        self.vecnorm(tensordict)
        with torch.no_grad():
            self.encoder_teacher(tensordict)

        for _epoch in range(self.cfg.distill_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                self.encoder_student(minibatch)
                valid = (~minibatch["is_init"]).float()
                valid_cnt = valid.sum().clamp_min(1.0)
                feat_loss = self.distill_loss_fn(
                    minibatch["_cmd_pred"], minibatch["_cmd_feature"]
                )
                feat_loss = (feat_loss.mean(dim=-1, keepdim=True) * valid).sum() / valid_cnt

                self.opt_distill.zero_grad(set_to_none=True)
                feat_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.encoder_student.parameters(), self.max_grad_norm
                )
                self.opt_distill.step()
                infos.append(
                    {
                        "distill/feat_loss": feat_loss.detach(),
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
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[CMD_TEACHER_KEY] = self.cmd_teacher_transform(tensordict[CMD_TEACHER_KEY])
        if self.cmd_student_transform is not None:
            symmetry[CMD_STUDENT_KEY] = self.cmd_student_transform(
                tensordict[CMD_STUDENT_KEY]
            )
        else:
            symmetry[CMD_STUDENT_KEY] = tensordict[CMD_STUDENT_KEY]
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        return torch.cat([tensordict, symmetry])

    @ScopedTimer("ppo_update")
    def _update(self, tensordict: TensorDict):
        assert self.cfg.stage == "teacher"
        bsize = tensordict.shape[0] // 2 if self.cfg.symaug else tensordict.shape[0]

        self.vecnorm(tensordict)
        self.encoder_teacher(tensordict)
        self._teacher_to_actor(tensordict)

        valid = (~tensordict["is_init"]).float()
        valid_cnt = valid.sum().clamp_min(1.0)
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]

        self.actor(tensordict)
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
        actor_grad_norm = nn.utils.clip_grad_norm_(
            list(self.encoder_teacher.parameters()) + list(self.actor.parameters()),
            self.max_grad_norm,
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
        if self.cmd_teacher_transform is not None:
            state_dict["cmd_teacher_transform"] = self.cmd_teacher_transform.state_dict()
        if self.cmd_student_transform is not None:
            state_dict["cmd_student_transform"] = self.cmd_student_transform.state_dict()
        if self.obs_transform is not None:
            state_dict["obs_transform"] = self.obs_transform.state_dict()
        if self.act_transform is not None:
            state_dict["act_transform"] = self.act_transform.state_dict()
        state_dict["last_stage"] = self.cfg.stage
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
