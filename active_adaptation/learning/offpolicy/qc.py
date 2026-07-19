from __future__ import annotations

import copy
import math
import einops
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal, Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from active_adaptation.learning.modules import VecNorm, IndependentNormal, ConditionalBlock, CatTensors
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
from active_adaptation.learning.offpolicy.distributional import (
    C51Critic,
    ScalarCritic,
)
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.distribution import FasterTransformedDistribution
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import (
    check_parameters,
    unwrap_ddp,
    wrap_ddp,
)
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.symmetry import SymmetryTransform
from tensordict.nn.probabilistic import interaction_type, InteractionType

cs = ConfigStore.instance()

OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY   = "action"
REWARD_KEY   = ("next", "reward")
TERM_KEY     = ("next", "terminated")
DONE_KEY     = ("next", "done")
OBS_KEY      = "policy"
CMD_KEY      = "command"

def _init_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)

@dataclass
class QCConfig:
    """
    QC config.
    """
    _target_: str = "active_adaptation.learning.offpolicy.qc.QC"
    name: str = "qc"

    # general setting
    debug: bool = True
    vecnorm: bool = True
    clamp_reward: bool = True
    normalize_reward: bool = False
    muon: bool = True
    q_agg: str = "min"
    actor_type: str = "best-of-n"
    actor_nums: int = 4
    gamma: float = 0.99
    horizon_length: int = 4
    flow_steps: int = 10
    critic_hidden_dims: tuple[int] = (512, 512, 512, 512)
    actor_hidden_dims: tuple[int]  = (512, 512, 512, 512)
    # offline stage
    prior_data_path: str | None = '/home/cv/zjx/active-adaptation/scripts/rollout/G1LocoFlat-sac/2026-07-08-20-32-07/rollout_1000_4096.pt'
    batch_size: int = 2
    tau_critic: float = 0.1


    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)


cs.store(name="qc", node=QCConfig, group="algo")


class CriticTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_num: int = 2,
        hidden_dim: int = 512,
        activation: type[nn.Module] = nn.SiLU,
        norm: Literal["rms"] | None = "rms",
        condition_dim: int = 0,
    ):
        super().__init__()

        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.in_layer.weight._non_muon = True
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.out_layer.weight._non_muon = True

        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    norm=norm,
                    condition_dim=condition_dim,
                )
                for _ in range(hidden_num)
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        x = self.out_layer(x)
        return x


class SimpleDoubleCritic(nn.Module):
    def __init__(self, fn: Callable[..., nn.Module]):
        super().__init__()
        self.critic_1 = fn()
        self.critic_2 = fn()

    def forward(self, obs, act):
        if act.dim() == 2:
            input = torch.cat([obs, act], dim=-1)
            q1 = self.critic_1(input)
            q2 = self.critic_2(input)
            return torch.cat([q1, q2], dim=-1)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "batch obs -> (batch k) obs", k=k)
            act_flat = einops.rearrange(act, "batch k act_dim -> (batch k) act_dim")
            qs = self.forward(obs_flat, act_flat)
            return einops.rearrange(qs, "(batch k) fused -> batch k fused", batch=b, k=k)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")


class ActorVectorField(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_num: int = 4,
        hidden_dim: int = 512,
        action_init: Literal['orthogonal', 'zeros'] = 'orthogonal'
    ):
        super().__init__()
        self.in_layer = nn.Linear(obs_dim + action_dim + 1, hidden_dim)
        self.in_layer.weight._non_muon = True
        self.trunk = nn.Sequential()

        for _ in range(hidden_num):
            self.trunk.append(
                ConditionalBlock(
                    hidden_dim=hidden_dim, condition_dim=0, norm="rms"
                )
            )
        self.trunk.append(nn.RMSNorm(hidden_dim))
        self.action = nn.Linear(hidden_dim, action_dim)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_linear)

        if action_init == "orthogonal":
            self.action.apply(
                lambda m: _init_linear(m, gain=0.01)
            )
        elif action_init == "zeros":
            nn.init.zeros_(self.action.weight, 0.0)
            nn.init.zeros_(self.action.bias, 0.0)
        else:
            raise ValueError(f"Invalid action_init: {action_init}")
        
    def forward(self, observation, actions, time):
        input = torch.concat([observation, actions, time], dim=-1)
        action = self.trunk(self.in_layer(input))
        return self.action(action)


def TwinScalarCritic(
    obs_dim: int,
    act_dim: int,
    activation: type[nn.Module] = nn.SiLU,
):
    critic_input_dim = obs_dim + act_dim
    module = SimpleDoubleCritic(
        fn=lambda: CriticTrunk(
            input_dim=critic_input_dim,
            hidden_dim=512,
            output_dim=1,
            activation=activation,
        )
    )
    return ScalarCritic(module)


class QC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: QCConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: Composite,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_spec      = action_spec
        self.reward_spec      = reward_spec
        self.device           = device

        self.action_dim       = action_spec.shape[-1]
        self.full_action_dim  = self.action_dim * cfg.horizon_length

        self._distributed = False
        
        fake = observation_spec.zero()
        preproc = []
        if CMD_KEY in observation_spec.keys(True, True):
            train_keys = (
                CMD_KEY, OBS_KEY, ("next", OBS_KEY), ("next", CMD_KEY), ACTION_KEY,
                REWARD_KEY, TERM_KEY, DONE_KEY, ("next", "discount"), "is_init",
            )
            obs_dim = fake[OBS_KEY].shape[-1] + fake[CMD_KEY].shape[-1]
            preproc.append(CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False))
        else:
            train_keys = (
                OBS_KEY, ("next", OBS_KEY), ACTION_KEY,
                REWARD_KEY, TERM_KEY, DONE_KEY, ("next", "discount"), "is_init",
            )
            obs_dim = fake[OBS_KEY].shape[-1]
            preproc.append(Mod(nn.Identity(), [OBS_KEY], ["_input"]))
        self.training_keys = train_keys

        self.obs_dim = obs_dim
        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim).to(device)
        else:
            self.vecnorm_obs = nn.Identity()

        preproc.append(Mod(self.vecnorm_obs, ["_input"], ["_input_normed"]))
        self.preproc = Seq(*preproc).to(device)

        self.has_symmetry = False
        self.reward_normalizer = None
        self.msr = MultiStepReturn(self.cfg.gamma, self.cfg.horizon_length).to(device)
        self.compute_target = self._compute_target
        # Construct Networks
        self.Q = TwinScalarCritic(
            obs_dim=self.obs_dim, 
            act_dim=self.full_action_dim
        ).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()

        self.actor = ActorVectorField(
            obs_dim=self.obs_dim,
            action_dim=self.full_action_dim
        )

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


    def on_stage_start(self, stage: str, env: _EnvBase):
        if stage == "offline":
            if self.cfg.prior_data_path is None:
                raise ValueError(f"Offline training need prior data but got None")
            else:
                observation_keys = set(env.observation_spec.keys(True, True))
                observation_keys = observation_keys = {"prev_noise", "rho"}
                self.prior_rb = ReplayBuffer.from_rollout(
                    self.cfg.prior_data_path,
                    fake_bootstrap=True,
                    observation_keys=list(observation_keys)
                )
                print("Prior data buffer:")
                print(self.prior_rb)
        elif stage == "online":
            raise NotImplementedError
        else:
            raise ValueError(f"Stage {stage} is invalid.")
        

    @classmethod
    def from_env(cls, cfg: QCConfig, env, device: torch.device):
        """
        Create QC agent from env.
        """
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device
        )


    def step_offline(self):
        "Sample and update once."
        infos = {}

        batch = self.prior_rb.sample(
            batch_size=self.cfg.batch_size,
            steps=self.cfg.horizon_length,
            next_obs=True
        ).to(self.device)

        if self.cfg.debug:
            print(batch.batch_size)
            print(batch.keys(True, True))
            print(batch)

        info = self.train_critic(batch, diagnostic=True)

        infos.update(info)

        return dict(sorted(infos.items()))

    
    def train_critic(self, batch: TensorDict, diagnostic: bool = True):
        self.Q.train()
        batch = batch.select(*self.training_keys, inplace=True, strict=False)
        
        def collate_reward(reward: torch.Tensor | TensorDict) -> torch.Tensor:
            if isinstance(reward, TensorDict):
                reward = torch.cat(list(reward.values()), dim=-1)
            reward = reward.sum(dim=-1, keepdim=True)
            if self.cfg.clamp_reward:
                reward = reward.clamp_min(0.0)
            return reward
        
        reward = collate_reward(batch[REWARD_KEY])
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            reward = reward * (1.0 - self.cfg.gamma)

        # maybe concat or normalize the obs.
        self.preproc(batch)
        self.preproc(batch["next"])

        assert self.cfg.horizon_length > 1
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

        if self.cfg.debug:
            print("=========================")
            print(f'{act_n.shape}\n{next_obs.shape}\n{reward.shape}\n{discount.shape}\n{terminated.shape}')
            print("=========================")

        with ScopedTimer("compute_target"):
            q_target = self.compute_target(next_obs, reward, discount)
            print(q_target)

        act_concated = torch.concat(act_n, dim=1)
        pred = self.Q(obs, act_concated)
        per_sample_loss = self.Q.compute_loss(pred, q_target)
        valid = (1.0 - is_init.float()).reshape_as(per_sample_loss)
        denom = valid.sum().clamp_min(1e-8)
        q_loss = (per_sample_loss * valid).sum() / denom

        self.
        return {}

    @torch.no_grad()
    def _compute_target(self, next_obs: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        next_action = self.sample_actions(next_obs)

        q_target = self.Q_target.compute_target(
            next_obs,
            next_action,
            reward,
            discount,
        )
        return q_target
    
    def sample_actions(self, next_obs: torch.Tensor):
        ex_actions = torch.rand(self.cfg.batch_size * self.cfg.actor_nums, self.action_spec.shape[-1] * self.cfg.horizon_length)

        if self.cfg.actor_type == "best-of-n":
            assert ex_actions.dim() == 2
            assert next_obs.dim() == 2
            noises = torch.rand_like(ex_actions)
            next_obss = next_obs.repeat(self.cfg.actor_nums, 1)
            actions = self.compute_flow_actions(next_obss, noises)

            q_values = self.Q_target(next_obss, actions)
            
            if self.cfg.debug:
                print(q_values.shape) #[batch_size * actor_num, 2]

            if self.cfg.q_agg == "mean":
                q_values = torch.mean(q_values, dim=-1)
            elif self.cfg.q_agg == "min":
                q_values, _ = torch.min(q_values, dim=-1)
            # [batch_size * actor_num,]
            q_values = torch.reshape(q_values, (self.cfg.batch_size, self.cfg.actor_nums))
            print(q_values.shape)
            indices = torch.argmax(q_values, dim=-1)
            actions = torch.reshape(actions, (self.cfg.batch_size, self.cfg.actor_nums, actions.shape[-1]))
            actions = actions[:, indices, :].squeeze()
            return actions
        elif self.cfg.actor_type == "distll-ddpg":
            raise NotImplementedError


    def compute_flow_actions(self, next_obs, noises: torch.Tensor):
        actions = noises
        for i in range(self.cfg.flow_steps):
            time = torch.full_like(actions[..., 0], i / self.cfg.flow_steps).to(next_obs.device)
            time = time[..., None]
            assert time.dim() == 2
            assert next_obs.dim() == 2
            assert actions.dim() == 2
            vels = self.actor(next_obs, actions, time)
            actions += vels / self.cfg.flow_steps

        return actions