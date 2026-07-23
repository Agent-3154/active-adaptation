from __future__ import annotations

import copy
import einops
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from torchrl.data import Composite

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
from active_adaptation.utils.profiling import ScopedTimer

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import ScalarCritic
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from torchrl.objectives import hold_out_net
from tensordict.nn.probabilistic import interaction_type, InteractionType

cs = ConfigStore.instance()

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
class QC1Config:
    """
    QC1 config.
    """
    _target_: str = "active_adaptation.learning.offpolicy.qc1.QC1"
    name: str = "qc1"

    # general setting
    debug: bool = False
    use_prior_online: bool = True
    soft_bound: float = 7 * torch.pi

    vecnorm: bool = True
    clamp_reward: bool = False
    lr: float = 3e-4
    weight_decay: float = 0.0
    muon: bool = True
    actor_nums: int = 32
    gamma: float = 0.99
    horizon_length: int = 3
    critic_hidden_dims: tuple[int] = (512, 512)
    actor_hidden_dims: tuple[int]  = (512, 512)
    # offline stage
    prior_data_path: str | None = '/home/cv/zjx/active-adaptation/scripts/rollout/G1LocoFlat-sac/2026-07-08-20-32-07/rollout_1000_4096.pt'
    bootstrap_observation_keys: Tuple[str, ...] = ("prev_noise", "rho")
    batch_size: int = 256
    tau_critic: float = 5e-3
    max_grad_norm: float = 1.0
    # online stage
    buffer_size: int = 10_000_000
    utd_ratio: int = 1
    warm_up_steps: int = 100
    # FlashSAC-style: scale rewards by running discounted-return variance.
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8
    # SAC-style actor: entropy bonus coefficient (fixed; no alpha tuner).
    entropy_bonus: float = 1.0


    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)


cs.store(name="qc1", node=QC1Config, group="algo")


class CriticTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        activation: type[nn.Module] = nn.SiLU,
        norm: Literal["rms"] | None = "rms",
        condition_dim: int = 0,
    ):
        super().__init__()
        if len(set(hidden_dims)) != 1:
            raise ValueError("CriticTrunk currently expects equal hidden dimensions.")
        hidden_dim = hidden_dims[0]

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
                for _ in hidden_dims
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

def _init_sac_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)

import math
class NormalActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        std_max: float = 1.0,
        std_min: float = 0.001,
        action_init: Literal["zeros", "orthogonal"] = "zeros",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.in_layer = nn.Linear(obs_dim, 384)
        self.in_layer.weight._non_muon = True
        self.trunk = nn.Sequential(
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            ConditionalBlock(hidden_dim=384, condition_dim=0, norm="rms"),
            nn.RMSNorm(384),
        )
        self.action = nn.Linear(384, act_dim * 2)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_sac_linear)
        
        if action_init == "orthogonal":
            self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
        elif action_init == "zeros":
            # zero-init following FastSAC
            nn.init.constant_(self.action.weight, 0.0) # zero-init the weight
            nn.init.constant_(self.action.bias, 0.0) # zero-init the bias
        else:
            raise ValueError(f"Invalid action_init: {action_init}")

        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.log_std_max = math.log(std_max)
        self.log_std_min = math.log(std_min)

    def forward(self, obs: torch.Tensor, ):
        feat = self.trunk(self.in_layer(obs))
        mean, raw = self.action(feat).chunk(2, dim=-1)
        # log_std = self.log_std_max - F.softplus(raw)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw))
        return mean, torch.exp(log_std)



def TwinScalarCritic(
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple[int, ...],
    activation: type[nn.Module] = nn.SiLU,
):
    critic_input_dim = obs_dim + act_dim
    module = SimpleDoubleCritic(
        fn=lambda: CriticTrunk(
            input_dim=critic_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )
    )
    return ScalarCritic(module)


class QC1(TensorDictModuleBase):
    def __init__(
        self,
        cfg: QC1Config,
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

        self.msr = MultiStepReturn(self.cfg.gamma, self.cfg.horizon_length).to(device)

        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            dev = device if isinstance(device, torch.device) else torch.device(device)
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                load_rms=False,
                device=dev,
                epsilon=float(self.cfg.reward_norm_epsilon),
            )

        critic_hidden_dims = tuple(self.cfg.critic_hidden_dims)
        actor_hidden_dims = tuple(self.cfg.actor_hidden_dims)

        self.Q = TwinScalarCritic(
            obs_dim=self.obs_dim,
            act_dim=self.full_action_dim,
            hidden_dims=critic_hidden_dims,
        ).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()

        self.actor = NormalActor(self.obs_dim, self.full_action_dim).to(device)
        self.DistClass = IndependentNormal

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

        self.offline_steps = 0
        self.critic_steps = 0
        self.online_steps = 0
        self.policy_offline_steps = 0


    def on_stage_start(self, stage: str, env: _EnvBase):
        if stage == "offline":
            if self.cfg.prior_data_path is None:
                raise ValueError("Offline training requires prior data but got None.")

            num_envs = env.fake_tensordict().shape[0]
            max_time_steps = max(self.cfg.buffer_size // num_envs, 1)

            self.rb = ReplayBuffer.from_rollout(
                self.cfg.prior_data_path,
                max_size=max_time_steps,
                fake_bootstrap=True,
                observation_keys=list(self.cfg.bootstrap_observation_keys),
            )
            print("offline Replay buffer:")
            print(self.rb)
        elif stage == "online":
            if not self.cfg.use_prior_online:
                self.rb._current_size = 0
            print("online Replay buffer:")
            print(self.rb)
        else:
            raise ValueError(f"Stage {stage} is invalid.")
        

    @classmethod
    def from_env(cls, cfg: QC1Config, env, device: torch.device):
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


    @ScopedTimer("update_flow_policy")
    def update_policy(self):
        batch = self.rb.sample(
            batch_size=self.cfg.batch_size,
            steps=self.cfg.horizon_length,
            next_obs=True
        ).to(self.device)
        batch = batch.select(*self.training_keys, inplace=True, strict=False)
        self.preproc(batch)
        self.preproc(batch["next"])
        self.policy_offline_steps += 1
        infos: dict = { "training/policy_step": self.policy_offline_steps }
        infos.update(self.train_actor(batch))
        return dict(sorted(infos.items()))


    @ScopedTimer("update_critic")
    def update_critic(self):
        batch = self.rb.sample(
            batch_size=self.cfg.batch_size,
            steps=self.cfg.horizon_length,
            next_obs=True
        ).to(self.device)
        batch = batch.select(*self.training_keys, inplace=True, strict=False)
        self.preproc(batch)
        self.preproc(batch["next"])
        self.critic_steps += 1
        infos: dict = { "training/critic_steps": self.critic_steps }
        infos.update(self.train_critic(batch))
        return dict(sorted(infos.items()))


    def update_rb(self, tensordict: TensorDictBase):
        self.online_steps += 1
        td = tensordict.exclude(("next", "stats"), "collector")

        # Per-step reward statistics.
        r = td[REWARD_KEY]
        if isinstance(r, TensorDict):
            r = torch.cat(list(r.values()), dim=-1)
        r = r.sum(dim=-1, keepdim=True)

        if self.reward_normalizer is not None and not self.cfg.debug:
            truncated = td.get(("next", "truncated"))
            if truncated is None:
                truncated = torch.zeros_like(td[TERM_KEY])
            self.reward_normalizer.update_reward_stats(
                reward=r,
                terminated=td[TERM_KEY],
                truncated=truncated,
            )

        self.rb.push(td)

        infos: dict = {
            "rb_size": len(self.rb),
            "training/online_steps": self.online_steps,
            "training/neg_rew_ratio": (r <= 0).float().mean().item(),
            "reward/step_mean": r.mean().item(),
            "reward/step_std": r.std().item(),
        }

        return infos
    
    def update_online(self, update_policy: bool = True):
        if self.online_steps < self.cfg.warm_up_steps:
            return {}

        infos: dict = {}
        for _ in range(self.cfg.utd_ratio):
            batch = self.rb.sample(
                batch_size=self.cfg.batch_size,
                steps=self.cfg.horizon_length,
                next_obs=True,
            ).to(self.device)

            batch = batch.select(*self.training_keys, inplace=False, strict=False)
            self.preproc(batch)
            self.preproc(batch["next"])

            infos.update(self.train_actor(batch))
            if update_policy:
                infos.update(self.train_critic(batch))

        return dict(sorted(infos.items()))

    @ScopedTimer("train_critic")
    def train_critic(self, batch: TensorDict):
        """Update critic on a preprocessed batch (``batch.select`` + ``preproc`` already applied)."""
        self.Q.train()

        def _collate_reward(reward: torch.Tensor | TensorDict) -> torch.Tensor:
            if isinstance(reward, TensorDict):
                reward = torch.cat(list(reward.values()), dim=-1)
            reward = reward.sum(dim=-1, keepdim=True)
            if self.cfg.clamp_reward:
                reward = reward.clamp_min(0.0)
            return reward

        reward = _collate_reward(batch[REWARD_KEY])
        reward_raw = reward.clone()

        if self.cfg.debug:
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            # scale by effective horizon (SAC fallback when normalizer is off)
            reward = reward * (1.0 - self.cfg.gamma)

        obs = batch["_input_normed"][0]
        act_n = batch[ACTION_KEY]
        env_disc_ms = batch.get(("next", "discount"))
        if env_disc_ms is not None:
            env_disc_ms = env_disc_ms[: self.msr.n_steps]

        act_n, next_obs, reward, discount, terminated = self.msr(
            actions=act_n,
            next_observations=batch["next", "_input_normed"],
            rewards=reward[: self.msr.n_steps],
            terminated=batch[TERM_KEY],
            done=batch[DONE_KEY],
            env_discount=env_disc_ms,
        )

        with ScopedTimer("compute_target"):
            q_target = self._compute_target(next_obs, reward, discount)

        act_concated = act_n.flatten(start_dim=1)
        pred = self.Q(obs, act_concated)
        q_loss = self.Q.compute_loss(pred, q_target).mean()

        self.opt_Q.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.Q.parameters(), max_norm=self.cfg.max_grad_norm,
        )
        self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, self.cfg.tau_critic)

        infos: dict = {
            "critic/q_loss": q_loss.detach().item(),
            "critic/grad_norm": critic_grad_norm.item(),
            "critic/q_mean": pred.detach().mean().item(),
            "critic/q_max": pred.detach().max().item(),
            "critic/q_min": pred.detach().min().item(),
            "critic/target_mean": q_target.detach().mean().item(),
            "critic/target_max": q_target.detach().max().item(),
            "reward/mean": reward_raw.mean().item(),
            "reward/std": reward_raw.std().item(),
            "critic/a_scale": act_n.detach().mean().item(),
            "critic/sa_scale": self.sample_actions(obs).detach().mean().item()
        }

        with torch.no_grad():
            q_values = self.Q.get_values(obs, act_concated).mean(dim=-1)
            q_log = q_values
            if self.reward_normalizer is not None:
                q_log = self.reward_normalizer.denormalize_return_values(q_log)
            infos["critic/q_value"] = q_log.mean().item()
            infos["critic/q_std"] = q_log.std(dim=-1).mean().item()

            if terminated.any():
                term_mask = terminated.reshape(q_values.shape[0])
                infos["critic/q_value_terminated"] = q_log[term_mask].mean().item()
                infos["critic/q_loss_terminated"] = (
                    self.Q.compute_loss(pred[term_mask], q_target[term_mask]).mean().item()
                )

        return infos


    @ScopedTimer("train_actor")
    def train_actor(self, batch: TensorDict):
        """SAC-style actor update: sample action chunks from NormalActor,
        maximize Q(s, a) + entropy_bonus * H[a] with a soft-bound regularizer.
        """
        self.actor.train()

        obs = batch["_input_normed"][0]
        batch_size = obs.shape[0]

        loc, scale = self.actor(obs)
        dist = self.DistClass(loc, scale)
        # Reparameterized samples: [K, B, full_action_dim]
        action_samples = dist.rsample((self.cfg.actor_nums,))
        # log_prob over the full chunk: [K, B]
        log_prob = dist.log_prob(action_samples)
        entropy_est = -log_prob.mean(dim=0)  # [B]

        with hold_out_net(self.Q):
            # Q expects [B, K, full_action_dim] -> [B, K, 2]; mean over twin -> [B, K]
            action_samples_bk = action_samples.permute(1, 0, 2).contiguous()
            q = self.Q.get_values(obs, action_samples_bk).mean(dim=-1)
        q_mean = q.mean(dim=1)  # [B] — average over K samples
        policy_term = -q_mean  # maximize Q

        soft_term = 0.01 * ((loc / self.cfg.soft_bound) ** 6).sum(-1)  # [B]

        actor_loss = (
            policy_term
            + self.cfg.entropy_bonus * (-entropy_est)
            + soft_term
        ).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.cfg.max_grad_norm,
        )
        self.opt_actor.step()

        infos: dict = {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/q_mean": q_mean.detach().mean().item(),
            "actor/entropy": entropy_est.detach().mean().item(),
            "actor/soft_term": soft_term.detach().mean().item(),
            "actor/mean_loc": loc.detach().abs().mean().item(),
            "actor/mean_scale": scale.detach().mean().item(),
        }
        return infos


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
    

    @ScopedTimer("sample_actions")
    def sample_actions(self, next_obs: torch.Tensor):
        """Sample a single action chunk from NormalActor for target computation."""
        loc, scale = self.actor(next_obs)
        dist = self.DistClass(loc, scale)
        return dist.sample()  # [B, full_action_dim]


    def get_rollout_policy(self, mode: str = "eval", critic: bool = False) -> TensorDictModuleBase:
        """Return a :class:`QCRolloutPolicy` for eval / rollout."""
        return QCRolloutPolicy(
            vecnorm_obs=self.vecnorm_obs,
            actor=self.actor,
            DistClass=self.DistClass,
            horizon_length=self.cfg.horizon_length,
            action_dim=self.action_dim,
            obs_key=OBS_KEY,
            cmd_key=CMD_KEY,
        )

    def state_dict(self) -> "OrderedDict[str, Any]":
        """Include reward-normalizer running stats alongside nn.Module params."""
        sd: "OrderedDict[str, Any]" = OrderedDict(super().state_dict())
        if self.reward_normalizer is not None:
            sd["reward_normalizer"] = self.reward_normalizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: "OrderedDict[str, Any]", strict: bool = True):
        rn_state = state_dict.pop("reward_normalizer", None)
        super().load_state_dict(state_dict, strict=strict)
        if self.reward_normalizer is not None and rn_state is not None:
            self.reward_normalizer.load_state_dict(rn_state)


class QCRolloutPolicy(TensorDictModuleBase):
    """Rollout policy for QC1 with action chunking.

    Samples a full H-step action chunk from NormalActor (mean in MODE,
    stochastic in RANDOM) and caches it. Subsequent calls return the next
    pre-computed action from the chunk, re-planning only after H steps or
    when an environment resets.
    """

    def __init__(
        self,
        vecnorm_obs: nn.Module,
        actor: nn.Module,
        DistClass: type,
        horizon_length: int,
        action_dim: int,
        obs_key: str = "policy",
        cmd_key: str = "command",
    ):
        super().__init__()
        self.vecnorm_obs = vecnorm_obs
        self.actor = actor
        self.DistClass = DistClass
        self.horizon_length = horizon_length
        self.action_dim = action_dim
        self.obs_key = obs_key
        self.cmd_key = cmd_key

        self.in_keys = [obs_key]
        self.out_keys = [ACTION_KEY]

        self._chunk_step: torch.Tensor | None = None
        self._action_chunk: torch.Tensor | None = None

    @torch.no_grad()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict[self.obs_key]
        if self.cmd_key in tensordict.keys(True, True):
            obs = torch.cat([tensordict[self.cmd_key], obs], dim=-1)
        obs = self.vecnorm_obs(obs)

        batch_size = obs.shape[0]
        device = obs.device

        if (
            self._chunk_step is None
            or self._chunk_step.shape[0] != batch_size
        ):
            self._chunk_step = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            self._action_chunk = torch.zeros(
                batch_size, self.horizon_length, self.action_dim, device=device
            )

        is_init = tensordict.get("is_init")
        need_replan = self._chunk_step >= self.horizon_length
        if is_init is not None:
            need_replan = need_replan | is_init.squeeze(-1).bool()

        if need_replan.any():
            replan_idx = need_replan.nonzero(as_tuple=True)[0]
            obs_replan = obs[replan_idx]
            loc, scale = self.actor(obs_replan)
            if interaction_type() == InteractionType.MODE:
                new_actions = loc
            else:
                dist = self.DistClass(loc, scale)
                new_actions = dist.sample()
            new_chunk = new_actions.view(
                -1, self.horizon_length, self.action_dim,
            )
            if need_replan.all():
                self._action_chunk = new_chunk
            else:
                self._action_chunk[replan_idx] = new_chunk
            self._chunk_step[replan_idx] = 0

        idx = self._chunk_step.clamp(max=self.horizon_length - 1)
        action = self._action_chunk[torch.arange(batch_size, device=device), idx]
        self._chunk_step += 1

        tensordict[ACTION_KEY] = action
        return tensordict
