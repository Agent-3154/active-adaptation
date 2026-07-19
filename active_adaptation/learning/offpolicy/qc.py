from __future__ import annotations

import copy
import einops
from dataclasses import dataclass
from typing import Callable, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from torchrl.data import Composite

from active_adaptation.learning.modules import VecNorm, ConditionalBlock, CatTensors
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
from active_adaptation.learning.utils.opt import MuonAdamWWrapper

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
class QCConfig:
    """
    QC config.
    """
    _target_: str = "active_adaptation.learning.offpolicy.qc.QC"
    name: str = "qc"

    # general setting
    vecnorm: bool = True
    clamp_reward: bool = False
    lr: float = 3e-4
    weight_decay: float = 0.0
    muon: bool = True
    q_agg: str = "mean"
    actor_type: str = "best-of-n"
    actor_nums: int = 32
    gamma: float = 0.99
    horizon_length: int = 5
    flow_steps: int = 10
    compile_flow: bool = False
    critic_hidden_dims: tuple[int] = (512, 512, 512, 512)
    actor_hidden_dims: tuple[int]  = (512, 512, 512, 512)
    # offline stage
    prior_data_path: str | None = '/home/cv/zjx/active-adaptation/scripts/rollout/G1LocoFlat-sac/2026-07-08-20-32-07/rollout_1000_4096.pt'
    bootstrap_observation_keys: Tuple[str, ...] = ("prev_noise", "rho")
    batch_size: int = 256
    tau_critic: float = 0.005
    max_grad_norm: float = 1.0
    # online stage
    buffer_size: int = 10_000_000
    utd_ratio: int = 1
    warm_up_steps: int = 0


    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)


cs.store(name="qc", node=QCConfig, group="algo")


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
            nn.init.zeros_(self.action.weight)
            nn.init.zeros_(self.action.bias)
        else:
            raise ValueError(f"Invalid action_init: {action_init}")
        
    def forward(self, observation, actions, time):
        input = torch.concat([observation, actions, time], dim=-1)
        action = self.trunk(self.in_layer(input))
        return self.action(action)


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

        self.actor = ActorVectorField(
            obs_dim=self.obs_dim,
            action_dim=self.full_action_dim,
            hidden_num=len(actor_hidden_dims),
            hidden_dim=actor_hidden_dims[0],
        ).to(device)

        # Pre-register timestep values for flow integration (avoids per-iteration allocations).
        ts = torch.arange(cfg.flow_steps, dtype=torch.float32, device=device) / cfg.flow_steps
        self.register_buffer("_flow_ts", ts)

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

        if self.cfg.compile_flow:
            self.compute_flow_actions = torch.compile(
                self.compute_flow_actions, mode="reduce-overhead",
            )

        self.global_step = 0


    def on_stage_start(self, stage: str, env: _EnvBase):
        if stage == "offline":
            if self.cfg.prior_data_path is None:
                raise ValueError("Offline training requires prior data but got None.")
            self.prior_rb = ReplayBuffer.from_rollout(
                self.cfg.prior_data_path,
                fake_bootstrap=True,
                observation_keys=list(self.cfg.bootstrap_observation_keys),
            )
            print("Prior data buffer:")
            print(self.prior_rb)
        elif stage == "online":
            # Create replay buffer and pre-fill with offline data so that
            # online transitions gradually overwrite older entries via FIFO,
            # matching the reference implementation's data mixing strategy.
            fake_rb = (
                env.fake_tensordict()
                .exclude(("next", "stats"), "collector")
            )
            num_envs = fake_rb.shape[0]
            max_time_steps = max(self.cfg.buffer_size // num_envs, 1)

            observation_keys = set(env.observation_spec.keys(True, True))
            observation_keys = observation_keys - set(self.cfg.bootstrap_observation_keys)
            self.rb = ReplayBuffer.from_fake(
                max_time_steps,
                fake_rb,
                fake_bootstrap=True,
                observation_keys=list(observation_keys),
            )

            prior_size = len(self.prior_rb)
            take = min(prior_size, max_time_steps)
            if take > 0:
                self.rb._td[:take] = self.prior_rb._td[:take].to(self.rb._td.device)
                self.rb._current_size = take
                self.rb._ptr = take % max_time_steps

            print("Online replay buffer:")
            print(self.rb)
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


    @ScopedTimer("step_offline")
    def step_offline(self):
        """Sample a batch, preprocess once, then update critic and actor."""
        batch = self.prior_rb.sample(
            batch_size=self.cfg.batch_size,
            steps=self.cfg.horizon_length,
            next_obs=True,
        ).to(self.device)

        batch = batch.select(*self.training_keys, inplace=False, strict=False)
        self.preproc(batch)
        self.preproc(batch["next"])

        self.global_step += 1

        infos: dict = {"training/global_step": self.global_step}
        infos.update(self.train_critic(batch))
        infos.update(self.train_actor(batch))
        return dict(sorted(infos.items()))


    @ScopedTimer("step")
    def step(self, tensordict: TensorDictBase) -> dict:
        """Online training step: push transition, sample batch, update."""
        self.global_step += 1

        td = tensordict.exclude(("next", "stats"), "collector")
        self.rb.push(td)

        # Per-step reward statistics.
        r = td[REWARD_KEY]
        if isinstance(r, TensorDict):
            r = torch.cat(list(r.values()), dim=-1)
        r = r.sum(dim=-1)

        infos: dict = {
            "rb_size": len(self.rb),
            "training/global_step": self.global_step,
            "training/neg_rew_ratio": (r <= 0).float().mean().item(),
            "reward/step_mean": r.mean().item(),
            "reward/step_std": r.std().item(),
        }

        if self.global_step < self.cfg.warm_up_steps:
            return infos

        for _ in range(self.cfg.utd_ratio):
            batch = self.rb.sample(
                batch_size=self.cfg.batch_size,
                steps=self.cfg.horizon_length,
                next_obs=True,
            ).to(self.device)

            batch = batch.select(*self.training_keys, inplace=False, strict=False)
            self.preproc(batch)
            self.preproc(batch["next"])

            infos.update(self.train_critic(batch))
            infos.update(self.train_actor(batch))

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
        }

        with torch.no_grad():
            q_values = self.Q.get_values(obs, act_concated).mean(dim=-1)
            infos["critic/q_value"] = q_values.mean().item()
            infos["critic/q_std"] = q_values.std(dim=-1).mean().item()

            if terminated.any():
                term_mask = terminated.reshape(q_values.shape[0])
                infos["critic/q_value_terminated"] = q_values[term_mask].mean().item()
                infos["critic/q_loss_terminated"] = (
                    self.Q.compute_loss(pred[term_mask], q_target[term_mask]).mean().item()
                )

        return infos


    @ScopedTimer("train_actor")
    def train_actor(self, batch: TensorDict):
        """Update actor on a preprocessed batch (``batch.select`` + ``preproc`` already applied)."""
        self.actor.train()

        obs = batch["_input_normed"][0]
        batch_actions = einops.rearrange(
            batch[ACTION_KEY], "t b a -> b (t a)"
        ).contiguous()
        batch_size, action_dim = batch_actions.shape

        x0 = torch.rand(batch_actions.shape, device=batch_actions.device)
        x1 = batch_actions
        t = torch.rand((batch_size, 1), device=batch_actions.device)
        xt = (1 - t) * x0 + t * x1
        v = x1 - x0
        pred_v = self.actor(obs, xt, t)

        per_element_loss = (pred_v - v) ** 2

        # Per-timestep valid mask: action[t] is valid only when no episode
        # boundary occurs before t within the chunk (matching reference).
        _done = batch[DONE_KEY].squeeze(-1)  # [H, B]
        prev_done = torch.cat([torch.zeros_like(_done[:1]), _done[:-1]], dim=0)
        valid_t = 1.0 - prev_done.cumsum(dim=0).clamp(max=1.0)  # [H, B]

        per_element_loss = per_element_loss.reshape(
            batch_size, self.cfg.horizon_length, self.action_dim,
        )
        actor_loss = (per_element_loss * valid_t.permute(1, 0).unsqueeze(-1)).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=self.cfg.max_grad_norm,
        )
        self.opt_actor.step()

        infos: dict = {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/pred_v_norm": pred_v.detach().norm(dim=-1).mean().item(),
            "actor/valid_frac": valid_t.mean().item(),
        }

        # Per-timestep flow loss breakdown.
        with torch.no_grad():
            loss_by_step = per_element_loss.mean(dim=(0, 2))  # [H]
            for step_idx in range(min(self.cfg.horizon_length, loss_by_step.shape[0])):
                infos[f"actor/flow_loss_t{step_idx}"] = loss_by_step[step_idx].item()

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
        batch_size = next_obs.shape[0]
        device = next_obs.device

        if self.cfg.actor_type == "best-of-n":
            next_obss = next_obs.repeat_interleave(
                self.cfg.actor_nums, dim=0
            )

            noise = torch.rand(
                batch_size * self.cfg.actor_nums,
                self.full_action_dim,
                device=device
            )

            with ScopedTimer("compute_flow_actions"):
                actions = self.compute_flow_actions(next_obss, noise)
            with ScopedTimer("critic_forward"):
                q_values = self.Q_target(next_obss, actions)

            if self.cfg.q_agg == "mean":
                q_values = q_values.mean(dim=-1)
            elif self.cfg.q_agg == "min":
                q_values = q_values.min(dim=-1).values
            else:
                raise NotImplementedError(f"Unknown q_agg: {self.cfg.q_agg}")
            
            q_values = q_values.view(batch_size, self.cfg.actor_nums)
            actions = actions.view(
                batch_size,self.cfg.actor_nums,self.full_action_dim
            )

            indices = q_values.argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=device)

            return actions[batch_indices, indices]
        
        elif self.cfg.actor_type == "distll-ddpg":
            raise NotImplementedError


    def get_rollout_policy(self, mode: str = "eval", critic: bool = False) -> TensorDictModuleBase:
        """Return a :class:`QCRolloutPolicy` for eval / rollout."""
        return QCRolloutPolicy(
            vecnorm_obs=self.vecnorm_obs,
            sample_fn=self.sample_actions,
            horizon_length=self.cfg.horizon_length,
            action_dim=self.action_dim,
            obs_key=OBS_KEY,
            cmd_key=CMD_KEY,
        )

    def compute_flow_actions(self, next_obs, noises: torch.Tensor):
        actions = noises
        for i in range(self.cfg.flow_steps):
            time = self._flow_ts[i].expand(actions.shape[0], 1)
            actions += self.actor(next_obs, actions, time) / self.cfg.flow_steps
        return actions


class QCRolloutPolicy(TensorDictModuleBase):
    """Rollout policy for QC with action chunking.

    Samples a full H-step action chunk via flow matching and caches it.
    Subsequent calls return the next pre-computed action from the chunk,
    re-planning only after H steps or when an environment resets.
    """

    def __init__(
        self,
        vecnorm_obs: nn.Module,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon_length: int,
        action_dim: int,
        obs_key: str = "policy",
        cmd_key: str = "command",
    ):
        super().__init__()
        self.vecnorm_obs = vecnorm_obs
        self.sample_fn = sample_fn
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
            if need_replan.all():
                new_chunk = self.sample_fn(obs)
            else:
                obs_replan = obs[replan_idx]
                new_chunk = self._action_chunk.clone()
                new_chunk[replan_idx] = self.sample_fn(obs_replan).view(
                    replan_idx.shape[0], self.horizon_length, self.action_dim
                )
            self._action_chunk = new_chunk.view(
                batch_size, self.horizon_length, self.action_dim
            )
            self._chunk_step[replan_idx] = 0

        idx = self._chunk_step.clamp(max=self.horizon_length - 1)
        action = self._action_chunk[torch.arange(batch_size, device=device), idx]
        self._chunk_step += 1

        tensordict[ACTION_KEY] = action
        return tensordict
