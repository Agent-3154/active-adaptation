from __future__ import annotations


import math
import copy
import einops
import torch
from torch import nn
from typing import Literal, Callable, Optional, Tuple, TYPE_CHECKING

import active_adaptation as aa
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.modules import ConditionalBlock, CatTensors, VecNorm
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.dormancy import DormancyTracker
# from active_adaptation.learning.offpolicy.noise import ParallelPinkNoiseProcess
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    OBS_KEY,
    CMD_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)
from tensordict import TensorDict
from dataclasses import dataclass
from collections import OrderedDict
from torchrl.data import Composite
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential
from .distributional import ScalarCritic, C51Critic
from torchrl.objectives import hold_out_net
from hydra.core.config_store import ConfigStore
from tensordict.nn.probabilistic import interaction_type, InteractionType


def _init_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)

cs = ConfigStore.instance()

@dataclass
class TD3Config:
    """
    TD3 Config.
    """
    _target_: str = "active_adaptation.learning.offpolicy.td3.TD3Config"
    name: str = "td3"
    delayed: int = 2
    train_every: int = 4
    soft_bound: float = math.pi
    buffer_size: int = 1500
    warm_up_steps: int = 200
    lr: float = 5e-4
    # Network init config
    act_init: str = "zeros"
    act_orthogonal_gain: float | None = 0.01
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper` (see ``ppo_symaug``).
    muon: bool = True
    weight_decay: float = 0.02
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    # architecture
    distributional: bool = False
    v_min: float = -1.0 # used if no reward normalizer
    v_max: float = 9.0 # used if no reward normalizer
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    sym_aug: bool = False # not supported.
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    noise_type: str = "white" # white or pink(https://github.com/martius-lab/pink-noise-rl)
    rollout_action_noise: float = 0.10
    target_action_noise: float = 0.10

    tau_Q: float = 0.1
    tau_actor: float = 0.1
    max_grad_norm: float = 1.0

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); GradScaler for critic, V head, standalone train_v, and actor (alpha stays fp32).
    use_amp: bool = False # not supported
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8

    # path to prior data for RLPD
    prior_data: str | None = None
    prior_data_ratio: float = 0.4
    # "binary": stop mixing prior data once success rate exceeds ``prior_data_stop_success``.
    # "linear": linearly reduce prior data ratio from ``prior_data_schedule_lower`` to
    # ``prior_data_schedule_upper`` success rate.
    prior_data_schedule: str = "linear" # linear or binary
    prior_data_stop_success: float | None = 0.2
    prior_data_schedule_lower: float = 0.1
    prior_data_schedule_upper: float = 0.2

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, ACTION_KEY)

    def get_class(self):
        return TD3

cs.store(name="td3", node=TD3Config, group="algo")


class DormancyScope(nn.Module):
    def __init__(self, actor: nn.Module, q_online: nn.Module):
        super().__init__()
        self.actor = actor
        self.Q = q_online


class CriticTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,

        hidden_num: int = 2,
        hidden_dim: int = 512,

        # for conditional block
        activation: type[nn.Module] = nn.SiLU,
        norm: Literal['rms'] | None = "rms",
        condition_dim: int = 0
    ):
        super().__init__()

        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.in_layer.weight._non_muon = True
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.out_layer.weight._non_muon = True
        
        self.trunk = nn.Sequential()
        for _ in range(hidden_num):
            self.trunk.append(
                ConditionalBlock(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    norm=norm,
                    condition_dim=condition_dim
                )
            )
        
        self.norm = nn.RMSNorm(hidden_dim)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        x = self.in_layer(x)
        x = self.trunk(x)
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
        elif act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "batch obs -> (batch k) obs", k=k)
            act_flat = einops.rearrange(act, "batch k act_dim -> (batch k) act_dim")
            qs = self.forward(obs_flat, act_flat)
            return einops.rearrange(qs, "(batch k) fused -> batch k fused", batch=b, k=k)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hid_dim: int = 384,
        hid_num: int = 2,

        act_init: Literal['zeros', 'orthogonal'] = "zeros",
        act_orthogonal_gain: float | None = None
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.in_layer = nn.Linear(obs_dim, hid_dim)
        self.in_layer.weight._non_muon = True

        self.trunk = nn.Sequential(
            *[
                ConditionalBlock(hid_dim) for _ in range(hid_num)
            ]
        )
        self.trunk.append(nn.RMSNorm(hid_dim))

        self.action = nn.Linear(hid_dim, act_dim)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_linear)

        if act_init == "zeros":
            nn.init.zeros_(self.action.weight)
            nn.init.zeros_(self.action.bias)
        elif act_init == "orthogonal":
            assert act_orthogonal_gain > 0, "orthogonal gain must > 0 while using orthogonal init for action."
            self.action.apply(lambda m: _init_linear(m, gain=0.01))
        else:
            raise ValueError(f"Invalid action_init: {act_init}")
        
    def forward(self, obs):
        feat = self.trunk(self.in_layer(obs))
        return self.action(feat)
    

def Critic(   
    obs_dim: int,
    act_dim: int,
    activation: type[nn.Module] = nn.SiLU
):
    critic_input_dim = obs_dim + act_dim
    module = SimpleDoubleCritic(
        fn=lambda: CriticTrunk(
            input_dim=critic_input_dim,
            activation=activation
        )
    )
    return ScalarCritic(module)


def DistC51Critic(
    obs_dim: int,
    act_dim: int,
    num_atoms: int,
    v_max: float,
    v_min:float,
    activation: type[nn.Module] = nn.SiLU
):
    critic_input_dim = obs_dim + act_dim
    module = SimpleDoubleCritic(
        fn=lambda: CriticTrunk(
            input_dim=critic_input_dim,
            output_dim=num_atoms,
            activation=activation
        )
    )
    return C51Critic(
        module=module,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms
    )


class TD3(TensorDictModuleBase):
    """
    Twin Delayed DDPG.
    """
    def __init__(
        self,
        cfg: TD3Config,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: Composite,
        device,
        *,
        obs_transform: Optional[SymmetryTransform],
        act_transform: Optional[SymmetryTransform]
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec

        self.obs_transform = obs_transform.to(device) if obs_transform is not None else None
        self.act_transform = act_transform.to(device) if act_transform is not None else None

        self.world_size = aa.get_world_size()

        # What are not supported:
        # 1. distributional training.
        assert not aa.is_distributed(), "This TD3 Implementation does not support distributed training."
        # 2. data symmetry augmentation.
        assert not cfg.sym_aug, "This TD3 implementation does not support symmetry augmentation."
        # 3. AMP.
        assert not self.cfg.use_amp, "TD3 do not support AMP."

        fake_obs = observation_spec.zero()
        preproc  = []
        
        # ====================================================================================
        if CMD_KEY in observation_spec.keys(True, True):
            self.train_keys = (
                CMD_KEY, OBS_KEY, ("next", OBS_KEY), ("next", CMD_KEY), ACTION_KEY,
                REWARD_KEY, TERM_KEY, DONE_KEY, ("next", "discount"), "is_init",
            )

            obs_dim = fake_obs[OBS_KEY].shape[-1] + fake_obs[CMD_KEY].shape[-1]
            preproc.append(
                CatTensors([CMD_KEY, OBS_KEY], "_input", del_keys=False, sort=False)
            )
        else:
            self.train_keys = (
                OBS_KEY, ("next", OBS_KEY), ACTION_KEY,
                REWARD_KEY, TERM_KEY, DONE_KEY, ("next", "discount"), "is_init",
            )
            obs_dim = fake_obs[OBS_KEY].shape[-1]
            preproc.append(
                TensorDictModule(nn.Identity(), [OBS_KEY], "_input")
            )
        act_dim = action_spec.shape[-1]

        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim, decay=1.0).to(device)
        else:
            self.vecnorm_obs = nn.Identity()
        preproc.append(TensorDictModule(self.vecnorm_obs, ["_input"], ["_input_normed"]))
        self.preproc = TensorDictSequential(*preproc).to(device)
        # ====================================================================================
        
        if not self.cfg.distributional:
            self.Q = Critic(obs_dim, act_dim).to(device)
        else:
            if self.cfg.normalize_reward:
                # Std-normalized returns are O(1); fixed atom support (not task-tuned).
                v_min, v_max, num_atoms = -0.5, 5.0, 101
            else:
                v_min, v_max = self.cfg.v_min, self.cfg.v_max
                num_atoms    = int((v_max - v_min) / 0.05) + 1
            self.Q = DistC51Critic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_atoms=num_atoms,
                v_max=v_max,
                v_min=v_min
            ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.Q_target.requires_grad_(False)

        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_init=self.cfg.act_init,
            act_orthogonal_gain=self.cfg.act_orthogonal_gain
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_target.requires_grad_(False)

        if self.cfg.muon:
            self.opt_actor = MuonAdamWWrapper(
                [self.actor],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
            self.opt_Q = MuonAdamWWrapper(
                [self.Q],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        else:
            self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            self.opt_Q     = torch.optim.AdamW(self.Q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        self.global_step = 0 # env steps
        self.critic_step = 0 # critic update steps
        self.actor_step  = 0

        self.msr = (
            MultiStepReturn(
                self.cfg.gamma, self.cfg.n_steps
            ).to(device) if self.cfg.n_steps > 1 else None
        )

        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                load_rms=False,
                device=self.device if isinstance(self.device, torch.device) else torch.device(self.device),
                epsilon=float(self.cfg.reward_norm_epsilon)
            )

        scope = DormancyScope(self.actor, self.Q)
        self._dormancy_tracker = DormancyTracker(scope)

        self.compute_target = torch.compile(
            self._compute_target,
            mode="reduce-overhead"
        )

    def _flush_dormancy(self, infos: dict):
        dormancy = self._dormancy_tracker.compute_dormancy(0.02)
        for module_name, value in dormancy.items():
            infos[f"dormancy/{module_name}"] = value
        self._dormancy_tracker.reset()

    @classmethod
    def from_env(cls, cfg: TD3Config, env: _EnvBase, device: torch.device):
        if cfg.sym_aug:
            obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform()
            act_transform = env.action_manager.symmetry_transform()
            if CMD_KEY in env.observation_spec.keys(True, True):
                cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transform()
                obs_transform = SymmetryTransform.cat([cmd_transform, obs_transform])
        else:
            obs_transform = None
            act_transform = None
        
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
            obs_transform=obs_transform,
            act_transform=act_transform
        )
    
    def get_rollout_policy(self, mode: str = "train", critic: bool = False) -> TensorDictModuleBase:
        return TD3RolloutPolicy(
            preproc=self.preproc if mode == "train" else VecNorm.freeze()(self.preproc),
            actor=self.actor,
            Q=self.Q,
            noise_type=self.cfg.noise_type,
            noise_scale=self.cfg.rollout_action_noise,
            act_dim=self.action_spec.shape[-1],
            device=self.device,
            seq_len=1000,
            reward_normalizer=self.reward_normalizer,
            critic=critic
        )

    def on_stage_start(self, stage: str, env: _EnvBase):
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
        )

        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys = observation_keys - {"prev_noise", "rho"}
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size, 
            fake_rb,
            fake_bootstrap=True,
            observation_keys=list(observation_keys)
        )
        print("Primary Buffer:")
        print(self.rb)

        if self.cfg.prior_data is not None:
            self.rb_prior = ReplayBuffer.from_rollout(
                self.cfg.prior_data,
                fake_bootstrap=True,
                observation_keys=list(observation_keys)
            )
            print("Prior data buffer:")
            print(self.rb_prior)
        else:
            self.rb_prior = None

        # used in prior data schedule
        self._success_rate: float | None = None
        
        self.enable_actor = True # TODO: Shoud this be removed?

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    def _prior_data_ratio(self) -> float:
        if self.rb_prior is None:
            return 0.0

        base = self.cfg.prior_data_ratio
        sr = self._success_rate

        if self.cfg.prior_data_schedule == "binary":
            stop_at = self.cfg.prior_data_stop_success
            if stop_at is None or sr is None:
                return base
            return base if sr <= stop_at else 0.0

        lower = self.cfg.prior_data_schedule_lower
        upper = self.cfg.prior_data_schedule_upper
        if sr is None or upper <= lower:
            return base
        if sr <= lower:
            return base
        if sr >= upper:
            return 0.0
        t = (sr - lower) / (upper - lower)
        return base * (1.0 - t)

    @property
    def use_prior_data(self) -> bool:
        return self._prior_data_ratio() > 0.0


    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict, *, success_rate: float | None = None):
        if success_rate is not None:
            self._success_rate = success_rate
        self.global_step += self.cfg.train_every

        # Data processing
        # =============================================================================
        td = tensordict.exclude(("next", "stats"), "collector")

        reward = td[REWARD_KEY]

        if isinstance(reward, TensorDict):
            reward = torch.cat(list(reward.values()), dim=-1)

        if self.cfg.debug:
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
            neg_rew_ratio = 0.0
        else:
            reward = reward.sum(-1, keepdim=True)
            neg_rew_ratio = (reward <= 0.).float().mean().item()
        
        bs = td.batch_size
        for ti in range(int(bs[1])):
            sub = td[:, ti]
            if self.reward_normalizer is not None:
                self.reward_normalizer.update_reward_stats(
                    reward=reward[:, ti],
                    terminated=sub[TERM_KEY],
                    truncated=sub["next", "truncated"]
                )
            self.rb.push(sub)

        infos: dict = {"rb_size": len(self.rb), "critic/neg_rew_ratio": neg_rew_ratio}
        if self.rb_prior is not None:
            infos["prior_data/active"] = float(self.use_prior_data)
            infos["prior_data/ratio"] = self._prior_data_ratio()
            if self._success_rate is not None:
                infos["prior_data/success_rate"] = self._success_rate
        if self.global_step < self.cfg.warm_up_steps:
            self._flush_dormancy(infos)
            return infos
        # =============================================================================

        iters = self.cfg.train_every * self.cfg.utd_ratio
        for i in range(iters):
            self.critic_step += 1

            batch = self.rb.sample(
                batch_size=self.cfg.critic_batch_size,
                steps=self.cfg.n_steps,
                next_obs=False
            ).to(self.device)

            prior_ratio = self._prior_data_ratio()
            if prior_ratio > 0.0:
                batch_prior = self.rb_prior.sample(
                    batch_size=self.cfg.critic_batch_size * prior_ratio,
                    steps=self.cfg.n_steps
                ).to(self.device)
            else:
                batch_prior = None
                
            d = i == iters - 1
            info = self.train_critic(batch, batch_prior=batch_prior, diagnostics=d)
            infos.update(info)
            infos.update({"critic/step": self.critic_step})

            if self.enable_actor: # train actor every `delayed` critic updates.
                if self.critic_step % self.cfg.delayed == 0:
                    self.actor_step += 1
                    info = self.train_actor(diagnostics=True)
                    infos.update(info)
                    infos.update({"actor/step": self.actor_step})

        self._flush_dormancy(infos)
        return dict(sorted(infos.items()))


    @ScopedTimer("train_critic")
    def train_critic(
        self, 
        batch: TensorDict, 
        batch_prior: TensorDict | None = None,
        diagnostics: bool = False
    ):
        self.Q.train() # set training mode
        batch = batch.select(*self.train_keys, inplace=True, strict=False)
        
        if self.cfg.n_steps == 1:
            B_online = batch.shape[0]
        else:
            B_online = batch.shape[1]

        if batch_prior is not None:
            batch_prior = batch_prior.select(*self.train_keys, inplace=True, strict=False)
            if self.cfg.n_steps == 1:
                B_prior = batch_prior.shape[0]
                batch = torch.cat([batch, batch_prior], dim=0)
            else:
                B_prior = batch_prior.shape[1]
                batch = torch.cat([batch, batch_prior], dim=1)
                
        else:
            B_prior = 0
        
        B_eff = B_online + B_prior

        reward = batch[REWARD_KEY]

        if isinstance(reward, TensorDict):
            reward = torch.cat(list(reward.values()), dim=-1)

        reward = reward.sum(-1, keepdim=True).clamp_min(0.)


        if self.cfg.debug:
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
        
        # TODO: Why reward is scaled with gamma?
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            reward = reward * (1.0 - self.cfg.gamma)
        
        self.preproc(batch)
        self.preproc(batch['next'])

        # Process batch data and prepare:
        # (obs, act, next_obs, discount, is_init, terminated)
        if self.cfg.n_steps == 1:
            obs        = batch['_input_normed']
            act        = batch[ACTION_KEY]
            next_obs   = batch['next', '_input_normed']
            term       = batch[TERM_KEY].float()
            env_disc   = batch.get(('next', 'discount'))
            if env_disc is None:
                env_disc = torch.ones_like(term)
            discount   = self.cfg.gamma * env_disc * (1.0 - term)
            is_init    = batch['is_init']
            term_flat  = batch[TERM_KEY]
            if term_flat.dim() > 1 and term_flat.shape[-1] == 1:
                term_flat = term_flat.squeeze(-1)
            terminated = term_flat.bool()
        else:
            assert self.msr is not None
            batch_done = batch[DONE_KEY][:self.msr.n_steps]
            batch_term = batch[TERM_KEY][:self.msr.n_steps]
            if (next_obs := batch.get(("next", "_input_normed"))) is None:
                assert batch.shape[0] == self.msr.n_steps + 1
                next_obs = torch.where(
                    batch_done,
                    batch[OBS_KEY][:self.msr.n_steps],
                    batch[OBS_KEY][1:self.msr.n_steps+1]
                )
            obs = batch["_input_normed"][0]
            act_n = batch[ACTION_KEY]
            env_disc_ms = batch.get(("next", "discount"))
            if env_disc_ms is not None:
                env_disc_ms = env_disc_ms[:self.msr.n_steps]
            act_n, next_obs, reward, discount, terminated = self.msr(
                actions=act_n,
                next_observations=next_obs,
                rewards=reward[:self.msr.n_steps],
                terminated=batch_term,
                done=batch_done,
                env_discount=env_disc_ms
            )
            act = act_n[:, 0]
            is_init = batch["is_init"][0]
        
        with ScopedTimer("compute_target"):
            q_target = self.compute_target(next_obs, reward, discount)
        
        # print("[LOG] obs shape: ", obs.shape)
        # print("[LOG] act shape: ", act.shape)
        pred = self.Q(obs, act)
        per_sample_q_loss = self.Q.compute_loss(pred, q_target)
        valid = (1.0 - is_init.float()).reshape_as(per_sample_q_loss)
        denom = valid.sum().clamp_min(1e-8)
        q_loss = (per_sample_q_loss * valid).sum() / denom # MSBE loss

        self.opt_Q.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=self.cfg.max_grad_norm)
        self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, self.cfg.tau_Q)

        # ================= Log ===================
        if not diagnostics:
            return {}
        
        infos: dict = {
            "critic/q_loss": q_loss.item(),
            "critic/grad_norm": critic_grad_norm.item()
        }

        with torch.no_grad():
            if self.cfg.n_steps > 1: 
                obs_t1 = batch["_input_normed"][1, :B_online]
                act_t1 = batch[ACTION_KEY][1, :B_online]
                done_t0 = batch[DONE_KEY][0, :B_online].reshape(B_online)
                alive_t1 = ~done_t0.bool()
                if alive_t1.any():
                    policy_act_t1 = self.actor(obs_t1)[0]
                    l2_t1 = torch.linalg.vector_norm(policy_act_t1 - act_t1, dim=-1)
                    infos["critic/action_mismatch_t1"] = l2_t1[alive_t1].mean().item()

            if self.cfg.distributional:
                logits = self.Q(obs, act)
                q = self.Q.expected_values(logits)
                q_lower = self.Q.expected_values(logits, risk_alpha=0.5)
                q_upper = self.Q.expected_values(logits, risk_alpha=-0.5)
            else:
                q = self.Q.get_values(obs, act)
            
            if self.reward_normalizer is not None:
                q = self.reward_normalizer.denormalize_return_values(q)
                if self.cfg.distributional:
                    q_lower = self.reward_normalizer.denormalize_return_values(q_lower)
                    q_upper = self.reward_normalizer.denormalize_return_values(q_upper)

                    infos["critic/q_lower"] = q_lower.mean().item()
                    infos["critic/q_upper"] = q_upper.mean().item()

            # online Q statistics
            q_val_mean = q[:B_online].mean().item()
            q_val_max = q[:B_online].max().item()
            q_val_std = q[:B_online].std(dim=-1).mean().item() # why `mean`?
                
            infos["critic/q_value"] = q_val_mean
            infos["critic/q_max"] = q_val_max
            infos["critic/q_std"] = q_val_std

            if B_prior > 0:
                q_prior = q[B_online: B_eff]
                infos["critic/prior_q_mean"] = q_prior.mean().item()
                infos["critic/prior_q_max"] = q_prior.max().item()

            if terminated.any():
                q_val_terminated = q[terminated.reshape(q.shape[0])]
                infos["critic/q_value_terminated"] = q_val_terminated.mean().item()
                infos["critic/q_loss_terminated"] = per_sample_q_loss[terminated.reshape(q.shape[0])].mean().item()

            return infos    


    @torch.no_grad()
    def _compute_target(self, next_obs, reward, discount):
        """
        No hard clamp.
        """
        action = self.actor_target(next_obs)
        action = action + torch.randn_like(action) * self.cfg.target_action_noise

        q_target = self.Q_target.compute_target(
            next_obs,
            action,
            reward,
            discount
        )
        return q_target


    @ScopedTimer("train_actor")
    def train_actor(
        self,
        diagnostics: bool = False
    ):
        # Sample and process
        # ============================================================================
        batch = self.rb.sample(batch_size=self.cfg.actor_batch_size, steps=1, next_obs=False) \
                       .to(self.device) \
                       .select(*self.train_keys, strict=False)
        
        batch_prior = None
        prior_ratio = self._prior_data_ratio()
        if prior_ratio > 0.0:
            batch_prior = self.rb_prior.sample(
                batch_size=self.cfg.actor_batch_size * prior_ratio,
                steps=1
            ).select(*self.train_keys).to(self.device)

            batch = torch.cat([batch, batch_prior], dim=0)
            prior_action = batch_prior[ACTION_KEY]

        self.preproc(batch)
        obs = batch["_input_normed"]
        act = batch[ACTION_KEY]
        is_init = batch["is_init"]
        n_unaug = obs.shape[0]
        prior_obs = None
        prior_count = 0

        if batch_prior is not None:
            prior_count = batch_prior.shape[0]
            prior_obs = batch_prior["policy"]        
        # ============================================================================

        with hold_out_net(self.Q):
            update_act = self.actor(obs) # obs shape: [bs, do]; act shape: [bs, da]
            q = self.Q.get_values(obs, update_act).mean(dim=-1) # q shape: [bs, 1]

            policy_term = -q # actor loss shape: [bs, 1]
            soft_term = 0.01 * ((update_act/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
            actor_loss = policy_term + soft_term
            # valid shape: [bs, 1]
            valid = (1.0 - is_init.float()).reshape_as(actor_loss)
            denom = valid.sum().clamp_min(1e-8)
            actor_loss = (actor_loss * valid).sum() / denom # actor loss shape: [1]

            q_action_grad_norm: torch.Tensor | None = None
            if diagnostics:
                (grad_q_wrt_a, ) = torch.autograd.grad(
                    q.sum(),
                    update_act,
                    retain_graph=True,
                    create_graph=False
                )
                q_action_grad_norm = grad_q_wrt_a.norm(dim=-1).mean()
            
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_actor.step()

        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        # Log
        # ============================================================================
        if not diagnostics:
            return {}
        
        assert q_action_grad_norm is not None
        with torch.no_grad():
            q_for_log = q # q for log shape: [bs, 1]
            if self.reward_normalizer is not None:
                q_for_log = self.reward_normalizer.denormalize_return_values(q_for_log)
            infos = {
                "actor/loss": actor_loss.item(),
                "actor/grad_norm": actor_grad_norm.item(),
                "actor/q_std": q_for_log.std(dim=0).mean().item(),
                "actor/q_action_grad_norm": q_action_grad_norm.item(),
                "actor/mean_act": update_act.abs().mean().item()
            }

            if batch_prior is not None:
                q_prior = self.Q.get_values(prior_obs, prior_action).mean(dim=-1)
                q_policy_prior = q[:n_unaug][-prior_count:].mean(dim=-1)
                advantage = q_policy_prior - q_prior
                infos["actor/online_advantage"] = advantage.mean().item()
            
            return infos
        # ============================================================================


    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["Q"] = self.Q.state_dict()
        state_dict["actor"] = self.actor.state_dict()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        if self.reward_normalizer is not None:
            state_dict["reward_normalizer"] = self.reward_normalizer.state_dict()
        return state_dict


    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.Q.load_state_dict(state_dict["Q"], strict=strict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"], strict=strict)
        rk = state_dict.get("reward_normalizer")
        if rk is not None and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(rk)


class TD3RolloutPolicy(TensorDictModuleBase):
    def __init__(
        self,
        preproc: nn.Module,
        actor: nn.Module,
        noise_type: str,
        noise_scale: float,
        act_dim: int,
        seq_len: int,
        *,
        device: torch.device | str = 'cuda',
        Q: nn.Module | None = None,
        reward_normalizer: RewardNormalizer | None = None,
        critic: bool = False
    ):
        super().__init__()
        self.preproc = preproc
        self.actor = actor
        self.Q = Q
        self.noise_type = noise_type
        self.reward_normalizer = reward_normalizer
        self.critic = critic
        self.noise_scale = noise_scale
        self.device = device

        self.in_keys = [OBS_KEY]
        self.out_keys = [ACTION_KEY]
        if self.critic is not None:
            self.out_keys = self.out_keys + ["Q_value"]

        if self.noise_type == "pink":
            self.pink_noise = ParallelPinkNoiseProcess(
                size=[4096, act_dim, seq_len],
                scale=noise_scale
            )
            self.pink_noise.reset()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.preproc(tensordict)
        obs = tensordict["_input_normed"]
        act = self.actor(obs)
        ter = tensordict["done"]
        if isinstance(ter, TensorDict):
            ter = torch.cat(list(ter.values()), dim=-1)
            ter = torch.squeeze(ter)

        if interaction_type() == InteractionType.MODE:
            sample = act.clone()
        elif self.noise_type == "white":
            noise = torch.randn_like(act) * self.noise_scale
            sample = act + noise
        elif self.noise_type == "pink":
            noise = self.pink_noise.sample_one()
            noise = torch.as_tensor(noise, dtype=act.dtype, device=act.device)
            sample = act + noise
        
        if ter.any():
            self.pink_noise.reset(mask=ter.numpy())
        
        if self.critic and self.Q is not None:
            qs = self.Q.get_values(obs, sample).mean(dim=-1)
            if self.reward_normalizer is not None:
                qs = self.reward_normalizer.denormalize_return_values(qs)
            tensordict["Q_value"] = qs
        
        tensordict[ACTION_KEY] = sample
        tensordict["loc"] = act
        return tensordict