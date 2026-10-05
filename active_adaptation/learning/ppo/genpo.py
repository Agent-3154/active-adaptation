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

"""GenPO++ (arXiv:2606.06967): on-policy flow policies with exact likelihood ratios.

The actor is a reversible history-augmented solver. Each step updates

    x_{i+1} = (1 - σ) x_i + σ x_{i-1} + (1 + σ) Δt v_θ(x_i, t_i | s)

and the environment executes only the terminal state. The companion map has
log-determinant ``M * d * log|σ|``, independent of v_θ, so the PPO ratio between
two policies collapses to the ratio of standard-normal densities of the inverted
augmented noises. Symmetry augmentation is not implemented.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple, Union

import torch
import torch.distributed as distr
import torch.nn as nn
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase
from tensordict.nn import TensorDictSequential as Seq
from tensordict.nn.probabilistic import InteractionType, interaction_type
from torchrl.data import Composite, TensorSpec

import active_adaptation as aa
from active_adaptation.learning.modules import CatTensors, MLP, VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    REWARD_KEY,
    TERM_KEY,
    Critic,
    GAE,
    make_batch,
    normalize,
    resolve_clip_param,
)
from active_adaptation.learning.utils.distributed import (
    check_parameters,
    unwrap_ddp,
    wrap_ddp,
)
from active_adaptation.learning.utils.dormancy import DormancyTracker
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.utils.profiling import ScopedTimer

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


# Terminal solver state one step before the executed action. Kept off the
# leading-underscore private-key list so BufferCollector retains it.
FLOW_HIST_KEY = "flow_hist"


def base_log_prob(z: torch.Tensor) -> torch.Tensor:
    """Standard-normal log-density of the augmented noise, summed over the last dim."""
    latent_dim = z.shape[-1]
    return -0.5 * (z.square().sum(dim=-1) + latent_dim * math.log(2.0 * math.pi))


class ReversibleFlowActor(nn.Module):
    """Conditional velocity field plus the GenPO++ reversible solver.

    ``forward`` is the training path: invert a stored terminal pair and return
    the base log-density of the recovered noise. ``act`` is the rollout path.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation: type[nn.Module],
        hidden_dims: Tuple[int, ...],
        *,
        timestep_embed_dim: int,
        flow_steps: int,
        sigma: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.flow_steps = flow_steps
        self.sigma = float(sigma)
        self.dt = 1.0 / float(flow_steps)

        self.backbone = MLP(
            num_units=[obs_dim + timestep_embed_dim + action_dim, *hidden_dims],
            activation=activation,
            first_non_muon=True,
        )
        self.velocity_head = nn.Linear(hidden_dims[-1], action_dim)
        self.velocity_head.weight._non_muon = True
        half = timestep_embed_dim // 2
        freqs = 2.0 ** torch.arange(half, dtype=torch.float32)
        self.register_buffer("time_freqs", freqs, persistent=False)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.time_freqs.to(dtype=t.dtype)
        scaled = t * freqs
        return torch.cat((scaled.cos(), scaled.sin()), dim=-1)

    def velocity(self, obs: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v_θ(x, t | s). ``t`` has a trailing length-1 axis."""
        inp = torch.cat((obs, self._embed_time(t), x), dim=-1)
        return self.velocity_head(self.backbone(inp))

    def integrate(
        self,
        obs: torch.Tensor,
        x: torch.Tensor,
        x_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map augmented noise ``(x, x_hist)`` to the terminal pair ``(x_M, x_{M-1})``."""
        sigma = self.sigma
        dt = self.dt
        for i in range(self.flow_steps):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            v = self.velocity(obs, x, t)
            x_next = (1.0 - sigma) * x + sigma * x_hist + (1.0 + sigma) * dt * v
            x_hist = x
            x = x_next
        return x, x_hist

    def invert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        flow_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Recover ``z_0 = (x_0, x_{-1})`` from the stored terminal pair."""
        sigma = self.sigma
        inv_sigma = 1.0 / sigma
        dt = self.dt
        x_next = action
        x = flow_hist
        for i in range(self.flow_steps - 1, -1, -1):
            t = x.new_full(x.shape[:-1] + (1,), i * dt)
            v = self.velocity(obs, x, t)
            x_prev = inv_sigma * (
                x_next - (1.0 - sigma) * x - (1.0 + sigma) * dt * v
            )
            x_next = x
            x = x_prev
        return torch.cat((x_next, x), dim=-1)

    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``z_0``, integrate, and return ``(action, flow_hist, log p(z_0))``."""
        leading = obs.shape[:-1]
        deterministic = interaction_type() in (
            InteractionType.MODE,
            InteractionType.DETERMINISTIC,
        )
        if deterministic:
            x = obs.new_zeros(*leading, self.action_dim)
            x_hist = obs.new_zeros(*leading, self.action_dim)
        else:
            x = torch.randn(*leading, self.action_dim, device=obs.device, dtype=obs.dtype)
            x_hist = torch.randn(
                *leading, self.action_dim, device=obs.device, dtype=obs.dtype
            )
        # log|det J| depends only on σ and cancels in the likelihood ratio.
        log_prob = base_log_prob(torch.cat((x, x_hist), dim=-1))
        action, flow_hist = self.integrate(obs, x, x_hist)
        return action, flow_hist, log_prob

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        flow_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable base log-density of the noise that inverts to this pair."""
        return base_log_prob(self.invert(obs, action, flow_hist))

    def augmented_entropy(self) -> float:
        """Entropy of the pushforward. Constant in θ because log|det J| is constant."""
        latent_dim = 2 * self.action_dim
        base = 0.5 * latent_dim * (1.0 + math.log(2.0 * math.pi))
        log_abs_det = (
            self.flow_steps * self.action_dim * math.log(abs(self.sigma))
        )
        return base + log_abs_det


@dataclass
class GenPOConfig:
    _target_: str = "active_adaptation.learning.ppo.genpo.GenPOConfig"
    name: str = "genpo_pp"
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 5e-4
    desired_kl: Union[float, None] = 0.01
    # Scalar ε → [1-ε, 1+ε]. Length-2 list/tuple [eps_neg, eps_pos] →
    # [1-eps_neg, 1+eps_pos]. Typed as Any: Hydra cannot express that union.
    clip_param: Any = (0.2, 0.2)
    # Augmented entropy does not depend on θ, so this term has zero gradient.
    entropy_coef: float = 0.0

    clamp_reward: bool = False
    activation: str = "Mish"
    muon: bool = False

    actor_num_units: Tuple[int, ...] = (256, 256, 256)
    critic_num_units: Tuple[int, ...] = (512, 256, 256)
    timestep_embed_dim: int = 8
    flow_steps: int = 5
    # History-mixing coefficient. σ = 1 matches AB2 up to O(Δt^3); 0.75 is the
    # paper's default across the IsaacLab suite.
    sigma: float = 0.75

    compile: bool = False
    use_ddp: bool = True
    debug: bool = False

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)

    def __post_init__(self):
        if float(self.sigma) == 0.0:
            raise ValueError("sigma must be non-zero so the history update is invertible")
        if int(self.flow_steps) < 1:
            raise ValueError(f"flow_steps must be >= 1, got {self.flow_steps}")
        embed = int(self.timestep_embed_dim)
        if embed < 2 or embed % 2 != 0:
            raise ValueError(
                f"timestep_embed_dim must be a positive even integer, got {embed}"
            )
        if len(tuple(self.actor_num_units)) == 0:
            raise ValueError("actor_num_units must be non-empty")
        if len(tuple(self.critic_num_units)) == 0:
            raise ValueError("critic_num_units must be non-empty")

    def get_class(self):
        return GenPOPolicy


cs = ConfigStore.instance()
cs.store("genpo_pp", node=GenPOConfig, group="algo")


def vecnorm_sync_(module: nn.Module):
    if isinstance(module, VecNorm):
        module.synchronize(mode="broadcast")


class GenPOPolicy(TensorDictModuleBase):
    """PPO-style learner whose actor is a GenPO++ reversible flow."""

    def __init__(
        self,
        cfg: GenPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
    ):
        del reward_spec
        super().__init__()
        self.cfg = cfg
        if self.cfg.debug and self.cfg.compile:
            raise ValueError("Debug mode and compile mode cannot be enabled together")
        self.device = device

        self.max_grad_norm = 1.0
        self.clip_param = resolve_clip_param(self.cfg.clip_param)
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        self.should_reduce_grads = False
        self.world_size = 1

        fake_input = observation_spec.zero().to(self.device)
        if CMD_KEY in observation_spec.keys(True, True):
            self.in_keys = (CMD_KEY, OBS_KEY)
        else:
            self.in_keys = (OBS_KEY,)
        inp_dim = sum(int(fake_input[key].shape[-1]) for key in self.in_keys)
        self.vecnorm = Seq(
            CatTensors(list(self.in_keys), "_input", del_keys=False, sort=False),
            Mod(VecNorm((inp_dim,), decay=1.0), ["_input"], ["_obs_normed"]),
        ).to(self.device)

        if ACTION_KEY not in action_spec.keys(True, True):
            raise KeyError(f"action spec is missing {ACTION_KEY!r}")
        self.action_dim = int(action_spec[ACTION_KEY].shape[-1])

        activation = getattr(nn, self.cfg.activation)
        hidden = tuple(int(size) for size in self.cfg.actor_num_units)
        self.actor = ReversibleFlowActor(
            obs_dim=inp_dim,
            action_dim=self.action_dim,
            activation=activation,
            hidden_dims=hidden,
            timestep_embed_dim=int(self.cfg.timestep_embed_dim),
            flow_steps=int(self.cfg.flow_steps),
            sigma=float(self.cfg.sigma),
        ).to(self.device)
        # Bound method, not the module: registering the actor twice is illegal,
        # and rollout must call ``act`` rather than the training ``forward``.
        self.actor_rollout = Mod(
            self.actor.act,
            ["_obs_normed"],
            [ACTION_KEY, FLOW_HIST_KEY, "action_log_prob"],
        )

        critic_hidden = tuple(int(size) for size in self.cfg.critic_num_units)
        critic_mlp = MLP(
            num_units=[inp_dim, *critic_hidden],
            activation=activation,
            first_non_muon=True,
        )
        self.critic = Seq(
            Mod(critic_mlp, ["_obs_normed"], ["_critic_feature"]),
            Mod(Critic(1), ["_critic_feature"], ["state_value"]),
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor_rollout(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)
        self._rollout_dormancy_tracker: Union[DormancyTracker, None] = None
        self.update = self._update

    @classmethod
    def from_env(cls, cfg: GenPOConfig, env: "_EnvBase", device: str):
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
        )

    def on_stage_start(self, stage: str, env: "_EnvBase"):
        del stage, env
        if aa.is_distributed():
            aa.bind_local_rank_device()
            if self.cfg.use_ddp:
                self.actor = wrap_ddp(
                    self.actor, device_ids=[aa.get_local_cuda_index()]
                )
                self.critic = wrap_ddp(
                    self.critic, device_ids=[aa.get_local_cuda_index()]
                )
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
        self.world_size = aa.get_world_size()
        self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp

        if self.cfg.muon:
            self.opt = MuonAdamWWrapper(
                [unwrap_ddp(self.actor), unwrap_ddp(self.critic)],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01,
            )

        self.update = self._update
        if self.cfg.compile and not aa.is_distributed():
            self.update = torch.compile(self.update)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if self._rollout_dormancy_tracker is not None:
            self._rollout_dormancy_tracker.close()
            self._rollout_dormancy_tracker = None
        vecnorm = self.vecnorm if mode == "train" else VecNorm.freeze()(self.vecnorm)
        modules = [vecnorm, self.actor_rollout]
        if critic:
            modules.append(self.critic)
        policy = Seq(*modules)
        if self.cfg.compile:
            policy = torch.compile(policy)
        if self.cfg.debug:
            tracker = DormancyTracker(policy)
            policy.forward = tracker.wrap(policy.forward)
            self._rollout_dormancy_tracker = tracker
        return policy

    @torch.no_grad()
    def compute_value(self, tensordict: TensorDict):
        self.vecnorm(tensordict)
        return self.critic(tensordict)

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats").to(self.device, non_blocking=True)
        valid_ratio = (~tensordict["is_init"]).float().mean()
        infos = []

        self.vecnorm.to(self.device, non_blocking=True)
        self.actor.to(self.device)
        self.critic.to(self.device)

        with ScopedTimer("compute_advantage"):
            self.compute_advantage(
                tensordict, self.critic, "adv", "ret", self.cfg.clamp_reward
            )
            adv_unnormalized = tensordict["adv"]
            log_probs_before = tensordict["action_log_prob"]
            tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        ret_var = tensordict["ret"][~tensordict["is_init"]].var().clamp_min(1e-7)

        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self.update(minibatch, ret_var))

                if self.cfg.desired_kl is not None:
                    # Eq. 15: augmented KL -E[log r]. A negative MC estimate
                    # means no excess KL, so it does not raise the learning rate.
                    # Average across ranks so every rank applies the same LR step.
                    kl = infos[-1]["actor/aug_kl"].detach().clamp_min(0.0)
                    if aa.is_distributed():
                        distr.all_reduce(kl, op=distr.ReduceOp.SUM)
                        kl = kl / self.world_size
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.cfg.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.cfg.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-3, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr

        with torch.no_grad():
            self.vecnorm(tensordict)
            log_probs_after = unwrap_ddp(self.actor)(
                tensordict["_obs_normed"],
                tensordict[ACTION_KEY],
                tensordict[FLOW_HIST_KEY],
            )
            pg_loss_after = (
                log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            )
            pg_loss_before = (
                log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized
            )

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/value_max"] = tensordict["ret"].max().item()
        infos["critic/adv_mean"] = adv_unnormalized.mean().item()
        infos["critic/adv_std"] = adv_unnormalized.std().item()
        reward_aggregated = tensordict["next", "reward_aggregated"]
        infos["critic/neg_rew_ratio"] = (reward_aggregated <= 0.0).float().mean().item()
        infos["critic/valid_ratio"] = valid_ratio.item()

        if self.cfg.debug and self._rollout_dormancy_tracker is not None:
            dormancy = self._rollout_dormancy_tracker.compute_dormancy()
            for module_name, value in dormancy.items():
                infos[f"dormancy/{module_name}"] = value
            self._rollout_dormancy_tracker.reset()

        if aa.is_distributed():
            self.vecnorm.apply(vecnorm_sync_)
            if self.cfg.debug:
                infos["actor/diff"] = check_parameters(self.actor)
                infos["critic/diff"] = check_parameters(self.critic)
        return dict(sorted(infos.items()))

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
            with tensordict.view(-1) as td_flat:
                critic(self.vecnorm(td_flat))
                critic(self.vecnorm(td_flat["next"]))

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

    @ScopedTimer("genpo_update")
    def _update(self, tensordict: TensorDict, ret_var: torch.Tensor):
        self.vecnorm(tensordict)
        obs = tensordict["_obs_normed"]
        action = tensordict[ACTION_KEY]
        flow_hist = tensordict[FLOW_HIST_KEY]

        log_probs = self.actor(obs, action, flow_hist)
        log_probs_data = tensordict["action_log_prob"].reshape(log_probs.shape)
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = log_ratio.exp()

        valid = (~tensordict["is_init"]).reshape(log_ratio.shape).float()
        valid_cnt = valid.sum().clamp_min(1.0)

        adv = tensordict["adv"]
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param[0], 1.0 + self.clip_param[1])
        policy_loss = -(torch.min(surr1, surr2) * valid).sum() / valid_cnt
        # Constant in θ; kept so ``entropy_coef`` matches the PPO loss layout.
        entropy = log_probs.new_tensor(unwrap_ddp(self.actor).augmented_entropy())
        entropy_loss = -self.cfg.entropy_coef * entropy

        returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(returns, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if self.should_reduce_grads:
            for module in (self.actor, self.critic):
                for param in module.parameters():
                    if param.grad is None:
                        continue
                    distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                    param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.opt.step()

        with torch.no_grad():
            aug_kl = -(log_ratio * valid).sum() / valid_cnt
            approx_kl = (((ratio - 1.0) - log_ratio) * valid).sum() / valid_cnt
            info = {
                "actor/policy_loss": policy_loss.detach(),
                "actor/entropy": entropy.detach(),
                "actor/grad_norm": actor_grad_norm,
                "actor/approx_kl": approx_kl,
                "actor/aug_kl": aug_kl,
                "actor/clamp_pos": (ratio > 1.0 + self.clip_param[1]).float().mean(),
                "actor/clamp_neg": (ratio < 1.0 - self.clip_param[0]).float().mean(),
                "critic/value_loss": value_loss.detach(),
                "critic/explained_var": 1.0 - value_loss / ret_var,
                "critic/grad_norm": critic_grad_norm,
            }
        return info

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            module = unwrap_ddp(module)
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            module_state = state_dict.get(name, {})
            try:
                unwrap_ddp(module).load_state_dict(module_state, strict=strict)
                succeed_keys.append(name)
            except Exception as exc:
                warnings.warn(f"Failed to load state dict for {name}: {exc}")
                failed_keys.append(name)
        if failed_keys:
            print(f"Failed to load {failed_keys}. Loaded {succeed_keys}.")
        else:
            print(f"Successfully loaded {succeed_keys}.")
        return failed_keys
