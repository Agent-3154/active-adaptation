import torch
import hydra
import numpy as np
import wandb
import logging
import time
import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir, JobConf
from hydra.core.config_store import ConfigStore

from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.ppo_base import PPOBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "ppo"},
    "_self_",
]


@dataclass
class IsaacAppConfig:
    """Isaac Lab AppLauncher settings (resolved from parent config)."""

    headless: bool = "${..headless}"
    """Mirror ``headless``; passed to Isaac Lab's AppLauncher."""
    enable_cameras: bool = "${..eval_render}"
    """Mirror ``eval_render``; enables camera sensors for final eval rendering."""


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    name: str = "${..exp_name}/${now:%m-%d}_${now:%H-%M}"
    """Run display name (derived from ``exp_name`` and timestamp)."""
    job_type: str = "train"
    """WandB job type label."""
    project: str = "${oc.select:task.project,active_adaptation}"
    """WandB project; falls back to ``active_adaptation`` if unset on the task."""
    mode: str = "online"
    """WandB mode: ``online``, ``offline``, or ``disabled``."""
    tags: List[str] = field(default_factory=list)
    """Optional tags attached to the WandB run."""


@dataclass
class TrainConfig:
    """Hydra root config for PPO training."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    """Hydra defaults list: task config, algo config, then this config."""
    hydra: HydraConf = field(default_factory=HydraConf)
    """Hydra runtime settings (output directory, chdir, etc.)."""

    headless: bool = True
    """Run simulation without a rendering window."""
    exp_name: str = "${oc.select:task.name,test}-${oc.select:algo.name,none}"
    """Experiment label used in run names and WandB metadata."""
    backend: str = "isaac"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for training (adjusted per local rank when using CUDA)."""

    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""
    total_frames: int = 150_000_000
    """Total environment frames to collect across all ranks before stopping."""

    eval_render: bool = False
    """Render the environment during the final post-training evaluation."""
    checkpoint_interval: int = 4
    """Save a local checkpoint every N training iterations."""
    upload_interval: int = 100
    """Upload a checkpoint to WandB every N training iterations."""

    seed: int = 42
    """Random seed (offset by local rank in distributed runs)."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to resume from; ``null`` trains from scratch."""
    discard_unused_obs: bool = True
    """Drop observation groups not listed in ``algo.in_keys``."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    """WandB logging configuration."""


cs = ConfigStore.instance()
cs.store(
    name="train",
    node=TrainConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_train/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            ),
            job=JobConf(chdir=True),
        )
    ),
)


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


class StackingCollector:
    """Collect rollouts by appending per-step outputs then stacking.

    This collector is simple and robust, but it allocates new per-step tensors
    and a stacked output each iteration, which increases transient memory usage.
    """

    def __init__(
        self,
        env: TransformedEnv,
        steps: int,
        transitions: bool = False,
    ):
        self.env = env
        self.steps = steps
        self.transitions = transitions
        self.device = env.device
        self._observation_keys = list(env.observation_spec.keys(True, True))

    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(self, carry: TensorDictBase, rollout_policy: TensorDictModuleBase):
        rollout_policy = rollout_policy.to(device=self.device)
        data = []
        for _ in range(self.steps):
            with ScopedTimer("policy_inference"):
                carry = rollout_policy(carry)
            td, carry = self.env.step_and_maybe_reset(carry)
            private_keys = [
                key
                for key in td.keys(True, True)
                if isinstance(key, str) and key.startswith("_")
            ]
            td = td.exclude(*private_keys)
            if not self.transitions:
                td["next"] = td["next"].exclude(*self._observation_keys)
            if self.device is not None:
                td = td.to(self.device)
            data.append(td)
        data = torch.stack(data, dim=1)
        return data, carry


class BufferCollector:
    """Collect rollouts into a preallocated TensorDict buffer.

    This collector reuses storage across iterations and can be more memory
    efficient, but requires a fixed schema and careful handling of aliasing when
    returning the internal buffer.
    """

    def __init__(self, env: TransformedEnv, steps: int, transitions: bool=False):
        self.env = env
        self.steps = steps
        self.transitions = transitions
        self._observation_keys = list(env.observation_spec.keys(True, True))
        
        buffer = env.fake_tensordict()
        if not transitions:
            buffer["next"] = buffer["next"].exclude(*self._observation_keys)
        self._buffer = buffer.unsqueeze(1).expand(env.shape[0], steps).clone()
        buffer_nbytes = sum(
            value.numel() * value.element_size()
            for _, value in self._buffer.items(True, True)
            if torch.is_tensor(value)
        )
        logging.info(
            "BufferCollector allocated %.2f MiB on %s",
            buffer_nbytes / (1024**2),
            self._buffer.device,
        )
    
    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(self, carry: TensorDictBase, rollout_policy: TensorDictModuleBase):
        for i in range(self.steps):
            with ScopedTimer("policy_inference"):
                carry = rollout_policy(carry)
            td, carry = self.env.step_and_maybe_reset(carry)
            private_keys = [
                key
                for key in td.keys(True, True)
                if isinstance(key, str) and key.startswith("_")
            ]
            td = td.exclude(*private_keys)
            if not self.transitions:
                td["next"] = td["next"].exclude(*self._observation_keys)
            self._buffer[:, i] = td
        # TensorDict.copy() returns a shallow copy
        return self._buffer.copy(), carry


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: TrainConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        run_dir = Path(run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        run.save(str(cfg_save_path), policy="now")
        run.save(str(run_dir / "config.yaml"), policy="now")

    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.utils.helpers import EpisodeStats

    env, policy = make_env_policy(
        task_cfg=cfg.task,
        algo_cfg=cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        discard_unused_obs=cfg.discard_unused_obs,
        checkpoint_path=cfg.checkpoint_path,
    )
    policy: PPOBase

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.total_frames // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    
    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval

    max_episode_length = cfg.task.max_episode_length
    log_interval = (max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        run_dir = Path(run.dir)
        ckpt_path = run_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        
        torch.save(state_dict, ckpt_path)
        if upload_to_wandb:
            run.save(str(ckpt_path), policy="now", base_path=run.dir)
        
        latest_link = run_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        logging.info(f"Saved checkpoint to {ckpt_path}" + (" (wandb)" if upload_to_wandb else ""))
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0 or i % upload_interval == 0

    ckpt_path = None
    carry = env.reset()
    transitions = cfg.algo.get("store_transitions", True)
    collector = StackingCollector(
        env,
        steps=cfg.algo.train_every,
        transitions=transitions,
    )

    env_frames = 0

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    for stage in stages:

        policy.on_stage_start(stage, env)
        rollout_policy = policy.get_rollout_policy(
            "train",
            critic=not transitions,
        )

        if aa.is_main_process():
            progress = tqdm(range(total_iters), desc=stage)
        else:
            progress = range(total_iters)

        for i in progress:
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                data, carry = collector.collect(carry, rollout_policy)
                if not transitions:
                    state_value = data["state_value"]
                    next_state_value = policy.compute_value(carry.copy())["state_value"]
                    next_state_value = torch.cat([
                        data["state_value"][:, 1:],
                        next_state_value.unsqueeze(1),
                    ], dim=1)
                    # Since terminal next observations are dropped, approximate V_{t+1} with V_t.
                    data["next", "state_value"] = torch.where(
                        data["next", "done"],
                        state_value,
                        next_state_value,
                    )
            rollout_time = rollout_timer.last_time

            episode_stats.add(data)
            env_frames += data.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data))
            training_time = training_timer.last_time

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)

            if aa.is_main_process():
                ScopedTimer.print_summary(clear=True, depth=3)
                print(
                    OmegaConf.to_yaml(
                        {k: v for k, v in info.items() if isinstance(v, (float, int))}
                    )
                )
                print(f"Latest checkpoint: {ckpt_path}")
                info.update(env.extra)
                info.update(env.stats_ema)  # step-wise exponential moving average of stats
                run.log(info)

    if aa.is_main_process():
        ckpt_path = save(policy, "checkpoint_final")
        policy_eval = policy.get_rollout_policy("eval")
        info, trajs, stats = evaluate(
            env, policy_eval, render=cfg.eval_render, seed=cfg.seed
        )
        info["env_frames"] = env_frames
        run.log(info)
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    exit(0)


if __name__ == "__main__":
    main()
