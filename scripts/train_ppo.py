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
from active_adaptation.pipeline_io import (
    RUN_STATE_FILENAME,
    get_run_state_dir,
    write_run_state,
)
from active_adaptation.utils.experiment_logging import (
    RUN_STATUS_FILENAME,
    metrics_export_enabled,
    export_iteration_monitoring,
    write_run_status,
    assess_health,
)
from active_adaptation.utils.profiling import (
    PROFILE_JSONL_FILENAME,
    ScopedTimer,
    profile_export_enabled,
    profile_print_enabled,
    profile_print_every,
)
from active_adaptation.utils.memory_profiling import (
    MEMORY_JSONL_FILENAME,
    ScopedMemoryTimer,
    append_memory_jsonl,
    build_iter_memory_record,
    cuda_memory_snapshot,
    memory_export_enabled,
    reset_cuda_peak_memory,
)
from active_adaptation.utils.torchrl import tensordict_nbytes
from active_adaptation.learning.ppo.ppo_base import PPOBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


DEFAULTS = [
    {"task": "???"},
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
    backend: str = "isaaclab"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for training (adjusted per local rank when using CUDA)."""

    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""
    total_frames: int = 150_000_000
    """Total environment frames to collect across all ranks before stopping."""

    eval: bool = True
    """Evaluate the policy after training."""
    eval_render: bool = False
    """Render the environment during the final post-training evaluation."""
    checkpoint_interval: int = 50
    """Overwrite ``checkpoint_latest.pt`` every N training iterations (local only)."""
    upload_interval: int = 100
    """Also write/upload a versioned ``checkpoint_{i}.pt`` every N iterations."""

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
        self._functional_keys = [
            key for key, group in env.observation_groups.items()
            if group.is_functional
        ]

    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(self, carry: TensorDictBase, rollout_policy: TensorDictModuleBase):
        rollout_policy = rollout_policy.to(device=self.device)
        data = []
        for _ in range(self.steps):
            compact = {}
            if self._functional_keys:
                with ScopedTimer("materialization"):
                    for key in self._functional_keys:
                        compact[key] = carry[key]
                        carry[key] = self.env.observation_groups[key].materialize(carry)
            with ScopedTimer("policy_inference"):
                carry = rollout_policy(carry)
            carry.update(compact)
            td, carry = self.env.step_and_maybe_reset(carry)
            # td.update(compact)
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
        self._functional_keys = [
            key for key, group in env.observation_groups.items()
            if group.is_functional
        ]
        
        buffer = env.fake_tensordict()
        if not transitions:
            buffer["next"] = buffer["next"].exclude(*self._observation_keys)
        self._buffer = buffer.unsqueeze(1).expand(env.shape[0], steps).clone()
    
    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(self, carry: TensorDictBase, rollout_policy: TensorDictModuleBase):
        for i in range(self.steps):
            compact = {}
            if self._functional_keys:
                with ScopedTimer("materialization"):
                    for key in self._functional_keys:
                        compact[key] = carry[key]
                        carry[key] = self.env.observation_groups[key].materialize(carry)
            with ScopedTimer("policy_inference"):
                carry = rollout_policy(carry)
            carry.update(compact)
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


def run(cfg: TrainConfig) -> dict[str, str]:
    """Train a PPO policy and return checkpoint paths for downstream stages."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

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

    # Sensor / Warp / Kit setup can reset current_device; both ranks must finish
    # env construction on the correct GPU before any post-env NCCL collective.
    aa.bind_local_rank_device()
    if aa.is_distributed():
        import torch.distributed as dist

        print(
            f"[rank {aa.get_local_rank()}] env ready, "
            f"current_device={torch.cuda.current_device()}, waiting on barrier…",
            flush=True,
        )
        dist.barrier()
        print(f"[rank {aa.get_local_rank()}] barrier passed", flush=True)

    wandb_run = None
    proc_name = None  # process name for `setproctitle`
    profiling_jsonl_path = None
    memory_jsonl_path = None
    run_dir = None
    if aa.is_main_process():
        wandb_run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run_idx = wandb_run.name.split("-")[-1]

        wandb_run.config.update(OmegaConf.to_container(cfg))
        wandb_run.config["world_size"] = aa.get_world_size()

        timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        wandb_run.name = f"{run_idx}-{cfg.exp_name}-{timestr}"
        proc_name = f"{run_idx}-{cfg.exp_name}"  # shorter for better display

        run_dir = Path(wandb_run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        OmegaConf.save(cfg.task, run_dir / "cfg_task.yaml")
        OmegaConf.save(cfg.algo, run_dir / "cfg_algo.yaml")
        wandb_run.save(str(cfg_save_path), policy="now")
        wandb_run.save(str(run_dir / "config.yaml"), policy="now")
        profiling_jsonl_path = run_dir / PROFILE_JSONL_FILENAME
        memory_jsonl_path = (
            run_dir / MEMORY_JSONL_FILENAME if memory_export_enabled() else None
        )

    if aa.is_distributed():
        import torch.distributed as dist

        aa.bind_local_rank_device()
        # Explicit device: Isaac AppLauncher / Warp may leave current_device at 0.
        name_list = [proc_name]
        dist.broadcast_object_list(
            name_list,
            src=0,
            device=torch.device(f"cuda:{aa.get_local_cuda_index()}"),
        )
        proc_name = name_list[0]

    if proc_name is not None:
        if aa.is_main_process():
            setproctitle(proc_name)
        else:
            setproctitle(f"{proc_name}-rank{aa.get_local_rank()}")

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

    def save(
        policy,
        *,
        archive_name: str | None = None,
        upload_to_wandb: bool = False,
    ) -> tuple[str, str | None]:
        """Refresh local ``checkpoint_latest.pt``; optionally archive + upload a versioned copy.

        Returns ``(latest_path, archived_path_or_None)``.
        """
        run_dir = Path(wandb_run.dir)
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": wandb_run.name, "id": wandb_run.id}
        state_dict["policy"] = policy.state_dict()

        latest_path = run_dir / "checkpoint_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        torch.save(state_dict, latest_path)

        archived_path = None
        if archive_name is not None:
            archived_path = run_dir / f"{archive_name}.pt"
            if archived_path.resolve() != latest_path.resolve():
                torch.save(state_dict, archived_path)
            if upload_to_wandb:
                wandb_run.save(
                    str(archived_path), policy="now", base_path=wandb_run.dir
                )

        return str(latest_path), (
            str(archived_path) if archived_path is not None else None
        )

    assert env.training

    def should_save(i: int) -> bool:
        if not aa.is_main_process():
            return False
        if checkpoint_interval > 0 and i % checkpoint_interval == 0:
            return True
        if upload_interval > 0 and i % upload_interval == 0:
            return True
        return False

    local_ckpt_path = None
    local_ckpt_iter: int | None = None
    uploaded_ckpt_path = None
    carry = env.reset()
    transitions = cfg.algo.get("store_transitions", True)
    # collector = StackingCollector(
    #     env,
    #     steps=cfg.algo.train_every,
    #     transitions=transitions,
    # )
    collector = BufferCollector(
        env,
        steps=cfg.algo.train_every,
        transitions=transitions,
    )

    buffer_MiB: float | None = None
    if aa.is_main_process():
        fake = env.fake_tensordict()
        if not transitions:
            del fake["next"]
        storage_nbytes = tensordict_nbytes(
            fake
            .unsqueeze(1)
            .expand(env.num_envs, cfg.algo.train_every)
        )
        buffer_MiB = storage_nbytes / (1024**2)
        wandb_run.summary["buffer_MiB"] = buffer_MiB

    env_frames = 0
    last_algo_metrics: dict[str, float | int] = {}

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

        t0 = time.time()
        for i in progress:
            rollout_start = time.perf_counter()
            phase_snapshots: dict[str, dict[str, float]] = {}
            if memory_export_enabled():
                reset_cuda_peak_memory()
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
            if memory_export_enabled() and aa.is_main_process():
                phase_snapshots["after_rollout"] = cuda_memory_snapshot()

            episode_stats.add(data)
            env_frames += data.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            if memory_export_enabled():
                ScopedMemoryTimer.clear_summary()
            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data))
            training_time = training_timer.last_time
            train_op_scopes: list[dict[str, object]] = []
            if memory_export_enabled() and aa.is_main_process():
                phase_snapshots["after_training"] = cuda_memory_snapshot()
                train_op_scopes = ScopedMemoryTimer.collect_summary()
                if memory_jsonl_path is not None:
                    append_memory_jsonl(
                        memory_jsonl_path,
                        build_iter_memory_record(
                            iter_idx=i,
                            env_frames=env_frames * aa.get_world_size(),
                            num_envs=env.num_envs,
                            buffer_MiB=buffer_MiB,
                            phase_snapshots=phase_snapshots,
                            train_op_scopes=train_op_scopes or None,
                        ),
                    )
                ScopedMemoryTimer.clear_summary()

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
                should_upload = upload_interval > 0 and i % upload_interval == 0
                local_ckpt_path, archived = save(
                    policy,
                    archive_name=f"checkpoint_{i}" if should_upload else None,
                    upload_to_wandb=should_upload,
                )
                if archived is not None:
                    uploaded_ckpt_path = archived
                local_ckpt_iter = i

            if aa.is_main_process():
                remaining = (time.time() - t0) / (i + 1) * (total_iters - i)
                setproctitle(f"{proc_name} ETA {tqdm.format_interval(remaining)}")
                if profile_export_enabled() and profiling_jsonl_path is not None:
                    profile = ScopedTimer.collect_summary(depth=-1)
                    record = {
                        "iter": i,
                        "env_frames": info["env_frames"],
                        "backend": cfg.backend,
                        "num_envs": env.num_envs,
                        "train_every": cfg.algo.train_every,
                        "performance": {
                            "rollout_fps": info["performance/rollout_fps"],
                            "rollout_time": info["performance/rollout_time"],
                            "training_time": info["performance/training_time"],
                            "iter_time": info["performance/iter_time"],
                        },
                        **profile,
                    }
                    ScopedTimer.append_profiling_jsonl(profiling_jsonl_path, record)
                if profile_print_enabled() and (i % profile_print_every() == 0):
                    ScopedTimer.print_summary(clear=False, depth=3)
                ScopedTimer.clear_summary()
                print(
                    OmegaConf.to_yaml(
                        {k: v for k, v in info.items() if isinstance(v, (float, int))}
                    )
                )
                if local_ckpt_path is not None:
                    if local_ckpt_iter is not None:
                        print(
                            f"Local checkpoint (iter {local_ckpt_iter}): {local_ckpt_path}"
                        )
                    else:
                        print(f"Local checkpoint: {local_ckpt_path}")
                if uploaded_ckpt_path is not None:
                    print(f"Last uploaded checkpoint: {uploaded_ckpt_path}")
                env_extra = dict(env.extra)
                stats_ema = dict(env.stats_ema)
                if metrics_export_enabled() and run_dir is not None:
                    memory_snapshot = phase_snapshots.get("after_training")
                    last_algo_metrics = export_iteration_monitoring(
                        run_dir,
                        iter_idx=i,
                        env_frames=info["env_frames"],
                        info=info,
                        env_extra=env_extra,
                        stats_ema=stats_ema,
                        backend=cfg.backend,
                        num_envs=env.num_envs,
                        state="running",
                        memory_snapshot=memory_snapshot,
                    )
                info.update(env_extra)
                info.update(stats_ema)
                wandb_run.log(info)

    run_state: dict[str, str] = {}
    if aa.is_main_process():
        local_ckpt_path, uploaded_ckpt_path = save(
            policy, archive_name="checkpoint_final", upload_to_wandb=True
        )
        if cfg.eval:
            policy_eval = policy.get_rollout_policy("eval")
            eval_info, trajs, stats = evaluate(
                env, policy_eval, render=cfg.eval_render, seed=cfg.seed
            )
            eval_info["env_frames"] = env_frames
            wandb_run.log(eval_info)
        if metrics_export_enabled() and run_dir is not None:
            health, health_issues = assess_health(last_algo_metrics)
            write_run_status(
                run_dir / RUN_STATUS_FILENAME,
                state="completed",
                iter_idx=total_iters - 1,
                env_frames=env_frames * aa.get_world_size(),
                metrics=last_algo_metrics or {"env_frames": env_frames * aa.get_world_size()},
                health=health,
                health_issues=health_issues,
                backend=cfg.backend,
                num_envs=env.num_envs,
            )
        wandb.finish()
        print(f"Final checkpoint: {uploaded_ckpt_path}")
        run_state = {
            "checkpoint_path": uploaded_ckpt_path or local_ckpt_path,
            "run_dir": str(run_dir),
            "task": str(cfg.task.name),
            "algo": str(cfg.algo.name),
        }
        run_state_path = write_run_state(run_state, run_dir / RUN_STATE_FILENAME)
        print(f"Wrote run state to {run_state_path}")
        pipeline_dir = get_run_state_dir()
        if pipeline_dir is not None and pipeline_dir.resolve() != run_dir.resolve():
            write_run_state(run_state, pipeline_dir / RUN_STATE_FILENAME)
    return run_state


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: TrainConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
