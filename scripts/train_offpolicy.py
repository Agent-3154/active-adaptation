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
    export_iteration_monitoring,
    metrics_export_enabled,
    assess_health,
    write_run_status,
)
from active_adaptation.utils.profiling import ScopedTimer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "sac"},
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

    eval_render: bool = False
    """Render the environment during the final post-training evaluation."""
    log_interval: int = 64
    """Log statistics every N training iterations."""
    checkpoint_interval: int = 1600
    """Overwrite ``checkpoint_latest.pt`` every N training iterations (local only)."""
    upload_interval: int = 3200
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


def run(cfg: TrainConfig) -> dict[str, str]:
    """Train an off-policy policy and return checkpoint paths for downstream stages."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    wandb_run = None
    if aa.is_main_process():
        wandb_run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        wandb_run.config.update(OmegaConf.to_container(cfg))
        wandb_run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = wandb_run.name.split("-")[-1]
        wandb_run.name = f"{run_idx}-{default_run_name}"
        setproctitle(wandb_run.name)

        run_dir = Path(wandb_run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        wandb_run.save(str(cfg_save_path), policy="now")
        wandb_run.save(str(run_dir / "config.yaml"), policy="now")
    else:
        run_dir = None

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

    total_iters = cfg.total_frames // (aa.get_world_size() * env.num_envs)
    
    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval

    max_episode_length = cfg.task.max_episode_length
    log_interval = cfg.log_interval
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
    env_frames = 0
    last_algo_metrics: dict[str, float | int] = {}
    private_keys = None
    observation_keys = list(env.observation_spec.keys(True, True))

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)
    
    from tensordict import TensorDict
    def step_and_maybe_reset(tensordict: TensorDict) -> TensorDict:
        if tensordict.device != env.device:
            tensordict = tensordict.to(env.device)
        tensordict = env.step(tensordict)
        tensordict_ = env._step_mdp(tensordict)
        if hasattr(policy, "maybe_reset"):
            tensordict_ = policy.maybe_reset(tensordict_)
        tensordict_ = env.maybe_reset(tensordict_)
        return tensordict, tensordict_

    for stage in stages:
        policy.on_stage_start(stage, env)
        rollout_policy = policy.get_rollout_policy(mode="train")

        if aa.is_main_process():
            progress = tqdm(range(total_iters), desc=stage)
        else:
            progress = range(total_iters)

        last_log_episode_stats = 0
        t0 = time.time()

        for i in progress:
            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            with torch.no_grad():

                with (
                    set_exploration_type(ExplorationType.RANDOM),
                    ScopedTimer("policy_inference")
                ):
                    carry = rollout_policy(carry)

                with ScopedTimer("env_step") as timer:
                    # td, carry = env.step_and_maybe_reset(carry)
                    td, carry = step_and_maybe_reset(carry)
                    if not private_keys:
                        private_keys = [
                            key
                            for key in td.keys(True, True)
                            if isinstance(key, str) and key.startswith("_")
                        ]
                    td = td.exclude(*private_keys)
                    td["next"] = td["next"].exclude(*observation_keys)

            episode_stats.add(td)
            new_frames = td.numel()
            env_frames += new_frames
            train_info: dict = policy.step(td)

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

            if aa.is_main_process() and ((i % log_interval == 0) or len(train_info) > 0):
                info = {**train_info}
                info["env_frames"] = env_frames * aa.get_world_size()
                info["performance/rollout_fps"] = (1 / timer.last_time) * new_frames * aa.get_world_size()
                remaining = (time.time() - t0) / (i + 1) * (total_iters - i)
                setproctitle(f"{wandb_run.name} ETA {tqdm.format_interval(remaining)}")

                if (i - last_log_episode_stats >= max_episode_length) and len(episode_stats) > 0:
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                        info[key] = torch.mean(v.float()).item()
                    last_log_episode_stats = i
                
                ScopedTimer.print_summary(clear=True, depth=3)
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
                    )
                info.update(env_extra)
                info.update(stats_ema)
                wandb_run.log(info)

    run_state: dict[str, str] = {}
    if aa.is_main_process():
        local_ckpt_path, uploaded_ckpt_path = save(
            policy, archive_name="checkpoint_final", upload_to_wandb=True
        )
        policy_eval = policy.get_rollout_policy("eval")
        info, trajs, stats = evaluate(
            env, policy_eval, render=cfg.eval_render, seed=cfg.seed
        )
        info["env_frames"] = env_frames
        wandb_run.log(info)
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
        if cfg.algo.get("prior_data") is not None:
            run_state["prior_data"] = str(cfg.algo.prior_data)
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
