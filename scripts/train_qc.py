import torch
import hydra
import wandb
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Optional
from collections import OrderedDict
from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir, JobConf
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from setproctitle import setproctitle
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa

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
    headless: bool = "${..headless}"
    enable_cameras: bool = "${..eval_render}"


@dataclass
class WandbConfig:
    name: str = "${..exp_name}/${now:%m-%d}_${now:%H-%M}"
    job_type: str = "train"
    project: str = "${oc.select:task.project,active_adaptation}"
    mode: str = "online"
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)

    headless: bool = True
    exp_name: str = "${oc.select:task.name,test}-${oc.select:algo.name,none}"
    backend: str = "isaac"
    device: str = "cuda"

    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)

    eval_render: bool = False
    checkpoint_interval: int = -1

    seed: int = 42
    checkpoint_path: Optional[str] = None
    discard_unused_obs: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

    offline_iters: int = 10_000
    offline_eval_interval: int = -1

    online_iters: int = 10_000
    online_eval_interval: int = 1000

    log_interval: int = 100


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


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: TrainConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, "
        f"local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
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
    else:
        run = None
        run_dir = None

    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.utils.helpers import EpisodeStats
    from active_adaptation.utils.profiling import ScopedTimer

    env, policy = make_env_policy(
        task_cfg=cfg.task,
        algo_cfg=cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        discard_unused_obs=cfg.discard_unused_obs,
        checkpoint_path=cfg.checkpoint_path,
    )

    assert env.training

    max_episode_length = cfg.task.max_episode_length
    log_interval = cfg.log_interval

    stats_keys = [
        k for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(checkpoint_name: str, *, upload_to_wandb: bool = True):
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
        logging.info(
            "Saved checkpoint to %s%s", ckpt_path,
            " (wandb)" if upload_to_wandb else "",
        )
        return str(ckpt_path)

    # ── offline stage ────────────────────────────────────────────────
    policy.on_stage_start(stage="offline", env=env)
    ckpt_path = None

    pbar = tqdm(range(cfg.offline_iters), desc="offline", unit="step")
    for i in pbar:
        info = policy.step_offline()

        if aa.is_main_process():
            if i % log_interval == 0 or len(info) > 0:
                run.log(info, step=i)

            if cfg.checkpoint_interval > 0 and i % cfg.checkpoint_interval == 0:
                ckpt_path = save(f"checkpoint_{i:06d}")

            if (
                cfg.offline_eval_interval > 0
                and i % cfg.offline_eval_interval == 0
                and i != 0
            ):
                with set_exploration_type(ExplorationType.MODE):
                    policy_eval = policy.get_rollout_policy("eval")
                    eval_info, _, _ = evaluate(env, policy_eval, seed=cfg.seed + i)
                run.log(eval_info, step=i)
                pbar.set_postfix({
                    "q_loss": info.get("critic/q_loss", 0),
                    "a_loss": info.get("actor/loss", 0),
                    "eval_ret": eval_info.get("eval/return", 0),
                })

    # ── online stage ─────────────────────────────────────────────────
    if cfg.online_iters > 0:
        policy.on_stage_start(stage="online", env=env)
        rollout_policy = policy.get_rollout_policy("train")

        carry = env.reset()
        env_frames = 0

        if aa.is_main_process():
            pbar = tqdm(range(cfg.online_iters), desc="online", unit="step")
        else:
            pbar = range(cfg.online_iters)

        observation_keys = list(env.observation_spec.keys(True, True))
        private_keys = None
        last_log_episode_stats = 0

        for i in pbar:
            with torch.no_grad():
                with (
                    set_exploration_type(ExplorationType.RANDOM),
                    ScopedTimer("policy_inference"),
                ):
                    carry = rollout_policy(carry)

                with ScopedTimer("env_step") as timer:
                    td, carry = env.step_and_maybe_reset(carry)
                    if private_keys is None:
                        private_keys = [
                            key for key in td.keys(True, True)
                            if isinstance(key, str) and key.startswith("_")
                        ]
                    td = td.exclude(*private_keys)
                    td["next"] = td["next"].exclude(*observation_keys)

            episode_stats.add(td)
            new_frames = td.shape[0]
            env_frames += new_frames
            info = policy.step(td)

            if aa.is_main_process():
                step = cfg.offline_iters + i
                if i % log_interval == 0 or len(info) > 0:
                    log_info = {**info}
                    log_info["env_frames"] = env_frames * aa.get_world_size()
                    log_info["performance/rollout_fps"] = (
                        1.0 / timer.last_time * new_frames * aa.get_world_size()
                    )

                    if (
                        i - last_log_episode_stats >= max_episode_length
                        and len(episode_stats) > 0
                    ):
                        for k, v in sorted(episode_stats.pop().items(True, True)):
                            key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                            log_info[key] = torch.mean(v.float()).item()
                        last_log_episode_stats = i

                    ScopedTimer.print_summary(clear=True, depth=3)
                    log_info.update(getattr(env, "extra", {}))
                    log_info.update(getattr(env, "stats_ema", {}))
                    run.log(log_info, step=step)

                if cfg.checkpoint_interval > 0 and i % cfg.checkpoint_interval == 0:
                    ckpt_path = save(f"checkpoint_{step:06d}")

                if (
                    cfg.online_eval_interval > 0
                    and i % cfg.online_eval_interval == 0
                    and i != 0
                ):
                    with set_exploration_type(ExplorationType.MODE):
                        policy_eval = policy.get_rollout_policy("eval")
                        eval_info, _, _ = evaluate(
                            env, policy_eval, seed=cfg.seed + i,
                        )
                    run.log(eval_info, step=step)

        total_iters = cfg.offline_iters + cfg.online_iters
    else:
        total_iters = cfg.offline_iters

    # ── final eval & save ────────────────────────────────────────────
    if aa.is_main_process():
        with set_exploration_type(ExplorationType.MODE):
            policy_eval = policy.get_rollout_policy("eval")
            final_info, _, _ = evaluate(
                env, policy_eval, seed=cfg.seed + 9999, render=cfg.eval_render,
            )
        run.log(final_info, step=total_iters)
        ckpt_path = save("checkpoint_final")
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
        print(f"Final eval return: {final_info.get('eval/return', 'N/A')}")

    exit(0)


if __name__ == "__main__":
    main()
