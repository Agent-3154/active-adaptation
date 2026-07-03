import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Any, Optional
from hydra.conf import HydraConf, RunDir, JobConf
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
from active_adaptation.learning.offpolicy import SAC
from omegaconf import OmegaConf
import torch
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
import logging
import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
import wandb
import datetime
from setproctitle import setproctitle

DEFAULTS = [
    {"task": "G1/G1LocoFlat"},
    {"algo": "sac"},
    "_self_"
]
FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
DATA_PATH = FILE_PATH.parent / "data"
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
    total_frame: int = 150_000_000

    eval_render: bool = False
    checkpoint_interval: int = 4
    upload_interval: int = 100

    seed: int = 42
    checkpoint_path: Optional[str] = None
    discard_unused_obs: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

cs = ConfigStore.instance()
cs.store(
    name="train",
    node=TrainConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_train/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            ),
            job=JobConf(chdir=True)
        )
    ),
)

from torchrl.envs import TransformedEnv
class StackingCollector:
    def __init__(self, env: TransformedEnv, steps: int, transitions: bool = False):
        self.env = env
        self.steps = steps
        self.transitions = transitions
        self.device = env.device
        self._observation_keys = list(env.observation_spec.keys(True, True))
        print("[INFO-EXTRA]observation_keys: ", self._observation_keys)

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

@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: TrainConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False) # enable dynamic field adding.

    aa.init(cfg, auto_rank=True)
    print(
        f'is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}'
    )

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags
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

    env, policy = make_env_policy(cfg)
    policy: SAC

    frame_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.total_frame // aa.get_world_size()
    total_frames = total_frames // frame_per_batch * frame_per_batch
    total_iters = total_frames // frame_per_batch

    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval

    max_episode_length = cfg.task.max_episode_length
    log_interval = (max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    carry = env.reset()
    transitions = cfg.algo.get("store_transitions", True)
    collector = StackingCollector(
        env,
        steps=cfg.algo.train_every,
        transitions=transitions
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
            critic= not transitions
        )

        if aa.is_main_process():
            progress = tqdm(range(total_iters), desc=stage)
        else:
            progress = range(total_iters)

        for i in progress:
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                data, carry = collector.collect(carry, rollout_policy)

                assert transitions, "Only support storing transitions."
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


            def should_save(i):
                if not aa.is_main_process():
                    return False
                return i % checkpoint_interval == 0 or i % upload_interval == 0
            def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
                run_dir = Path(run.dir)
            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)

if __name__ == "__main__":
    main()