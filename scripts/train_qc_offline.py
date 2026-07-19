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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "sac"},
    "_self_",
]

class SchemaEnv:
    def __init__(self, schema, device="cpu"):
        self.observation_spec = schema["observation_spec"].to(device)
        self.action_spec = schema["action_spec"].to(device)
        self.reward_spec = schema["reward_spec"].to(device)
        self._fake_tensordict = schema["fake_tensordict"].to(device)
        self.num_envs = int(schema["num_envs"])
        self.device = torch.device(device)
        self.step_dt = schema["step_dt"]
        self.physics_dt = schema["physics_dt"]

    def fake_tensordict(self):
        return self._fake_tensordict.clone()

def load_fake_env(schema_path, device):
    schema = torch.load(schema_path, map_location="cpu", weights_only=False)
    return SchemaEnv(schema, device)

from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
from omegaconf import DictConfig
from active_adaptation.utils.wandb import parse_checkpoint


def make_policy(
    fake_env: SchemaEnv,
    algo_cfg: DictConfig,
    device: str,
    checkpoint_path: str | None = None
):
    # Parse checkpoint in parallel with environment creation.
    with ThreadPoolExecutor(max_workers=1) as executor:
        checkpoint_future = executor.submit(parse_checkpoint, checkpoint_path)
        
        policy_in_keys = algo_cfg.get("in_keys", None)
        if policy_in_keys is None:
            raise ValueError("Specify `in_keys` (e.g., `policy`, `priv`) in `cfg.algo`.")

        checkpoint = checkpoint_future.result()

    if checkpoint is not None:
        checkpoint.update()
    checkpoint_path = checkpoint.get_path() if checkpoint else None

    print(f"[Info]: Using checkpoint from: {checkpoint_path}")
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}

    # setup policy
    policy_cls = hydra.utils.get_class(algo_cfg._target_)
    print(f"Creating policy {policy_cls} on device {device}")
    policy = policy_cls.from_env(algo_cfg, fake_env, device=device)
    
    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])

    return policy


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
    """Hydra root config for training."""

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
    log_interval: int = 32
    """Log statistics every N training iterations."""
    checkpoint_interval: int = 400
    """Save a local checkpoint every N training iterations."""
    upload_interval: int = 3200
    """Upload a checkpoint to WandB every N training iterations."""

    seed: int = 42
    """Random seed (offset by local rank in distributed runs)."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to resume from; ``null`` trains from scratch."""
    discard_unused_obs: bool = True
    """Drop observation groups not listed in ``algo.in_keys``."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    """WandB logging configuration."""


    """global settings"""
    horizon_length: int = 4
    discount: float = 0.99
    env_schema_path: str = "/home/cv/zjx/active-adaptation/env_schemas/G1LocoFlat.pt"
    
    """offline settings"""
    offline_iters: int = 1
    offline_eval_interval: int = 100

    """online settings"""
    online_iters: int = 1_000
    online_eval_interval: int = 100


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

    from active_adaptation.helpers import make_env_policy
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

    ckpt_path = None

    # TODO: training
    # offline stage
    policy.on_stage_start(stage="offline", env=env)
    for i in tqdm(range(cfg.offline_iters)):
        offline_info = policy.step_offline()

        if cfg.offline_eval_interval == 0 or \
            i % cfg.offline_eval_interval == 0:
            # TODO: eval offline
            pass

    # online stage
    for i in tqdm(range(cfg.online_iters)):
        # TODO online training
        pass
    
    if aa.is_main_process():
        # ckpt_path = save(policy, "checkpoint_final")
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    exit(0)


if __name__ == "__main__":
    main()
