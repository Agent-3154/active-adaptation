"""
This script is used to play and visualize a policy in the environment.
"""

import time
import torch
import hydra
import itertools
import datetime
import copy
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.wandb import parse_checkpoint


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "ppo"},
    "_self_",
]


@dataclass
class IsaacAppConfig:
    """Isaac Lab AppLauncher settings (resolved from parent config)."""

    headless: str = "${..headless}"
    """Mirror ``headless``; passed to Isaac Lab's AppLauncher."""
    enable_cameras: str = "${..record_video}"
    """Mirror ``record_video``; enables camera sensors when recording."""


@dataclass
class PlayTaskOverride:
    """Play-specific overrides merged into the selected task config."""

    num_envs: int = 4
    """Number of parallel environments (kept small for interactive playback)."""


@dataclass
class PlayConfig:
    """Hydra root config for policy playback and visualization."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    """Hydra defaults list: task config, algo config, then this config."""
    hydra: HydraConf = field(default_factory=HydraConf)
    """Hydra runtime settings (output directory, etc.)."""
    headless: bool = False
    """Run with a visible GUI window (``false``) or headless (``true``)."""
    backend: str = "isaac"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for policy inference (e.g. ``cuda``, ``cpu``)."""
    record_video: bool = False
    """Record an MP4 of the rollout (Isaac backend only)."""
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""
    seed: int = 42
    """Random seed (offset by local rank in distributed runs)."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to a policy checkpoint; ``null`` starts from scratch."""
    export_policy: bool = False
    """Export the deploy policy to ONNX after loading."""
    discard_unused_obs: bool = True
    """Drop observation groups not listed in ``algo.in_keys``."""
    task: PlayTaskOverride = field(default_factory=PlayTaskOverride)
    """Task overrides applied on top of the selected task config."""


cs = ConfigStore.instance()
cs.store(
    name="play",
    node=PlayConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_play/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            )
        )
    )
)


FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu()
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"policy-{time_str}.onnx"
    export_onnx(deploy_policy, fake_input, str(path))


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: PlayConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)
    
    from active_adaptation.helpers import make_env_policy
    checkpoint = parse_checkpoint(cfg.checkpoint_path)
    env, policy = make_env_policy(cfg, checkpoint)
    
    if cfg.export_policy:
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_policy(env, policy, export_dir)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval").to(env.device)
    
    env.base_env.eval()
    carry = env.reset()
    
    assert not env.base_env.training

    timer = Timer(env.step_dt)

    # Optional: refresh from URL/wandb in background so play loop never blocks on updates
    if checkpoint is not None and checkpoint.remote:
        print("Starting background checkpoint refresh")
        checkpoint.start_background_refresh(interval_sec=60)

    # Optional video recording (Isaac backend only). This remains safe under
    # KeyboardInterrupt because the recorder is a context manager that flushes
    # buffered frames on exit.
    record_enabled = bool(cfg.get("record_video", False))
    video_dir = FILE_PATH / "videos"
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    video_path = video_dir / f"{cfg.task.name}-{time_str}.mp4"

    with env.get_recorder(video_path, enabled=record_enabled)as rec, \
        torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        try:
            for i in itertools.count():
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)

                if record_enabled:
                    rec.add_frame()

                if len(episode_stats) >= env.num_envs:
                    print("Step", i)
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        print(k, torch.mean(v).item())

                timer.sleep()
        except KeyboardInterrupt:
            print(f"Interrupted by user, video saved to: {video_path}" if record_enabled else "Interrupted by user.")
    
    env.close()


if __name__ == "__main__":
    main()

