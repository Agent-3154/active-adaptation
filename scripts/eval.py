"""
Evaluate a trained policy and save metrics, trajectories, and stats to disk.
"""

import sys
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import torch
import termcolor
from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import active_adaptation as aa


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "ppo"},
    "_self_",
]

# Trajectory keys collected during evaluation (missing keys are skipped).
DEFAULT_EVAL_KEYS = [
    ("next", "stats"),
    ("next", "done"),
    ("next", "reward"),
    "value_obs",
    "value_priv",
    "value_adapt",
    "context_expert",
    "context_scale",
    "context_adapt",
    "context_adapt_scale",
    "action_kl",
]


@dataclass
class IsaacAppConfig:
    """Isaac Lab AppLauncher settings (resolved from parent config)."""

    headless: str = "${..headless}"
    """Mirror ``headless``; passed to Isaac Lab's AppLauncher."""
    enable_cameras: str = "${..eval_render}"
    """Mirror ``eval_render``; enables camera sensors when rendering eval video."""


@dataclass
class EvalTaskOverride:
    """Eval-specific overrides merged into the selected task config."""

    num_envs: int = 2048
    """Number of parallel environments for batched evaluation."""


@dataclass
class EvalConfig:
    """Hydra root config for policy evaluation."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    """Hydra defaults list: task config, algo config, then this config."""
    hydra: HydraConf = field(default_factory=HydraConf)
    """Hydra runtime settings (output directory, etc.)."""
    headless: bool = True
    """Run simulation without a rendering window."""
    backend: str = "isaac"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for policy inference (e.g. ``cuda``, ``cpu``)."""
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""
    eval_render: bool = False
    """Record a video during evaluation (saved by ``evaluate()``)."""
    seed: int = 0
    """Random seed for environment resets during evaluation."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to a policy checkpoint; ``null`` starts from scratch."""
    discard_unused_obs: bool = True
    """Drop observation groups not listed in ``algo.in_keys``."""
    task: EvalTaskOverride = field(default_factory=EvalTaskOverride)
    """Task overrides applied on top of the selected task config."""


cs = ConfigStore.instance()
cs.store(
    name="eval",
    node=EvalConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_eval/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            )
        )
    ),
)


FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _output_dir(cfg: EvalConfig) -> Path:
    """Hydra output dir when launched via CLI; fallback for programmatic callers."""
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    out = FILE_PATH / "eval" / str(cfg.task.name)
    out.mkdir(parents=True, exist_ok=True)
    return out


@hydra.main(config_path=str(CONFIG_PATH), config_name="eval", version_base=None)
def main(cfg: EvalConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=False)

    from active_adaptation.helpers import make_env_policy, evaluate

    env, policy = make_env_policy(cfg)
    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats = evaluate(
        env,
        policy_eval,
        render=cfg.eval_render,
        seed=cfg.seed,
        keys=DEFAULT_EVAL_KEYS,
    )

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    output_dir = _output_dir(cfg)

    torch.save(trajs, output_dir / f"trajs-{time_str}.pt")
    torch.save(stats, output_dir / f"stats-{time_str}.pt")
    OmegaConf.save(info, output_dir / f"metrics-{time_str}.yaml")

    print(termcolor.colored(OmegaConf.to_yaml(info), "light_yellow"))
    print(f"Saved eval artifacts to {output_dir}")

    env.close()


if __name__ == "__main__":
    main()
