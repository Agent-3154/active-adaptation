"""Resolve ``algo`` from a training-run sidecar ``cfg.yaml`` next to a checkpoint.

Play / rollout / eval default to ``algo=from_checkpoint``. ``make_env_policy``
replaces that sentinel with the ``algo`` block from the run's ``cfg.yaml``.
An explicit ``algo=...`` override always wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from termcolor import colored

from active_adaptation.utils.wandb import parse_checkpoint

FROM_CHECKPOINT_ALGO_NAME = "from_checkpoint"


@dataclass
class FromCheckpointAlgoConfig:
    """Hydra stub so play/rollout/eval can omit a real ``algo=`` choice.

    ``make_env_policy`` replaces this with the checkpoint sidecar ``algo`` config.
    """

    _target_: str = (
        "active_adaptation.utils.checkpoint_cfg.FromCheckpointAlgoConfig"
    )
    name: str = FROM_CHECKPOINT_ALGO_NAME

    def get_class(self):
        raise RuntimeError(
            f"algo={FROM_CHECKPOINT_ALGO_NAME} was not resolved by make_env_policy. "
            "Provide checkpoint_path=... (with a sibling cfg.yaml) or pass an "
            "explicit algo=... (e.g. algo=ppo_symaug)."
        )


cs = ConfigStore.instance()
cs.store(FROM_CHECKPOINT_ALGO_NAME, node=FromCheckpointAlgoConfig, group="algo")


def is_from_checkpoint_algo(algo_cfg) -> bool:
    """True if ``algo_cfg`` is the ``from_checkpoint`` sentinel (dict or instance)."""
    if isinstance(algo_cfg, FromCheckpointAlgoConfig):
        return True
    name = OmegaConf.select(algo_cfg, "name")
    return name == FROM_CHECKPOINT_ALGO_NAME


def _algo_target_str(algo_cfg) -> str | None:
    if hasattr(algo_cfg, "_target_"):
        target = algo_cfg._target_
    else:
        target = OmegaConf.select(algo_cfg, "_target_", default=None)
    return target if isinstance(target, str) else None


def validate_algo_cfg_target(algo_cfg) -> None:
    """Require ``_target_`` to name a Hydra config dataclass, not a policy class."""
    if is_from_checkpoint_algo(algo_cfg):
        return
    target = _algo_target_str(algo_cfg)
    if target is None:
        raise ValueError(
            "algo config must set `_target_` to a config dataclass "
            "(e.g. active_adaptation.learning.ppo.ppo_symaug.PPOConfig)."
        )
    class_name = target.rsplit(".", 1)[-1]
    if class_name.endswith("Config") or class_name.endswith("Cfg"):
        return
    raise ValueError(
        f"algo._target_ must point to a config dataclass (*Config / *Cfg), not a "
        f"policy class (got {target!r}). Instantiate via hydra.utils.instantiate "
        f"on the config, then policy_cls = cfg.get_class(); policy_cls.from_env(cfg, …)."
    )


def resolve_policy_class(algo_cfg):
    """Instantiated algo config → policy class via ``get_class()``."""
    if not hasattr(algo_cfg, "get_class"):
        target = _algo_target_str(algo_cfg)
        raise TypeError(
            f"Algo config {type(algo_cfg).__name__} has no get_class() "
            f"(_target_={target!r}). Use a config dataclass, not a policy class."
        )
    return algo_cfg.get_class()


def find_run_cfg_yaml(checkpoint_file: Path) -> Path | None:
    """Locate training ``cfg.yaml`` next to a local checkpoint ``.pt`` file."""
    parent = checkpoint_file.parent
    for candidate in (
        parent / "cfg.yaml",
        parent / "files" / "cfg.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_algo_cfg_from_local_pt(checkpoint_file: Path | str) -> DictConfig:
    """Load the saved ``algo`` block from ``cfg.yaml`` next to a local ``.pt``."""
    pt_path = Path(checkpoint_file).expanduser().resolve()
    cfg_yaml = find_run_cfg_yaml(pt_path)
    if cfg_yaml is None:
        raise FileNotFoundError(
            f"No cfg.yaml next to checkpoint {pt_path}. "
            f"Expected {pt_path.parent / 'cfg.yaml'} (written by train_* scripts). "
            f"Pass an explicit algo=... instead of algo={FROM_CHECKPOINT_ALGO_NAME}."
        )

    saved = OmegaConf.load(cfg_yaml)
    if "algo" not in saved:
        raise KeyError(f"{cfg_yaml} has no 'algo' section")
    algo_cfg = saved.algo
    validate_algo_cfg_target(algo_cfg)
    total_iters = OmegaConf.select(saved, "total_iters")
    if total_iters:
        for key in ("entropy_decay_start", "entropy_decay_end"):
            value = OmegaConf.select(algo_cfg, key)
            if value is not None and float(value) > 1.0:
                algo_cfg[key] = float(value) / float(total_iters)
    print(
        colored(
            f"[Info]: Loaded algo={OmegaConf.select(algo_cfg, 'name')} from {cfg_yaml}",
            "green",
        )
    )
    return algo_cfg


def load_algo_cfg_from_checkpoint(checkpoint_spec: str) -> DictConfig:
    """Download/resolve ``checkpoint_spec`` and return its saved ``algo`` config."""
    checkpoint = parse_checkpoint(checkpoint_spec)
    if checkpoint is None:
        raise ValueError("checkpoint_path is empty")
    checkpoint.update()
    local_path = checkpoint.get_path()
    if not local_path:
        raise FileNotFoundError(
            f"Could not resolve checkpoint path for {checkpoint_spec!r}"
        )
    return load_algo_cfg_from_local_pt(local_path)
