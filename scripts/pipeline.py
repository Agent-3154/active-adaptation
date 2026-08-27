"""
Run multi-stage experiment pipelines without shell glue.

Each stage is launched as a fresh Python subprocess (required for Isaac Lab).
Stages write a flat ``run_state.yaml``; the driver merges keys into
``work_dir/run_state.yaml``.

Optional per-stage ``gpus`` (e.g. ``\"0,1\"``) launches that stage under
``torchrun`` via :mod:`active_adaptation.ddp_launch`.

To resume from completed stages, comment them out and seed::

    python scripts/pipeline.py run_state=/path/to/wandb/.../files/run_state.yaml

References use flat keys, e.g. ``${run_state.checkpoint_path}``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import hydra
from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from active_adaptation.ddp_launch import build_torchrun_command
from active_adaptation.pipeline_io import (
    RUN_STATE_FILENAME,
    load_run_state,
    resolve_run_state_overrides,
    write_run_state,
)

FILE_PATH = Path(__file__).resolve().parent
REPO_ROOT = FILE_PATH.parent
CONFIG_PATH = REPO_ROOT / "cfg"

log = logging.getLogger(__name__)

# Recipe must come after _self_ so it overrides the empty structured-config defaults.
PIPELINE_DEFAULTS = [
    "_self_",
    {"recipe": "a2_relabel_rlpd"},
]


@dataclass
class StageConfig:
    """One subprocess stage in a pipeline recipe."""

    name: str
    """Stable stage name used in logs and per-stage output paths."""
    script: str
    """Entry script filename under ``scripts/`` (e.g. ``train_ppo.py``)."""
    overrides: List[str] = field(default_factory=list)
    """Hydra CLI overrides passed to the stage script."""
    enabled: bool = True
    """Skip this stage when false (prefer commenting it out of the recipe instead)."""
    gpus: Optional[str] = None
    """Comma-separated GPU ids for DDP (e.g. ``\"0,1\"``). ``null`` runs a single process."""


@dataclass
class PipelineConfig:
    """Hydra root config for sequential experiment pipelines."""

    defaults: List[Any] = field(default_factory=lambda: PIPELINE_DEFAULTS)
    """Hydra defaults list: structured config, then recipe YAML."""
    name: str = "pipeline"
    """Pipeline label used in the work directory."""
    work_dir: str = "./outputs_pipeline/${name}/${now:%Y-%m-%d-%H-%M-%S}"
    """Root directory for ``run_state.yaml`` and per-stage outputs."""
    stages: List[StageConfig] = field(default_factory=list)
    """Ordered list of stages to run."""
    run_state: Optional[str] = None
    """Optional YAML path used to seed flat run state before any stage runs."""
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(dir="./outputs_pipeline/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
            job=JobConf(chdir=False),
        )
    )
    """Hydra runtime settings (output directory, chdir, etc.)."""


cs = ConfigStore.instance()
cs.store(name="stage", node=StageConfig)
cs.store(name="pipeline", node=PipelineConfig)


def _resolve_work_dir(cfg: PipelineConfig) -> Path:
    """Resolve work_dir without evaluating ${run_state.*} placeholders in overrides."""
    work_cfg = OmegaConf.create({"name": cfg.name, "work_dir": cfg.work_dir})
    OmegaConf.resolve(work_cfg)
    return Path(work_cfg.work_dir).expanduser().resolve()


def run_stage(
    stage: StageConfig,
    *,
    work_dir: Path,
    run_state: dict[str, Any],
    python_executable: str,
) -> dict[str, Any]:
    """Run one stage as a subprocess and return its written run state.

    Resolves ``${run_state.*}`` placeholders, launches ``scripts/<stage.script>``
    (optionally under ``torchrun`` when ``stage.gpus`` is set) with
    ``AA_RUN_STATE_DIR`` pointing at ``work_dir/stages/<name>/``, then loads that
    directory's ``run_state.yaml``.
    """
    script_path = FILE_PATH / stage.script
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage script not found: {script_path}")

    run_state_dir = work_dir / "stages" / stage.name
    run_state_dir.mkdir(parents=True, exist_ok=True)

    overrides = OmegaConf.to_container(stage.overrides, resolve=False)
    overrides = resolve_run_state_overrides(overrides, run_state)
    env_updates = {"AA_RUN_STATE_DIR": str(run_state_dir)}

    if stage.gpus:
        cmd, env = build_torchrun_command(
            script_path,
            overrides,
            gpu_ids=stage.gpus,
            python_executable=python_executable,
        )
        env.update(env_updates)
    else:
        cmd = [python_executable, str(script_path), *overrides]
        env = {**os.environ, **env_updates}

    log.info("[%s] running %s", stage.name, " ".join(cmd))
    log.info("[%s] run_state -> %s", stage.name, run_state_dir / RUN_STATE_FILENAME)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    stage_run_state = load_run_state(run_state_dir / RUN_STATE_FILENAME)
    log.info("[%s] done: %s", stage.name, stage_run_state)
    return stage_run_state


def run_pipeline(cfg: PipelineConfig, *, python_executable: str | None = None) -> dict[str, Any]:
    """Execute enabled stages and return the accumulated flat run-state map."""
    work_dir = _resolve_work_dir(cfg)
    python_executable = python_executable or sys.executable
    enabled = [s for s in cfg.stages if s.enabled]
    run_state_path = work_dir / RUN_STATE_FILENAME

    run_state: dict[str, Any] = {}
    if cfg.run_state:
        run_state = load_run_state(cfg.run_state)
        log.info("seeded run_state from %s (%s)", cfg.run_state, list(run_state))

    log.info("pipeline=%s work_dir=%s stages=%d", cfg.name, work_dir, len(enabled))
    write_run_state(run_state, run_state_path)

    stage_num = 0
    for stage in cfg.stages:
        if not stage.enabled:
            log.info("[%s] skipped (disabled)", stage.name)
            continue

        stage_num += 1
        log.info("--- stage %d/%d: %s ---", stage_num, len(enabled), stage.name)
        run_state.update(
            run_stage(
                stage,
                work_dir=work_dir,
                run_state=run_state,
                python_executable=python_executable,
            )
        )
        write_run_state(run_state, run_state_path)

    log.info("pipeline finished: %s", run_state_path)
    return run_state


@hydra.main(config_path=str(CONFIG_PATH), config_name="pipeline", version_base=None)
def main(cfg: PipelineConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OmegaConf.set_struct(cfg, False)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
