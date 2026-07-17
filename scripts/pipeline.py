"""
Run multi-stage experiment pipelines without shell glue.

Each stage is launched as a fresh Python subprocess (required for Isaac Lab).
Stages communicate through per-stage ``artifacts.json`` files under the pipeline
work directory.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import hydra
from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from active_adaptation.pipeline_io import (
    PIPELINE_STATE_FILENAME,
    read_stage_artifacts,
    resolve_stage_overrides,
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
    name: str
    script: str
    overrides: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class PipelineConfig:
    defaults: List[Any] = field(default_factory=lambda: PIPELINE_DEFAULTS)
    name: str = "pipeline"
    work_dir: str = "./outputs_pipeline/${name}/${now:%Y-%m-%d-%H-%M-%S}"
    stages: List[StageConfig] = field(default_factory=list)
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            run=RunDir(dir="./outputs_pipeline/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
            job=JobConf(chdir=False),
        )
    )


cs = ConfigStore.instance()
cs.store(name="stage", node=StageConfig)
cs.store(name="pipeline", node=PipelineConfig)


def _resolve_work_dir(cfg: PipelineConfig) -> Path:
    """Resolve work_dir without evaluating ${artifact.*} placeholders in overrides."""
    work_cfg = OmegaConf.create({"name": cfg.name, "work_dir": cfg.work_dir})
    OmegaConf.resolve(work_cfg)
    return Path(work_cfg.work_dir).expanduser().resolve()


def _save_pipeline_state(work_dir: Path, state: dict[str, Any]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(state), work_dir / PIPELINE_STATE_FILENAME)


def run_stage(
    stage: StageConfig,
    *,
    work_dir: Path,
    stage_state: dict[str, dict[str, Any]],
    python_executable: str,
) -> dict[str, Any]:
    script_path = FILE_PATH / stage.script
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage script not found: {script_path}")

    artifacts_dir = work_dir / "stages" / stage.name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    overrides = resolve_stage_overrides(stage.overrides, stage_state)
    cmd = [python_executable, str(script_path), *overrides]
    env = {**os.environ, "AA_ARTIFACTS_DIR": str(artifacts_dir)}

    log.info("[%s] running %s", stage.name, " ".join(cmd))
    log.info("[%s] artifacts -> %s", stage.name, artifacts_dir)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    artifacts = read_stage_artifacts(artifacts_dir)
    artifacts.setdefault("stage_name", stage.name)
    artifacts.setdefault("script", stage.script)
    log.info("[%s] done: %s", stage.name, artifacts)
    return artifacts


def run_pipeline(cfg: PipelineConfig, *, python_executable: str | None = None) -> dict[str, Any]:
    work_dir = _resolve_work_dir(cfg)
    python_executable = python_executable or sys.executable
    enabled = [s for s in cfg.stages if s.enabled]

    log.info("pipeline=%s work_dir=%s stages=%d", cfg.name, work_dir, len(enabled))

    stage_state: dict[str, dict[str, Any]] = {}
    pipeline_state = {"name": cfg.name, "work_dir": str(work_dir), "stages": stage_state}
    _save_pipeline_state(work_dir, pipeline_state)

    stage_num = 0
    for stage in cfg.stages:
        if not stage.enabled:
            log.info("[%s] skipped (disabled)", stage.name)
            continue

        stage_num += 1
        log.info("--- stage %d/%d: %s ---", stage_num, len(enabled), stage.name)
        stage_state[stage.name] = run_stage(
            stage,
            work_dir=work_dir,
            stage_state=stage_state,
            python_executable=python_executable,
        )
        _save_pipeline_state(work_dir, pipeline_state)

    log.info("pipeline finished: %s", work_dir / PIPELINE_STATE_FILENAME)
    return pipeline_state


@hydra.main(config_path=str(CONFIG_PATH), config_name="pipeline", version_base=None)
def main(cfg: PipelineConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OmegaConf.set_struct(cfg, False)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
