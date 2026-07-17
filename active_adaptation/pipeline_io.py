"""Shared helpers for multi-stage experiment pipelines."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ARTIFACTS_FILENAME = "artifacts.json"
ARTIFACTS_ENV_VAR = "AA_ARTIFACTS_DIR"
PIPELINE_STATE_FILENAME = "pipeline_state.yaml"

_STAGE_REF_RE = re.compile(r"\$\{artifact\.([^.}]+)\.([^}]+)\}")


def get_artifacts_dir() -> Path | None:
    """Return the directory where the current stage should write ``artifacts.json``."""
    raw = os.environ.get(ARTIFACTS_ENV_VAR)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def write_stage_artifacts(
    artifacts: dict[str, Any],
    *,
    artifacts_dir: Path | str | None = None,
) -> Path:
    """Write stage outputs for the pipeline driver to consume."""
    target_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else get_artifacts_dir()
    if target_dir is None:
        raise ValueError(
            f"artifacts_dir is required when {ARTIFACTS_ENV_VAR} is unset"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / ARTIFACTS_FILENAME
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in artifacts.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s", path)
    return path


def read_stage_artifacts(artifacts_dir: Path | str) -> dict[str, Any]:
    """Load artifacts written by a completed pipeline stage."""
    path = Path(artifacts_dir).expanduser().resolve() / ARTIFACTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Stage artifacts not found: {path}")
    return json.loads(path.read_text())


def resolve_stage_overrides(
    overrides: list[str],
    stage_state: dict[str, dict[str, Any]],
) -> list[str]:
    """Replace ``${artifact.<name>.<key>}`` placeholders in Hydra CLI overrides."""

    def _replace(match: re.Match[str]) -> str:
        stage_name, key = match.group(1), match.group(2)
        try:
            value = stage_state[stage_name][key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown pipeline artifact reference: artifact.{stage_name}.{key}"
            ) from exc
        return str(value)

    return [_STAGE_REF_RE.sub(_replace, override) for override in overrides]
