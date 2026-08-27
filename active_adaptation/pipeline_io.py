"""Shared helpers for multi-stage experiment pipelines."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

RUN_STATE_FILENAME = "run_state.yaml"
RUN_STATE_ENV_VAR = "AA_RUN_STATE_DIR"

_RUN_STATE_REF_RE = re.compile(r"\$\{run_state\.([^}]+)\}")


def get_run_state_dir() -> Path | None:
    """Return ``Path(AA_RUN_STATE_DIR)`` when the env var is set, else ``None``."""
    raw = os.environ.get(RUN_STATE_ENV_VAR)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _to_plain(data: Any) -> Any:
    """Convert Paths and nested containers to YAML-friendly plain values."""
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): _to_plain(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_plain(value) for value in data]
    return data


def write_run_state(run_state: dict[str, Any], path: Path | str) -> Path:
    """Write a flat ``{key: value}`` mapping to the YAML file at ``path``."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(_to_plain(run_state)), path)
    log.info("wrote %s", path)
    return path


def load_run_state(path: Path | str) -> dict[str, Any]:
    """Load a flat ``{key: value}`` mapping from the YAML file at ``path``."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Run state file not found: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return _to_plain(data)


def resolve_run_state_overrides(
    overrides: list[str],
    run_state: dict[str, Any],
) -> list[str]:
    """Replace ``${run_state.<key>}`` placeholders in Hydra CLI overrides."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return str(run_state[key])
        except KeyError as exc:
            raise KeyError(
                f"Unknown pipeline run-state reference: run_state.{key}"
            ) from exc

    return [_RUN_STATE_REF_RE.sub(_replace, override) for override in overrides]
