"""Local JSONL sidecars for agent-friendly training monitoring (no W&B API)."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

METRICS_EXPORT_ENV = "AA_METRICS_EXPORT"
RUN_STATUS_FILENAME = "run_status.yaml"
ALGO_DIAGNOSTICS_JSONL_FILENAME = "algo_diagnostics.jsonl"
ENV_STATS_JSONL_FILENAME = "env_stats.jsonl"

_ALGO_KEY_PREFIXES = ("actor/", "critic/", "performance/")
_EPISODE_KEY_PREFIX = "train/"
_HEALTH_METRIC_KEYS = (
    "performance/rollout_fps",
    "actor/approx_kl",
    "actor/grad_norm",
    "critic/grad_norm",
    "critic/explained_var",
)


def metrics_export_enabled() -> bool:
    return os.environ.get(METRICS_EXPORT_ENV, "1").lower() in {"1", "true", "yes"}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _is_finite_scalar(value: Any) -> bool:
    if not _is_scalar(value):
        return False
    return math.isfinite(float(value))


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def partition_training_metrics(
    info: dict[str, Any],
    *,
    env_extra: dict[str, Any] | None = None,
    stats_ema: dict[str, Any] | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Split a training ``info`` dict into algo diagnostics and env state buckets."""
    algo: dict[str, float | int] = {}
    episode: dict[str, float | int] = {}

    for key, value in info.items():
        if not _is_scalar(value):
            continue
        if key == "env_frames" or key.startswith(_ALGO_KEY_PREFIXES):
            algo[key] = value
        elif key.startswith(_EPISODE_KEY_PREFIX):
            episode[key] = value

    ema = {
        key: value
        for key, value in (stats_ema or {}).items()
        if _is_scalar(value)
    }
    extra = {
        key: value
        for key, value in (env_extra or {}).items()
        if _is_scalar(value)
    }
    env_stats = {
        "episode": episode,
        "ema": ema,
        "extra": extra,
    }
    return algo, env_stats


def assess_health(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    """Return ``(health, issues)`` where health is ``ok``, ``warn``, or ``fail``."""
    issues: list[str] = []
    for key in ("actor/grad_norm", "critic/grad_norm"):
        value = metrics.get(key)
        if value is None:
            continue
        if not _is_finite_scalar(value):
            issues.append(f"non-finite `{key}`")

    kl = metrics.get("actor/approx_kl")
    if _is_finite_scalar(kl) and abs(float(kl)) > 0.5:
        issues.append(f"large `actor/approx_kl` (last={float(kl):.3g})")

    explained_var = metrics.get("critic/explained_var")
    if _is_finite_scalar(explained_var) and float(explained_var) < 0.0:
        issues.append(f"negative `critic/explained_var` (last={float(explained_var):.3g})")

    if any("non-finite" in issue for issue in issues):
        return "fail", issues
    if issues:
        return "warn", issues
    return "ok", issues


def write_run_status(
    path: Path | str,
    *,
    state: str,
    iter_idx: int,
    env_frames: int,
    metrics: dict[str, Any],
    health: str,
    health_issues: list[str],
    pid: int | None = None,
    backend: str | None = None,
    num_envs: int | None = None,
    memory_snapshot: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: metrics[key]
        for key in _HEALTH_METRIC_KEYS
        if key in metrics and _is_scalar(metrics[key])
    }
    payload = {
        "state": state,
        "iter": iter_idx,
        "env_frames": env_frames,
        "pid": pid if pid is not None else os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "health_issues": health_issues,
        "metrics": summary,
    }
    if backend is not None:
        payload["backend"] = backend
    if num_envs is not None:
        payload["num_envs"] = num_envs
    if memory_snapshot:
        payload["memory"] = {
            key: memory_snapshot[key]
            for key in (
                "allocated_MiB",
                "reserved_MiB",
                "peak_allocated_MiB",
                "peak_reserved_MiB",
            )
            if key in memory_snapshot
        }
    OmegaConf.save(OmegaConf.create(payload), path)
    return path


def load_run_status(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Run status file not found: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def write_training_records(
    *,
    algo_path: Path | str,
    env_path: Path | str,
    iter_idx: int,
    env_frames: int,
    info: dict[str, Any],
    env_extra: dict[str, Any] | None = None,
    stats_ema: dict[str, Any] | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    algo_metrics, env_stats = partition_training_metrics(
        info,
        env_extra=env_extra,
        stats_ema=stats_ema,
    )
    append_jsonl(
        algo_path,
        {"iter": iter_idx, "env_frames": env_frames, "metrics": algo_metrics},
    )
    append_jsonl(
        env_path,
        {"iter": iter_idx, "env_frames": env_frames, **env_stats},
    )
    return algo_metrics, env_stats


def export_iteration_monitoring(
    run_dir: Path | str,
    *,
    iter_idx: int,
    env_frames: int,
    info: dict[str, Any],
    env_extra: dict[str, Any] | None = None,
    stats_ema: dict[str, Any] | None = None,
    backend: str | None = None,
    num_envs: int | None = None,
    state: str = "running",
    memory_snapshot: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Write JSONL sidecars and ``run_status.yaml`` for one training iteration."""
    if not metrics_export_enabled():
        return {}
    run_dir = Path(run_dir)
    algo_metrics, _ = write_training_records(
        algo_path=run_dir / ALGO_DIAGNOSTICS_JSONL_FILENAME,
        env_path=run_dir / ENV_STATS_JSONL_FILENAME,
        iter_idx=iter_idx,
        env_frames=env_frames,
        info=info,
        env_extra=env_extra,
        stats_ema=stats_ema,
    )
    health, health_issues = assess_health(algo_metrics)
    write_run_status(
        run_dir / RUN_STATUS_FILENAME,
        state=state,
        iter_idx=iter_idx,
        env_frames=env_frames,
        metrics=algo_metrics,
        health=health,
        health_issues=health_issues,
        backend=backend,
        num_envs=num_envs,
        memory_snapshot=memory_snapshot,
    )
    return algo_metrics
