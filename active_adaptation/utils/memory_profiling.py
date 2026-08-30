"""CUDA memory profiling for training loops (local JSONL sidecar)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, List, Optional

import torch

MEMORY_EXPORT_ENV = "AA_MEMORY_EXPORT"
MEMORY_SYNC_ENV = "AA_MEMORY_SYNC"
MEMORY_JSONL_FILENAME = "memory.jsonl"


def memory_export_enabled() -> bool:
    return os.environ.get(MEMORY_EXPORT_ENV, "1").lower() in {"1", "true", "yes"}


def memory_sync_enabled() -> bool:
    return os.environ.get(MEMORY_SYNC_ENV, "0").lower() in {"1", "true", "yes"}


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _maybe_sync() -> None:
    if _cuda_available() and memory_sync_enabled():
        torch.cuda.synchronize()


def cuda_memory_snapshot() -> dict[str, float]:
    """Current CUDA memory counters in MiB (empty dict when CUDA is unavailable)."""
    if not _cuda_available():
        return {}
    return {
        "allocated_MiB": torch.cuda.memory_allocated() / (1024**2),
        "reserved_MiB": torch.cuda.memory_reserved() / (1024**2),
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / (1024**2),
    }


def reset_cuda_peak_memory() -> None:
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


def append_memory_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _peak_phase(
    phases: dict[str, dict[str, float]],
    *,
    key: str = "peak_allocated_MiB",
) -> str | None:
    best_name: str | None = None
    best_value = -1.0
    for name, snapshot in phases.items():
        value = snapshot.get(key)
        if value is None:
            continue
        if float(value) > best_value:
            best_value = float(value)
            best_name = name
    return best_name


def build_iter_memory_record(
    *,
    iter_idx: int,
    env_frames: int,
    num_envs: int,
    buffer_MiB: float | None,
    phase_snapshots: dict[str, dict[str, float]],
    train_op_scopes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "iter": iter_idx,
        "env_frames": env_frames,
        "num_envs": num_envs,
        "phases": phase_snapshots,
        "peak_phase": _peak_phase(phase_snapshots),
    }
    if buffer_MiB is not None:
        record["buffer_MiB"] = buffer_MiB
    if train_op_scopes:
        record["train_op"] = train_op_scopes
    if phase_snapshots:
        record["cuda"] = phase_snapshots.get("after_training") or next(
            iter(phase_snapshots.values())
        )
    return record


@contextmanager
def memory_scope(name: str) -> Iterator[None]:
    """No-op when memory export is disabled."""
    if not memory_export_enabled():
        yield
        return
    with ScopedMemoryTracker(name):
        yield


class ScopedMemoryTimer:
    """Track allocated/reserved deltas for a named CUDA scope (train_op internals)."""

    _instances: dict[str, "ScopedMemoryTimer"] = {}

    def __new__(cls, name: str):
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.name = name
            instance.count = 0
            instance.delta_allocated_MiB = 0.0
            instance.delta_reserved_MiB = 0.0
            instance.peak_allocated_MiB = 0.0
            instance.peak_reserved_MiB = 0.0
            cls._instances[name] = instance
        return cls._instances[name]

    def __enter__(self) -> "ScopedMemoryTimer":
        if not memory_export_enabled() or not _cuda_available():
            self._disabled = True
            return self
        self._disabled = False
        _maybe_sync()
        self._start_allocated = torch.cuda.memory_allocated()
        self._start_reserved = torch.cuda.memory_reserved()
        self._start_peak_allocated = torch.cuda.max_memory_allocated()
        self._start_peak_reserved = torch.cuda.max_memory_reserved()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if getattr(self, "_disabled", True):
            return
        _maybe_sync()
        end_allocated = torch.cuda.memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        end_peak_allocated = torch.cuda.max_memory_allocated()
        end_peak_reserved = torch.cuda.max_memory_reserved()

        self.count += 1
        self.delta_allocated_MiB += (end_allocated - self._start_allocated) / (1024**2)
        self.delta_reserved_MiB += (end_reserved - self._start_reserved) / (1024**2)
        self.peak_allocated_MiB += (
            end_peak_allocated - self._start_peak_allocated
        ) / (1024**2)
        self.peak_reserved_MiB += (
            end_peak_reserved - self._start_peak_reserved
        ) / (1024**2)

    @classmethod
    def collect_summary(cls) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, timer in cls._instances.items():
            if timer.count == 0:
                continue
            rows.append(
                {
                    "path": name,
                    "count": timer.count,
                    "delta_allocated_MiB": timer.delta_allocated_MiB,
                    "delta_reserved_MiB": timer.delta_reserved_MiB,
                    "peak_allocated_MiB": timer.peak_allocated_MiB,
                    "peak_reserved_MiB": timer.peak_reserved_MiB,
                }
            )
        return sorted(rows, key=lambda row: row["peak_allocated_MiB"], reverse=True)

    @classmethod
    def clear_summary(cls) -> None:
        for timer in cls._instances.values():
            timer.count = 0
            timer.delta_allocated_MiB = 0.0
            timer.delta_reserved_MiB = 0.0
            timer.peak_allocated_MiB = 0.0
            timer.peak_reserved_MiB = 0.0


# Backwards-compatible alias used in docs/skills.
ScopedMemoryTracker = ScopedMemoryTimer
