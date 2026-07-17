"""Helpers for launching training scripts with ``torchrun`` (DDP)."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

log = logging.getLogger(__name__)


def parse_gpu_ids(gpu_ids: str) -> list[str]:
    """Parse a comma-separated GPU id string into a non-empty list."""
    ids = [part.strip() for part in gpu_ids.split(",") if part.strip()]
    if not ids:
        raise ValueError(f"Expected at least one GPU id, got {gpu_ids!r}")
    return ids


def find_free_port() -> int:
    """Bind an ephemeral localhost port and return its number."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def default_omp_num_threads(num_gpus: int) -> int:
    """Split available CPUs across GPU processes (at least 1 thread each)."""
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    total_cpus = os.cpu_count() or 1
    return max(1, total_cpus // num_gpus)


def build_torchrun_command(
    script: Path | str,
    script_args: Sequence[str],
    *,
    gpu_ids: str,
    python_executable: str | None = None,
    torchrun_executable: str | None = None,
    master_port: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build a ``torchrun`` argv and env for a multi-GPU script launch.

    Returns ``(cmd, env)`` where ``env`` is a full process environment including
    ``CUDA_VISIBLE_DEVICES`` and a default ``OMP_NUM_THREADS`` when unset.
    """
    gpus = parse_gpu_ids(gpu_ids)
    script_path = Path(script)
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    torchrun = torchrun_executable or shutil.which("torchrun")
    if torchrun is None:
        # Fallback: module form using the same interpreter as the caller.
        python = python_executable or sys.executable
        cmd = [
            python,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(gpus)}",
            f"--master_port={master_port if master_port is not None else find_free_port()}",
            str(script_path),
            *list(script_args),
        ]
    else:
        cmd = [
            torchrun,
            f"--nproc_per_node={len(gpus)}",
            f"--master_port={master_port if master_port is not None else find_free_port()}",
            str(script_path),
            *list(script_args),
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if "OMP_NUM_THREADS" not in env:
        env["OMP_NUM_THREADS"] = str(default_omp_num_threads(len(gpus)))
    return cmd, env


def run_ddp(
    script: Path | str,
    script_args: Sequence[str],
    *,
    gpu_ids: str,
    cwd: Path | str | None = None,
    env_updates: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Launch ``script`` under ``torchrun`` on the given GPUs."""
    cmd, env = build_torchrun_command(
        script,
        script_args,
        gpu_ids=gpu_ids,
        python_executable=python_executable,
    )
    if env_updates:
        env.update(env_updates)
    log.info("ddp launch: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
    )
