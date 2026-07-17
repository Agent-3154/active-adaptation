#!/usr/bin/env python3
"""Launch a training script with torchrun (DDP).

Usage:
  python scripts/launch_ddp.py <gpu_ids> <script.py> [additional args...]

Example:
  uv run --project venv/isaac51 python scripts/launch_ddp.py 0,1 \\
    scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from active_adaptation.ddp_launch import run_ddp

FILE_PATH = Path(__file__).resolve().parent
REPO_ROOT = FILE_PATH.parent


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(
            "Usage: launch_ddp.py <gpu_ids> <script.py> [additional args...]",
            file=sys.stderr,
        )
        return 2

    gpu_ids = argv[0]
    script = Path(argv[1])
    if not script.is_file():
        script = FILE_PATH / argv[1]
    script_args = argv[2:]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    completed = run_ddp(
        script,
        script_args,
        gpu_ids=gpu_ids,
        cwd=REPO_ROOT,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
