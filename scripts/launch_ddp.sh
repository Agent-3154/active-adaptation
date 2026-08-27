#!/bin/bash

# Thin wrapper around scripts/launch_ddp.py (torchrun / DDP).
#
# Usage (recommended with uv):
#   uv run --project <env-dir> ./scripts/launch_ddp.sh <gpu_ids> <script.py> [additional args...]
# Example:
#   uv run --project venv/isaac51 ./scripts/launch_ddp.sh 0,1 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/launch_ddp.py" "$@"
