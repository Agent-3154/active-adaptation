#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/eval_run.py \
    -r elgceben/go2_room/6ok3pmxi \
    -p \
    --lights \
    --vis-rgb \
    "$@"
