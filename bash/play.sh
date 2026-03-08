#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/eval_run.py \
    -r elgceben/g1_room/6arz0a78 \
    -p \
    --lights \
    --vis-rgb \
    "$@"
