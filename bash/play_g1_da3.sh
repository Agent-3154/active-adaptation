#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Usage:
#   RUN_PATH=xxx bash bash/play_g1_da3.sh -p                      # interactive play (GUI)
#   RUN_PATH=xxx bash bash/play_g1_da3.sh --record                # headless Isaac sim robot video
#   RUN_PATH=xxx bash bash/play_g1_da3.sh --record-gs             # headless GS-rendered video
#   RUN_PATH=xxx bash bash/play_g1_da3.sh --record --record-gs    # both
#   RUN_PATH=xxx bash bash/play_g1_da3.sh --record --record-steps 5000

if [ -z "${RUN_PATH:-}" ]; then
    echo "Error: RUN_PATH not set."
    echo "Usage: RUN_PATH=<wandb_run_path> bash $0 [--record] [--record-steps N] [--record-res WxH]"
    exit 1
fi

python scripts/eval_run.py \
    -r "$RUN_PATH" \
    --lights \
    "$@"
