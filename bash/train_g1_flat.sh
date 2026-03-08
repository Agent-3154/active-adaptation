#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=G1Flat \
    task.num_envs=16 \
    algo=ppo_blind \
    headless=false \
    wandb.mode=disabled \
    "$@"
