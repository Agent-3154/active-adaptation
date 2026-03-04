#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=SiriusAilabRoom \
    task.num_envs=4 \
    algo=ppo \
    headless=false \
    wandb.mode=disabled \
    task.scene.point_lights=true \
    "$@"
