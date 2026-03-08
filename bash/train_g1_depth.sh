#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=G1AilabRoomNavDepth \
    task.num_envs=1 \
    algo=ppo_geoloco \
    headless=true \
    wandb.mode=disabled \
    task.scene.point_lights=true \
    task.command.vis_waypoints=false \
    "$@"
