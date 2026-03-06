#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=G1AilabRoomNav \
    task.num_envs=16 \
    algo=ppo_staged2 \
    headless=false \
    wandb.mode=disabled \
    task.scene.point_lights=true \
    task.command.vis_waypoints=true \
    "$@"
