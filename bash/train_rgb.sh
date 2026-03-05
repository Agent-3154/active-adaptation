#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=SiriusAilabRoomNavRGB \
    task.num_envs=4 \
    algo=ppo_staged2 \
    headless=true \
    wandb.mode=disabled \
    task.scene.point_lights=true \
    task.command.vis_waypoints=true \
    +vis_gs_rgb=true \
    "$@"
