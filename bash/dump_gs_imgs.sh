#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/train_ppo.py \
    task=SiriusAilabRoomNav \
    task.num_envs=16 \
    algo=ppo_staged2 \
    headless=true \
    wandb.mode=disabled \
    +dump_gs_imgs=true \
    +dump_gs_imgs_interval=1 \
    +dump_gs_imgs_dir=gs_dump \
    "$@"
