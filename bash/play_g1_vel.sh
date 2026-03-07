#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/play.py \
    task=G1AilabRoomVel \
    task.num_envs=1 \
    task.scene.n_repeats=1 \
    headless=false \
    task.scene.point_lights=true \
    "$@"
