#!/bin/bash

# Thin wrapper around scripts/launch_ddp.py (torchrun / DDP).
#
# Usage (recommended with uv):
#   uv run --project <env-dir> ./scripts/launch_ddp.sh <gpu_ids> <script.py> [additional args...]
# Example:
#   uv run --project venv/isaac51 ./scripts/launch_ddp.sh 0,1 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo

set -euo pipefail

GPU_IDS=$1
SCRIPT=$2
shift 2
UV_PROJECT=$(pwd)

if [ "$#" -gt 0 ]; then
    CANDIDATE_UV_PROJECT=$1
    if [ -d "$CANDIDATE_UV_PROJECT" ] || [[ "$CANDIDATE_UV_PROJECT" == */* && "$CANDIDATE_UV_PROJECT" != *=* ]]; then
        UV_PROJECT=$CANDIDATE_UV_PROJECT
        shift
    fi
fi

EXTRA_ARGS=("$@")

UV_RUN_ARGS=(--no-sync)
if [ "${AA_UV_SYNC:-0}" = "1" ]; then
    UV_RUN_ARGS=()
fi

UV_BIN=${UV_BIN:-$(command -v uv 2>/dev/null || true)}
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
    echo "uv executable not found; set UV_BIN or install uv in PATH" >&2
    exit 127
fi

# Count number of GPUs
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# Set a reasonable per-process OpenMP thread count unless provided.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    TOTAL_CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)
    OMP_NUM_THREADS=$(( TOTAL_CPUS / NUM_GPUS ))
    if [ "$OMP_NUM_THREADS" -lt 1 ]; then
        OMP_NUM_THREADS=1
    fi
    export OMP_NUM_THREADS
fi

# Find a free port
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# set CUDA_VISIBLE_DEVICES
# and launch torchrun
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$UV_BIN" --project "$UV_PROJECT" run "${UV_RUN_ARGS[@]}" torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$FREE_PORT" \
    "$SCRIPT" "${EXTRA_ARGS[@]}"
