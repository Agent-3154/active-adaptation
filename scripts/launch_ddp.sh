#!/bin/bash

# Usage (recommended with uv):
#   uv run --project <env-dir> ./scripts/launch_ddp.sh <gpu_ids> <script.py> [additional args...]
# Example:
#   uv run --project venv/isaac51 ./scripts/launch_ddp.sh 0,1 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_ids> <script.py> [additional args...]"
    exit 1
fi

GPU_IDS=$1
SCRIPT=$2
shift 2
EXTRA_ARGS="$@"

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

# Find a free port using the active Python environment.
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# set CUDA_VISIBLE_DEVICES
# and launch torchrun
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$FREE_PORT \
    $SCRIPT $EXTRA_ARGS