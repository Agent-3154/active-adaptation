#!/bin/bash

set -euo pipefail

# Usage:
#   scripts/launch_multinode_tmux.sh [--session NAME] [--repo-dir PATH] <node>... -- <script.py> [<uv_project_path>] [args...]
#
# Nodes can be plain hosts or host:gpu_ids. Plain hosts use $GPU_IDS, defaulting
# to all 8 GPUs. Each node starts one tmux session and runs one torchrun rank.

usage() {
    echo "Usage: $0 [--session NAME] [--repo-dir PATH] <node>... -- <script.py> [<uv_project_path>] [args...]"
    echo "Example: $0 --session hdmi-2x8 rp-h20-2 rp-h20-3 -- projects/hdmi/scripts/train.py venv/mjlab task=lafan"
}

quote() {
    printf "%q" "$1"
}

join_quoted() {
    local out=""
    local arg
    for arg in "$@"; do
        out+=" $(quote "$arg")"
    done
    printf "%s" "${out# }"
}

count_gpus() {
    local ids=$1
    local count=1
    local i
    if [ -z "$ids" ]; then
        echo 0
        return
    fi
    for ((i = 0; i < ${#ids}; i++)); do
        if [ "${ids:i:1}" = "," ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

SESSION_NAME="mn-$(date +%Y%m%d-%H%M%S)"
REPO_DIR=$(pwd)
DEFAULT_GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NODES=()
FORWARDED_ENV_VARS=(
    HF_HUB_OFFLINE
    HF_HUB_DISABLE_TELEMETRY
    HF_HUB_ETAG_TIMEOUT
    NCCL_IB_DISABLE
    NCCL_IB_HCA
    NCCL_IB_GID_INDEX
    NCCL_SOCKET_IFNAME
    NCCL_DEBUG
    PYTORCH_ALLOC_CONF
    ACTIVE_ADAPTATION_DISTRIBUTED_TIMEOUT_S
    AA_PRINT_PERF_INTERVAL
    AA_PRINT_METRICS_INTERVAL
    AA_PARAM_SYNC_ITERS
)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --session)
            SESSION_NAME=${2:?--session requires a value}
            shift 2
            ;;
        --repo-dir)
            REPO_DIR=${2:?--repo-dir requires a value}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            NODES+=("$1")
            shift
            ;;
    esac
done

if [ "${#NODES[@]}" -lt 1 ] || [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi

SCRIPT=$1
shift
UV_PROJECT=$REPO_DIR

if [ "$#" -gt 0 ]; then
    CANDIDATE_UV_PROJECT=$1
    if [ -d "$CANDIDATE_UV_PROJECT" ] || [[ "$CANDIDATE_UV_PROJECT" == */* && "$CANDIDATE_UV_PROJECT" != *=* ]]; then
        UV_PROJECT=$CANDIDATE_UV_PROJECT
        shift
    fi
fi

EXTRA_ARGS=("$@")
NNODES=${#NODES[@]}
FIRST_HOST=${NODES[0]%%:*}

MASTER_ADDR=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$FIRST_HOST" "hostname -I | awk '{print \$1; exit}'")
if [ -z "$MASTER_ADDR" ]; then
    echo "Could not resolve MASTER_ADDR from $FIRST_HOST." >&2
    exit 1
fi

MASTER_PORT=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$FIRST_HOST" \
    "python3 -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()'")
if [ -z "$MASTER_PORT" ]; then
    echo "Could not find a free MASTER_PORT on $FIRST_HOST." >&2
    exit 1
fi

EXPECTED_NPROC=""
HOSTS=()
GPU_LISTS=()
for node in "${NODES[@]}"; do
    host=${node%%:*}
    gpu_ids=$DEFAULT_GPU_IDS
    if [[ "$node" == *:* ]]; then
        gpu_ids=${node#*:}
    fi

    nproc=$(count_gpus "$gpu_ids")
    if [ "$nproc" -lt 1 ]; then
        echo "Node $node has an empty GPU list." >&2
        exit 2
    fi
    if [ -z "$EXPECTED_NPROC" ]; then
        EXPECTED_NPROC=$nproc
    elif [ "$EXPECTED_NPROC" -ne "$nproc" ]; then
        echo "torchrun requires the same GPU count on every node." >&2
        echo "Got $EXPECTED_NPROC for earlier nodes and $nproc for $node." >&2
        exit 2
    fi

    HOSTS+=("$host")
    GPU_LISTS+=("$gpu_ids")
done

echo "Launching $NNODES node(s) x $EXPECTED_NPROC process(es) per node"
echo "session=$SESSION_NAME master=$MASTER_ADDR:$MASTER_PORT repo=$REPO_DIR"

log_dir="outputs/multinode-tmux/$SESSION_NAME"
for rank in "${!HOSTS[@]}"; do
    host=${HOSTS[$rank]}
    gpu_ids=${GPU_LISTS[$rank]}
    rank_session="${SESSION_NAME}-r${rank}"
    log_path="$log_dir/rank${rank}-${host}.log"

    train_cmd=(
        uv --project "$UV_PROJECT" run --no-sync torchrun
        "--nnodes=$NNODES"
        "--nproc_per_node=$EXPECTED_NPROC"
        "--node_rank=$rank"
        "--master_addr=$MASTER_ADDR"
        "--master_port=$MASTER_PORT"
        "$SCRIPT"
        "${EXTRA_ARGS[@]}"
    )

    remote_cmd="set -euo pipefail"
    remote_cmd+="; cd $(quote "$REPO_DIR")"
    remote_cmd+="; mkdir -p $(quote "$log_dir")"
    remote_cmd+="; export PATH=\"\$HOME/.local/bin:/snap/bin:\$PATH\""
    remote_cmd+="; export CUDA_VISIBLE_DEVICES=$(quote "$gpu_ids")"
    for env_var in "${FORWARDED_ENV_VARS[@]}"; do
        if [ -n "${!env_var:-}" ]; then
            remote_cmd+="; export $env_var=$(quote "${!env_var}")"
        fi
    done
    remote_cmd+="; if [ -z \"\${OMP_NUM_THREADS:-}\" ]; then total_cpus=\$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc); export OMP_NUM_THREADS=\$(( total_cpus / $EXPECTED_NPROC )); if [ \"\$OMP_NUM_THREADS\" -lt 1 ]; then export OMP_NUM_THREADS=1; fi; fi"
    remote_cmd+="; echo host=\$(hostname) rank=$rank cuda_visible_devices=\$CUDA_VISIBLE_DEVICES omp_threads=\$OMP_NUM_THREADS master=$MASTER_ADDR:$MASTER_PORT"
    remote_cmd+="; set +e"
    remote_cmd+="; $(join_quoted "${train_cmd[@]}") 2>&1 | tee -a $(quote "$log_path")"
    remote_cmd+="; launch_status=\${PIPESTATUS[0]}"
    remote_cmd+="; set -e"
    remote_cmd+="; echo __AA_LAUNCH_EXIT_CODE__=\$launch_status | tee -a $(quote "$log_path")"
    remote_cmd+="; exit \$launch_status"

    echo "rank=$rank host=$host gpu_ids=$gpu_ids tmux=$rank_session log=$log_path"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "tmux new-session -d -s $(quote "$rank_session") bash -lc $(quote "$remote_cmd")"
done
