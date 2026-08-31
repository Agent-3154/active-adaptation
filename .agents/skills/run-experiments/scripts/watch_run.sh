#!/usr/bin/env bash
# Event-based local watcher for training runs (no agent tokens while polling).
#
# Usage:
#   watch_run.sh --run-dir /path/to/wandb/files --pid <train_pid> [options]
#
# Emits AGENT_WAKE_<purpose> JSON when check_run exits with a decision code.
# Exit codes from check_run:
#   0 gate_passed | 1 not_ready | 2 fail | 3 error | 4 complete

set -euo pipefail

RUN_DIR=""
PID=""
PURPOSE="experiment"
POLL_SEC=30
GATE_ITER=50
WARMUP_ITERS=3
MAX_KL=0.5
MIN_EXPLAINED_VAR=0.0
STUCK_MINUTES=10
HEARTBEAT_MIN=0
PYTHON="python3"

usage() {
  sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --pid) PID="$2"; shift 2 ;;
    --purpose) PURPOSE="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --gate-iter) GATE_ITER="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --max-kl) MAX_KL="$2"; shift 2 ;;
    --min-explained-var) MIN_EXPLAINED_VAR="$2"; shift 2 ;;
    --stuck-minutes) STUCK_MINUTES="$2"; shift 2 ;;
    --heartbeat-min) HEARTBEAT_MIN="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 3 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required" >&2
  usage
  exit 3
fi

emit_wake() {
  local kind="$1"
  local payload="$2"
  printf 'AGENT_WAKE_%s %s\n' "$PURPOSE" "$payload"
}

check_once() {
  local args=(
    -m active_adaptation.utils.check_run
    --run-dir "$RUN_DIR"
    --gate-iter "$GATE_ITER"
    --warmup-iters "$WARMUP_ITERS"
    --max-kl "$MAX_KL"
    --min-explained-var "$MIN_EXPLAINED_VAR"
    --stuck-minutes "$STUCK_MINUTES"
    --json
  )
  if [[ -n "$PID" ]]; then
    args+=(--pid "$PID")
  fi
  "$PYTHON" "${args[@]}"
}

last_heartbeat_epoch=0
while true; do
  set +e
  output="$(check_once)"
  code=$?
  set -e

  case "$code" in
    0|2|4)
      emit_wake "decision" "$output"
      exit 0
      ;;
    3)
      emit_wake "error" "$output"
      exit 3
      ;;
    1)
      if [[ "$HEARTBEAT_MIN" -gt 0 ]]; then
        now_epoch=$(date +%s)
        if (( now_epoch - last_heartbeat_epoch >= HEARTBEAT_MIN * 60 )); then
          emit_wake "heartbeat" "$output"
          last_heartbeat_epoch=$now_epoch
        fi
      fi
      sleep "$POLL_SEC"
      ;;
    *)
      emit_wake "unknown" "$output"
      exit "$code"
      ;;
  esac
done
