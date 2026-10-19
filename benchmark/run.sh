#!/usr/bin/env bash
# Cross-machine rollout FPS and GPU-memory benchmark.
#
# Writes benchmark/runs/<hostname>-<date>/ (gitignored). Override with STAMP or OUT.
# One suite per entry in NUM_ENVS. A backend is skipped when its venv directory is absent.
#
#   NUM_ENVS="1024 4096" ./benchmark/run.sh
#   NUM_ENVS=512 ISAAC_PROJECT=venv/isaac60 ./benchmark/run.sh
#
# PPO's default train_every is 32. Set PPO_TRAIN_EVERY if that default changes.
set -u
cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-1024 4096}"
ITERS="${ITERS:-6}"
PPO_TRAIN_EVERY="${PPO_TRAIN_EVERY:-32}"
ISAAC_PROJECT="${ISAAC_PROJECT:-venv/isaac51}"
MJLAB_PROJECT="${MJLAB_PROJECT:-venv/mjlab}"
HOST="$(hostname -s 2>/dev/null || hostname)"
STAMP="${STAMP:-${HOST}-$(date +%Y-%m-%d)}"
OUT="${OUT:-$(pwd)/benchmark/runs/${STAMP}}"
mkdir -p "${OUT}"

{
  echo "host: ${HOST}"
  echo "date: $(date -Is)"
  echo "uname: $(uname -srm)"
  echo "num_envs: ${NUM_ENVS}"
  echo "iters: ${ITERS}"
  echo "ppo_train_every: ${PPO_TRAIN_EVERY}"
  echo "isaac_project: ${ISAAC_PROJECT}"
  echo "mjlab_project: ${MJLAB_PROJECT}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "${OUT}/machine.txt"

run_one() {
  local suite="$1"
  local name="$2"
  local project="$3"
  shift 3
  local dest="${OUT}/${suite}"
  mkdir -p "${dest}"
  if [[ ! -d "${project}" ]]; then
    echo "SKIP ${suite}/${name} missing ${project}" | tee -a "${OUT}/manifest.txt"
    return 0
  fi
  local log="${dest}/${name}.log"
  local smi="${dest}/${name}.smi"
  local frames=$((suite * PPO_TRAIN_EVERY * ITERS))
  echo "START ${suite}/${name} $(date -Is)" | tee -a "${OUT}/manifest.txt"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "${smi}"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) >> "${smi}" &
  local smi_pid=$!
  AA_PROFILE_PRINT=0 PYTHONUNBUFFERED=1 \
    uv run --project "${project}" python scripts/train_ppo.py \
      "$@" \
      algo=ppo \
      'algo.in_keys=[policy]' \
      "task.num_envs=${suite}" \
      "total_frames=${frames}" \
      wandb.mode=disabled \
      eval=false \
      checkpoint_interval=0 \
      upload_interval=0 \
      seed=0 \
      headless=true \
      > "${log}" 2>&1
  local code=$?
  kill "${smi_pid}" 2>/dev/null || true
  wait "${smi_pid}" 2>/dev/null || true
  python3 - "${log}" "${dest}/${name}" <<'PY'
import shutil, sys
from pathlib import Path
log, dst = Path(sys.argv[1]), Path(sys.argv[2])
src = None
for line in log.read_text(errors="replace").splitlines():
    if "Wrote run state to " in line:
        src = Path(line.split("Wrote run state to ", 1)[1].strip()).parent
dst.mkdir(parents=True, exist_ok=True)
if src and src.is_dir():
    for path in src.iterdir():
        if path.suffix == ".pt":
            continue
        shutil.copy2(path, dst / path.name)
    print(f"archived {src} -> {dst}")
else:
    print(f"no run dir in {log}")
PY
  echo "END ${suite}/${name} exit=${code} $(date -Is)" | tee -a "${OUT}/manifest.txt"
}

for suite in ${NUM_ENVS}; do
  echo "bench num_envs=${suite} iters=${ITERS}" | tee -a "${OUT}/manifest.txt"
  run_one "${suite}" g1-aa-mjlab "${MJLAB_PROJECT}" \
    task=G1/G1LocoFlat backend=mjlab task.robot.name=g1_waist_unlocked
  run_one "${suite}" g1-mjlab-xml-mjlab "${MJLAB_PROJECT}" \
    task=G1/G1LocoFlat backend=mjlab task.robot.name=g1_mjlab
  run_one "${suite}" a2-mjlab "${MJLAB_PROJECT}" \
    task=A2/A2LocoFlat backend=mjlab
  run_one "${suite}" spot-mjlab "${MJLAB_PROJECT}" \
    task=Spot/SpotLocoFlat backend=mjlab
  run_one "${suite}" g1-aa-isaac "${ISAAC_PROJECT}" \
    task=G1/G1LocoFlat backend=isaaclab
  run_one "${suite}" a2-isaac "${ISAAC_PROJECT}" \
    task=A2/A2LocoFlat backend=isaaclab
  run_one "${suite}" spot-isaac "${ISAAC_PROJECT}" \
    task=Spot/SpotLocoFlat backend=isaaclab
done

python3 benchmark/summarize.py --run-dir "${OUT}" | tee "${OUT}/report.md"
echo "BENCH_DONE ${OUT}" | tee -a "${OUT}/manifest.txt"
