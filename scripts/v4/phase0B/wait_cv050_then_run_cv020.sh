#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CV050_CONFIG="pi05_v4_pair_cv050"
CV050_EXP="phase0b_cv050_10k_s42"
CV050_DIR="checkpoints/${CV050_CONFIG}/${CV050_EXP}"
CV050_FINAL_STEP="${CV050_FINAL_STEP:-9999}"

CV020_CONFIG="pi05_v4_pair_cv020"
CV020_EXP="phase0b_cv020_10k_s42"
CV020_DIR="checkpoints/${CV020_CONFIG}/${CV020_EXP}"

POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="logs/phase0B"
mkdir -p "${LOG_DIR}"
WATCH_LOG="${LOG_DIR}/wait_cv050_then_cv020_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date +%F_%T)] $*" | tee -a "${WATCH_LOG}"
}

latest_numeric_checkpoint() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 0
  fi
  find "${dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/ {print}' \
    | sort -n \
    | tail -1
}

cv050_running() {
  pgrep -f "openpi/scripts/train.py ${CV050_CONFIG} --exp_name ${CV050_EXP}" >/dev/null
}

cv020_running() {
  pgrep -f "openpi/scripts/train.py ${CV020_CONFIG} --exp_name ${CV020_EXP}" >/dev/null
}

log "Watcher started. Waiting for ${CV050_CONFIG}/${CV050_EXP} final checkpoint ${CV050_FINAL_STEP}."
log "Watcher log: ${WATCH_LOG}"

while true; do
  latest="$(latest_numeric_checkpoint "${CV050_DIR}")"
  latest="${latest:-none}"

  if [[ -d "${CV050_DIR}/${CV050_FINAL_STEP}" ]] && ! cv050_running; then
    log "cv050 final checkpoint found at ${CV050_DIR}/${CV050_FINAL_STEP}, and no cv050 train process is running."
    break
  fi

  if cv050_running; then
    log "cv050 still running; latest checkpoint=${latest}. Sleeping ${POLL_SECONDS}s."
  else
    log "cv050 process not running yet final checkpoint missing; latest checkpoint=${latest}. Sleeping ${POLL_SECONDS}s."
  fi
  sleep "${POLL_SECONDS}"
done

if cv020_running; then
  log "Refusing to start cv020 because a cv020 train process is already running."
  exit 1
fi

existing_cv020="$(latest_numeric_checkpoint "${CV020_DIR}")"
if [[ -n "${existing_cv020:-}" ]]; then
  log "Refusing to overwrite existing cv020 checkpoint ${CV020_DIR}/${existing_cv020}."
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL --xla_gpu_autotune_level=4 --xla_gpu_enable_cudnn_fmha=true"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

TRAIN_LOG="${LOG_DIR}/cv020_after_cv050_$(date +%Y%m%d_%H%M%S)_prealloc098.log"
log "Starting ${CV020_CONFIG} exp=${CV020_EXP}. Train log: ${TRAIN_LOG}"

uv run openpi/scripts/train.py "${CV020_CONFIG}" --exp_name "${CV020_EXP}" --overwrite 2>&1 | tee -a "${TRAIN_LOG}"

log "cv020 training command exited with status ${PIPESTATUS[0]}."
