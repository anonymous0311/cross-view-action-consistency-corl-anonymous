#!/usr/bin/env bash
# Wait for the Phase 0B FM-only screening run to finish, then launch the cv005
# config with the canonical XLA flag set (see project training notes §1).
#
# Usage:
#   bash scripts/v4/phase0B/launch_cv005_after_fm_only.sh [WAIT_PID]
#
# WAIT_PID is optional. If omitted, the script will pgrep for the running
# `train.py pi05_v4_pair_fm_only` process and wait on it.
#
# To detach (recommended so it survives ssh disconnect):
#   nohup bash scripts/v4/phase0B/launch_cv005_after_fm_only.sh >/tmp/cv005_launcher.log 2>&1 &
#   tail -f /tmp/cv005_launcher.log

set -euo pipefail
set -o pipefail

REPO_ROOT="."
cd "$REPO_ROOT"

# 1. Resolve which PID to wait on.
WAIT_PID="${1:-}"
if [[ -z "$WAIT_PID" ]]; then
    WAIT_PID=$(pgrep -f "train.py pi05_v4_pair_fm_only" | head -1 || true)
fi

if [[ -z "$WAIT_PID" ]]; then
    echo "[$(date '+%F %T')] No fm_only run detected. Launching cv005 immediately."
else
    echo "[$(date '+%F %T')] Waiting for PID $WAIT_PID (pi05_v4_pair_fm_only) to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date '+%F %T')] PID $WAIT_PID exited. Grace period 30s for clean GPU/checkpoint shutdown..."
    sleep 30
fi

# 2. Sanity check the FM-only checkpoint actually reached the final step.
# _PHASE0B_NUM_TRAIN_STEPS is 10_000 (config.py). If the run crashed early,
# bail out so we don't burn 6 GPUs on a screening pair that has no baseline.
FM_CKPT_DIR="$REPO_ROOT/checkpoints/pi05_v4_pair_fm_only/phase0b_fm_only_10k_s42"
if [[ ! -d "$FM_CKPT_DIR" ]]; then
    echo "[$(date '+%F %T')] WARNING: FM-only checkpoint dir not found at $FM_CKPT_DIR" >&2
    echo "                       Cannot verify FM-only completed. Aborting cv005 launch." >&2
    echo "                       To force launch anyway, comment out this guard block." >&2
    exit 1
fi
LAST_STEP=$(ls "$FM_CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
if [[ -z "$LAST_STEP" ]] || (( LAST_STEP < 9999 )); then
    echo "[$(date '+%F %T')] WARNING: FM-only last checkpoint step is '${LAST_STEP:-none}', expected ≥ 9999." >&2
    echo "                       FM-only run may have crashed. Aborting cv005 launch." >&2
    exit 1
fi
echo "[$(date '+%F %T')] FM-only completed at step $LAST_STEP. Proceeding to cv005."

# 3. Canonical XLA flags (see project training notes §1).
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL \
--xla_gpu_autotune_level=4 \
--xla_gpu_enable_cudnn_fmha=true"
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data/huggingface}"

# 4. Launch cv005.
LOG_DIR="$REPO_ROOT/logs/phase0B"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/phase0b_cv005_10k_s42_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Launching pi05_v4_pair_cv005 (exp_name=phase0b_cv005_10k_s42)"
echo "[$(date '+%F %T')] Log file: $LOG_FILE"
uv run openpi/scripts/train.py pi05_v4_pair_cv005 \
    --exp_name phase0b_cv005_10k_s42 \
    --num-train-steps 30000 \
    --overwrite 2>&1 | tee "$LOG_FILE"
