#!/usr/bin/env bash
# LIBERO pair-data.3 — Full LIBERO rerender across 6 GPUs in parallel.
#
# Renders ALL frames (100% sampling) from 4 eval suites once.
# This single dataset is used for both cross-view screening and confirmation.
#
# Expected output:
#   ~338,575 matched pairs  (~677K PNG images, ~40 GB)
#   ~2–4 hours on 6 GPUs
#
# Each GPU runs one shard (1/6 of 40 HDF5 files = 6–7 files per GPU).
# Logs go to logs/render_libero_multiview/render_shard_{0..5}.log
# When all 6 finish, run merge_render_shards.sh to combine manifests.
#
# Usage:
#   cd .
#   bash scripts/libero_pair_data/launch_render_6gpu.sh
#
# Monitor progress:
#   tail -f logs/render_libero_multiview/render_shard_0.log
#   find data/rendered_libero_multiview/ -name "*.png" | wc -l

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO}/venv-libero-plus/bin/python"
SCRIPT="${REPO}/scripts/libero_pair_data/render_libero_multiview_states.py"
LIBERO_ROOT="data/libero_hdf5_original"
OUTPUT_DIR="${REPO}/data/rendered_libero_multiview"
RESULTS_DIR="${REPO}/results/libero_pair_audit"
LOG_DIR="${REPO}/logs/render_libero_multiview"

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
    --libero-root    "${LIBERO_ROOT}"
    --suite          libero_spatial libero_object libero_goal libero_10
    --output-dir     "${OUTPUT_DIR}"
    --results-dir    "${RESULTS_DIR}"
    --img-size       256
    --timestep-sample-rate 1.0
    --views-per-state      1
    --val-demo-fraction    0.10
    --seed                 42
    --n-shards             6
)

echo "Launching 6 render workers ..."
echo ""

PIDS=()
for SHARD in 0 1 2 3 4 5; do
    GPU_ID=${SHARD}   # GPU i → shard i
    LOG="${LOG_DIR}/render_shard_${SHARD}.log"

    echo "  GPU ${GPU_ID}  shard ${SHARD}  →  ${LOG}"

    "${PYTHON}" "${SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --gpu-device-id "${GPU_ID}" \
        --shard-idx     "${SHARD}" \
        > "${LOG}" 2>&1 &

    PIDS+=($!)
done

# Save PIDs so you can check/kill them
printf '%s\n' "${PIDS[@]}" > "${LOG_DIR}/render_pids.txt"
echo ""
echo "All 6 workers launched.  PIDs: ${PIDS[*]}"
echo "PIDs saved to ${LOG_DIR}/render_pids.txt"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/render_shard_0.log"
echo ""
echo "After all finish, run:"
echo "  bash scripts/libero_pair_data/merge_render_shards.sh"
