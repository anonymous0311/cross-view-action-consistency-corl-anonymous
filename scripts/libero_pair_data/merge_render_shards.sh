#!/usr/bin/env bash
# LIBERO pair-data.3 — Merge per-shard JSONL manifests into final combined files.
#
# Run this AFTER all 6 GPU shards have completed.
#
# Usage:
#   cd .
#   bash scripts/libero_pair_data/merge_render_shards.sh
#
# Produces:
#   results/libero_pair_audit/libero_pair_manifest_train.jsonl     (matched pairs)
#   results/libero_pair_audit/libero_pair_manifest_val.jsonl
#   results/libero_pair_audit/libero_wrong_pair_manifest_train.jsonl
#   results/libero_pair_audit/libero_wrong_pair_manifest_val.jsonl

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${REPO}/results/libero_pair_audit"
N_SHARDS=6

echo "Merging ${N_SHARDS} shards from ${RESULTS}/"
echo ""

# Check all shard files exist
MISSING=0
for SHARD in $(seq 0 $((N_SHARDS - 1))); do
    TAG=$(printf "shard%02d" "${SHARD}")
    for FNAME in \
        "libero_pair_manifest_train_${TAG}.jsonl" \
        "libero_pair_manifest_val_${TAG}.jsonl" \
        "libero_wrong_pair_manifest_train_${TAG}.jsonl" \
        "libero_wrong_pair_manifest_val_${TAG}.jsonl"; do
        if [[ ! -f "${RESULTS}/${FNAME}" ]]; then
            echo "  [MISSING] ${FNAME}"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [[ ${MISSING} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${MISSING} shard file(s) missing. Ensure all GPU workers finished."
    exit 1
fi

echo "All shard files present. Merging ..."
echo ""

# Merge each manifest type
for KIND in \
    "libero_pair_manifest_train" \
    "libero_pair_manifest_val" \
    "libero_wrong_pair_manifest_train" \
    "libero_wrong_pair_manifest_val"; do

    OUT="${RESULTS}/${KIND}.jsonl"
    > "${OUT}"   # clear / create

    TOTAL=0
    for SHARD in $(seq 0 $((N_SHARDS - 1))); do
        TAG=$(printf "shard%02d" "${SHARD}")
        SRC="${RESULTS}/${KIND}_${TAG}.jsonl"
        N=$(wc -l < "${SRC}")
        cat "${SRC}" >> "${OUT}"
        TOTAL=$((TOTAL + N))
        echo "  shard${SHARD}: ${N} rows  →  ${KIND}.jsonl"
    done
    echo "  → ${KIND}.jsonl  total: ${TOTAL} rows"
    echo ""
done

# Print summary
echo "============================================================"
echo "Merged manifests:"
for KIND in \
    "libero_pair_manifest_train" \
    "libero_pair_manifest_val" \
    "libero_wrong_pair_manifest_train" \
    "libero_wrong_pair_manifest_val"; do
    N=$(wc -l < "${RESULTS}/${KIND}.jsonl")
    echo "  ${KIND}.jsonl   →  ${N} pairs"
done
echo "============================================================"
echo ""
echo "Done. Next step: LIBERO pair-data.5 — LeRobot exporter."
