#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/libero_multiview_lerobot_wrong}"

venv-libero-plus/bin/python scripts/v4/phase0A/export_to_lerobot.py \
  --manifest-train results/v4_pair_audit/libero_wrong_pair_manifest_train.jsonl \
  --manifest-val results/v4_pair_audit/libero_wrong_pair_manifest_val.jsonl \
  --hdf5-root data/libero_hdf5_original \
  --repo-root . \
  --output-dir "${OUT_DIR}"

venv-libero-plus/bin/python scripts/v4/phase0A/export_to_lerobot.py \
  --verify-only \
  --output-dir "${OUT_DIR}"
