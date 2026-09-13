#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [[ ! -d third_party/libero/.git ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
  git -C third_party/libero checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
fi
if [[ ! -d third_party/libero-plus/.git ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git third_party/libero-plus
  git -C third_party/libero-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
fi
uv sync --frozen --extra cuda --extra sim
.venv/bin/python scripts/data/download_simulator_assets.py
printf '%s\n' 'Simulator sources and environment are ready. Use MUJOCO_GL=egl on a headless machine.'
