# Cross-View Action Consistency

This repository contains the anonymous code release for **Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies**.

## Contents

- `openpi/src/openpi/models/v4_cv.py`: cross-view action-consistency model wrapper and losses.
- `openpi/src/openpi/models/v4_cv_config.py`: configuration dataclass for CVAC policies.
- `openpi/src/openpi/training/config.py`: training/evaluation configs used by the experiments.
- `scripts/v4/`: LIBERO pair construction, training launch helpers, and audit utilities.
- `scripts/real_robot/`: real-robot pair export and normalization-stat utilities.
- `assets/`: small normalization and camera-prior assets needed by the provided configs.

Large generated artifacts are intentionally not included: datasets, checkpoints, rendered frames, rollout records, logs, `wandb` runs, virtual environments, and result tables.

## Setup

The project uses Python 3.11 and `uv`.

```bash
uv sync
```

The code builds on the openpi package snapshot included under `openpi/`. External simulator datasets, LIBERO/LIBERO-plus assets, and trained checkpoints should be placed under local `data/` or `checkpoints/` paths, or supplied through command-line overrides.

## Basic Checks

```bash
uv run pytest openpi/src/openpi/models/v4_cv_test.py
uv run pytest openpi/src/openpi/training/config_test.py
```

## Notes

This release keeps only lightweight source code and small metadata assets. Legacy on-the-fly geometry extraction code and local virtual environments were removed from the public archive.
