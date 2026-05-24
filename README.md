# Cross-View Action Consistency

Anonymous code release for **Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies**.

This repository contains source code, lightweight camera/norm-stat assets, and scripts. It intentionally excludes large generated artifacts: datasets, checkpoints, rendered frames, rollout records, logs, result tables, virtual environments, and W&B runs.

## Repository Layout

- `openpi/src/openpi/models/v4_cv.py`: cross-view action-consistency model and losses.
- `openpi/src/openpi/models/v4_cv_config.py`: CVAC model configuration.
- `openpi/src/openpi/training/config.py`: training and evaluation configs.
- `scripts/v4/phase0A/`: LIBERO HDF5 audit, same-state camera rerendering, and LeRobot export.
- `scripts/v4/phase0B/`: paired-dataloader smoke checks and launch helpers.
- `scripts/real_robot/`: real-robot pair export and normalization utilities.
- `assets/`: small normalization and camera metadata assets used by configs.

## Environment Setup

Install `uv` and Python 3.11, then create the main environment:

```bash
./install_env.sh --main
```

For LIBERO rerendering or LIBERO/LIBERO-plus evaluation scripts, also create the simulator environment:

```bash
./install_env.sh --libero --fetch-libero
```

`--fetch-libero` clones the official LIBERO source into `openpi/third_party/libero`, which is required by the original-state rerender scripts. If you need the LIBERO-plus evaluation scripts, place a compatible LIBERO-plus checkout at `LIBERO-plus/`; the installer will install it when that directory exists.

This public release does not install or use a VGGT environment.

## Get Original LIBERO Data

Download the official original LIBERO HDF5 archives:

```bash
./.venv/bin/python scripts/v4/phase0A/download_libero_hdf5_original.py \
  --download-dir data/libero_hdf5_original \
  --datasets all
```

Audit the downloaded suites:

```bash
./.venv/bin/python scripts/v4/phase0A/audit_libero_hdf5_original.py \
  --libero-root data/libero_hdf5_original \
  --suite-dirs libero_spatial libero_object libero_goal libero_10 libero_90 \
  --output-dir results/v4_pair_audit
```

## Build The Paired LIBERO Training Set

First run a small rerender smoke test:

```bash
MUJOCO_GL=egl ./.venv-libero-plus/bin/python scripts/v4/phase0A/render_libero_multiview_states.py \
  --smoke-test \
  --suite libero_spatial \
  --max-tasks 1 \
  --libero-root data/libero_hdf5_original
```

Render same-state nominal/perturbed image pairs:

```bash
MUJOCO_GL=egl ./.venv-libero-plus/bin/python scripts/v4/phase0A/render_libero_multiview_states.py \
  --libero-root data/libero_hdf5_original \
  --suite libero_spatial libero_object libero_goal libero_10 libero_90 \
  --max-pairs-per-suite 25000 \
  --output-dir data/rendered_libero_multiview \
  --results-dir results/v4_pair_audit
```

Export the rendered pairs to the LeRobot layout expected by the training config:

```bash
./.venv-libero-plus/bin/python scripts/v4/phase0A/export_to_lerobot.py \
  --manifest-train results/v4_pair_audit/libero_pair_manifest_train.jsonl \
  --manifest-val results/v4_pair_audit/libero_pair_manifest_val.jsonl \
  --hdf5-root data/libero_hdf5_original \
  --repo-root . \
  --output-dir data/libero_multiview_lerobot
```

Verify the export:

```bash
./.venv-libero-plus/bin/python scripts/v4/phase0A/export_to_lerobot.py \
  --verify-only \
  --output-dir data/libero_multiview_lerobot
```

The B6b LIBERO config reads the paired dataset from `data/libero_multiview_lerobot` and uses the included normalization stats under `assets/pi05_v4_pair_fm_only/anonymous/libero_multiview_lerobot/`.

## Start B6b Training

Run a paired-dataloader smoke check before starting a long job:

```bash
./.venv/bin/python scripts/v4/phase0B/smoke_v4_pair_dataloader.py \
  --config-name pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time \
  --batch-size 1 \
  --num-workers 0
```

Start the main B6b training run:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
./.venv/bin/python openpi/scripts/train.py \
  pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time \
  --exp_name b6b_s42 \
  --overwrite
```

The config loads the pi0.5 base checkpoint from `gs://openpi-assets/checkpoints/pi05_base/params`, trains for 10k steps, and writes checkpoints under:

```text
checkpoints/pi05_v4_pair_multi_sample_bilateral_cross_view010_action_biased_time/b6b_s42/
```

Generated datasets, checkpoints, logs, and results are ignored by git.

## CPU-Only Tests

Use these commands for reviewer-side sanity tests without touching GPUs:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENPI_ALLOW_CPU=1 \
./.venv/bin/python -m pytest -q openpi/src/openpi/models/v4_cv_test.py

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENPI_ALLOW_CPU=1 \
./.venv/bin/python -m pytest -q openpi/src/openpi/training/config_test.py
```

The tests exercise the cross-view loss code and config resolution. They do not require LIBERO data or checkpoints.
