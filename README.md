# Cross-View Action Consistency

Anonymous code release for **Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies**.



The main training config is:

```text
pi05_cross_view_action_consistency
```

Key settings:

| Setting | Value |
|---|---|
| Cross-view weight | `lambda_cv = 0.10` |
| Flow samples per pair | `cv_num_samples = 2` |
| CV loss mode | `multi_sample_asymmetric` implementation path |
| Gradient mode | bilateral, `cv_stopgrad_anchor = False` |
| Time distribution | `beta_2p0_3p0` |
| Shared noise/time across paired views | true |
| Pair spatial augmentation | `none` |
| Pair photometric augmentation | `independent` |
| Inference inputs | scene RGB + language + robot state |
| Wrist image | disabled in cross-view action consistency training and inference |

The clean-wrong control uses the same `cross_view_action_consistency` hyperparameters but deranges only the cross-view counterpart inside the loss:

```text
pi05_cross_view_action_consistency_clean_wrong
```

## Repository Layout

- `openpi/src/openpi/models/cross_view_action_consistency.py`: cross-view action consistency paired loss and model path.
- `openpi/src/openpi/models/cross_view_action_consistency_config.py`: cross-view action consistency model configuration.
- `openpi/src/openpi/training/config.py`: cross_view_action_consistency, ablation, clean-wrong, LIBERO, and Realman configs.
- `scripts/libero_pair_data/`: original LIBERO HDF5 audit, same-state camera rerendering, and LeRobot export.
- `scripts/cross_view_action_consistency/`: paired dataloader, pair augmentation, and multi-sample CV loss smoke checks.
- `scripts/real_robot/`: Realman paired-data export and norm-stat utilities.
- `assets/`: small normalization and camera metadata assets used by configs.

## Environment Setup

Install `uv` and Python 3.11, then create the main environment:

```bash
./install_env.sh --main
```

For LIBERO rerendering or LIBERO/LIBERO-Plus evaluation scripts, also create the simulator environment:

```bash
./install_env.sh --libero --fetch-libero
```

`--fetch-libero` clones the official LIBERO source into `openpi/third_party/libero`, which is required by original-state rerender scripts. If you need LIBERO-Plus evaluation, place a compatible LIBERO-Plus checkout at `LIBERO-plus/`; the installer will install it when that directory exists.

This release intentionally does not create a VGGT environment.

## Get Original LIBERO Data

Download the official original LIBERO HDF5 archives:

```bash
./.venv/bin/python scripts/libero_pair_data/download_libero_hdf5_original.py \
  --download-dir data/libero_hdf5_original \
  --datasets all
```

Audit the downloaded suites:

```bash
./.venv/bin/python scripts/libero_pair_data/audit_libero_hdf5_original.py \
  --libero-root data/libero_hdf5_original \
  --suite-dirs libero_spatial libero_object libero_goal libero_10 libero_90 \
  --output-dir results/libero_pair_audit
```

## Build Paired LIBERO Data

Run a small same-state rerender smoke test:

```bash
MUJOCO_GL=egl ./.venv-libero-plus/bin/python scripts/libero_pair_data/render_libero_multiview_states.py \
  --smoke-test \
  --suite libero_spatial \
  --max-tasks 1 \
  --libero-root data/libero_hdf5_original
```

Render nominal/perturbed scene-camera pairs:

```bash
MUJOCO_GL=egl ./.venv-libero-plus/bin/python scripts/libero_pair_data/render_libero_multiview_states.py \
  --libero-root data/libero_hdf5_original \
  --suite libero_spatial libero_object libero_goal libero_10 libero_90 \
  --max-pairs-per-suite 25000 \
  --output-dir data/rendered_libero_multiview \
  --results-dir results/libero_pair_audit
```

Export the rendered pairs to the LeRobot layout expected by the cross-view action consistency configs:

```bash
./.venv-libero-plus/bin/python scripts/libero_pair_data/export_to_lerobot.py \
  --manifest-train results/libero_pair_audit/libero_pair_manifest_train.jsonl \
  --manifest-val results/libero_pair_audit/libero_pair_manifest_val.jsonl \
  --hdf5-root data/libero_hdf5_original \
  --repo-root . \
  --output-dir data/libero_multiview_lerobot
```

Verify the export:

```bash
./.venv-libero-plus/bin/python scripts/libero_pair_data/export_to_lerobot.py \
  --verify-only \
  --output-dir data/libero_multiview_lerobot
```

The cross_view_action_consistency config reads paired data from `data/libero_multiview_lerobot` and uses normalization stats under `assets/pi05_cross_view_fm_only/anonymous/libero_multiview_lerobot/`.


## Train Main Method

Run the paired dataloader smoke before launching a long job:

```bash
./.venv/bin/python scripts/cross_view_action_consistency/smoke_pair_dataloader.py \
  --config-name pi05_cross_view_action_consistency \
  --batch-size 1 \
  --num-workers 0
```

For GPU training, export the XLA flags before calling `openpi/scripts/train.py`:

```bash
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL \
--xla_gpu_autotune_level=4 \
--xla_gpu_enable_cudnn_fmha=true"
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

Launch the main `cross_view_action_consistency` run:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
./.venv/bin/python openpi/scripts/train.py \
  pi05_cross_view_action_consistency \
  --exp_name cross_view_s42 \
  --overwrite
```

The config loads `gs://openpi-assets/checkpoints/pi05_base/params`, trains for 10k steps, and writes checkpoints under:

```text
checkpoints/pi05_cross_view_action_consistency/cross_view_s42/
```

To train the `cross_view_action_consistency` clean-wrong control:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
./.venv/bin/python openpi/scripts/train.py \
  pi05_cross_view_action_consistency_clean_wrong \
  --exp_name cross_view_clean_wrong_s42 \
  --overwrite
```

## LIBERO-Plus Evaluation

Use the eval config corresponding to the cross_view_action_consistency checkpoint and keep the local cross-view action consistency orientation explicit:

```bash
./.venv-libero-plus/bin/python scripts/run_libero_plus_parallel.py \
  --checkpoint-dir checkpoints/pi05_cross_view_action_consistency/cross_view_s42/9999 \
  --config-name pi05_cross_view_action_consistency_eval \
  --results-dir results/libero_plus/cross_view_s42 \
  --image-flip-mode flipud \
  --gpu-ids 0,1,2,3,4,5
```

For local cross-view action consistency rerendered checkpoints, use `--image-flip-mode flipud` rather than the default `both`.

Generated datasets, checkpoints, logs, and results are ignored by git.
