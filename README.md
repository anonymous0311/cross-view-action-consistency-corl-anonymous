# Action-Flow Consistency Across Views

**Scene-Camera Robustness for Flow-Based VLA Policies**
Anonymous code for the ICRA submission.

We train a π0.5 policy with two scene-camera observations of the same state. Both views retain the same language, proprioception, and action label. In addition to flow matching, a bilateral consistency loss aligns their predicted action velocities at shared flow coordinates. Deployment uses one scene image, language, and proprioception.

## Installation

Use Linux, Python 3.11, and `uv`. GPU execution uses JAX; PyTorch is used by the data loader. The locked CUDA extra uses CUDA 13 and requires a compatible NVIDIA driver. On the validation machine we used an RTX PRO 6000 GPU.

```bash
git clone https://github.com/anonymous0311/cross-view-action-consistency-corl-anonymous.git
cd cross-view-action-consistency-corl-anonymous
uv sync --frozen --extra cuda
source .venv/bin/activate
bash scripts/setup_simulator.sh
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The simulator setup fetches pinned LIBERO and LIBERO-Plus sources into `third_party/` and downloads the official LIBERO-Plus asset archive (about 6.4 GB; allow about 16 GB for the archive and extracted assets). For CPU objective tests only, use `uv sync --frozen` without simulator setup. Initial training downloads the public π0.5 base parameters and tokenizer; allow disk space and network access for these assets.

## Prepare paired demonstrations

```bash
python scripts/data/download_libero_hdf5_original.py --download-dir data/libero_original
python scripts/data/render_libero_multiview_states.py \
  --libero-root data/libero_original --output-dir data/rendered --seed 42
python scripts/data/export_to_lerobot.py \
  --manifest data/rendered/pairs.jsonl --output-dir data/libero_pairs
```

The four suites are LIBERO-Spatial, Object, Goal, and Long (`libero_10`): 40 tasks and 2,000 demonstrations. The paper uses 338,575 paired states. A perturbed camera is sampled once per episode and held fixed; every stored state is rendered at the nominal and perturbed views. The exporter reads 10-step action chunks directly from the original demonstration and repeats the last action at an episode boundary. Images are upright and resized to 224×224 at the policy input. There is no spatial augmentation.

Rendering and export accept `--resume`. Keep the same inputs, output directory, seed, and support setting when resuming. Use `--max-tasks 1 --max-episodes 1 --max-frames 4` on the renderer for a small installation check; omit these limits for training data.

LIBERO normalization statistics are included in `assets/libero/norm_stats.json`. To compute statistics for a new dataset:

```bash
python scripts/compute_norm_stats.py --dataset data/libero_pairs --output-dir assets/libero
```

## Train

```bash
python scripts/train.py paired_cv --exp-name seed42 --seed 42
python scripts/train.py paired_fm --exp-name seed42 --seed 42
```

Each recipe runs 10,000 updates with a global batch of 192 pairs, a 10-action horizon, and the π0.5 base initialization. Paired recipes use two independent flow samples per pair, `t ~ Beta(2,3)`, and shared noise/time within each pair. Flow matching averages over both views and all 32 padded action dimensions; consistency averages over the 7 commanded dimensions. The consistency coefficient ramps linearly to 0.10 over the first 500 updates. Both branches receive gradients. Color jitter is independent across views, with brightness 0.3, contrast 0.4, and saturation 0.5.

Checkpoints are saved under `checkpoints/<recipe>/<exp-name>/<completed_updates>/`; the final checkpoint is `10000`. They include normalization assets and optimizer state. Resume with the same command plus `--resume`. Set `--fsdp-devices` to the number of visible GPUs used for parameter sharding. Batch size must be divisible by the device count. The small execution check below does not establish the memory requirement of full-model training at batch 192.

| Recipe | Training objective / inputs |
| --- | --- |
| `nominal_only` | Nominal scene image, ordinary flow matching |
| `mixed_camera` | Both scene views treated as individual training observations |
| `paired_fm` | Paired flow matching, no consistency |
| `paired_cv` | Paired flow matching + bilateral velocity consistency |
| `shuffled` | Deranged consistency partners; supervised labels unchanged |
| `action_consistency` | Consistency weighted by `t²`, coefficient 0.50 |
| `shared_jitter` | Identical color jitter for the two views |
| `coverage_05`, `coverage_25` | Consistency on task-stratified 5% / 25% episode sets |
| `restricted_fm`, `restricted_cv` | Restricted camera-support experiment |

Use seeds 42, 43, and 44 for the three-seed main comparison. See [experiment protocols](docs/experiments.md) for coverage, restricted support, and generated-view inputs, and [real-robot data](docs/real_robot.md) for timestamp alignment and hardware recipes.

Paths can be set without editing code:

| Environment variable | Meaning |
| --- | --- |
| `AFCV_DATASET` | Local exported LeRobot dataset directory |
| `AFCV_BASE_PARAMS` | π0.5 base `params/` directory or public URL |
| `AFCV_NORM_DIR` | Parent of the `libero/` or `robot/` normalization directory |
| `AFCV_MEMBERSHIP` | Episode membership JSON for coverage experiments |
| `LIBERO_ROOT`, `LIBERO_PLUS_ROOT` | Override simulator source directories |

## Evaluate and serve

```bash
python scripts/evaluate.py --benchmark nominal \
  --checkpoint-dir checkpoints/paired_cv/seed42/10000 --output-dir results/seed42/nominal
python scripts/evaluate.py --benchmark camera \
  --checkpoint-dir checkpoints/paired_cv/seed42/10000 --output-dir results/seed42/camera
```

Nominal evaluation runs 40 tasks × 50 initial states = 2,000 trials. Camera evaluation runs 1,599 LIBERO-Plus camera tasks × 3 initial states = 4,797 trials. Each plan predicts 10 actions and executes 5 before replanning at 20 Hz. Both evaluations use one scene camera and no wrist image. Results are written to `trials.jsonl` and `summary.json`. Add `--resume` to continue completed task groups. For an execution check, use `--max-tasks 1 --trials 1 --max-steps 5`. Parallel workers can use `--num-shards N --shard-index I`, distinct output directories, and separate visible GPUs.

```bash
python scripts/serve.py --checkpoint-dir checkpoints/paired_cv/seed42/10000
```

The websocket server listens on `127.0.0.1:8000`. The included `openpi_client.websocket_client_policy.WebsocketClientPolicy` accepts `observation/image` (upright uint8 RGB), `observation/state` (proprioception), and `prompt`; it returns `actions` with shape `[10,7]`. No second view or camera calibration is required.

## Execution checks

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
ruff check src scripts tests
python scripts/smoke_test.py --output-dir results/smoke
# Optionally exercise the rendered-data loader in the same compiled check:
python scripts/smoke_test.py --output-dir results/smoke-data --dataset data/libero_pairs
```

The smoke test uses small language/action modules with the regular vision backbone, trains two updates, restores the optimizer checkpoint, trains one more update, and checks single-view sampling. See [validation](docs/validation.md) for what has actually been run. Full benchmark training and evaluation are separate from these execution checks. This repository contains source and normalization assets; datasets and trained submission checkpoints are not bundled.

## Code layout and licenses

- `src/openpi/models/cross_view.py`: paired objective and prefix reuse.
- `src/openpi/training/`: paper recipes, data loading, optimization, and checkpoints.
- `src/afcv/`: camera evaluation, restricted support, and timestamp alignment.
- `scripts/`: data preparation, training, evaluation, and serving entry points.
- `tests/`: mathematical and data-contract checks.

The policy backbone and client are based on [OpenPI](https://github.com/Physical-Intelligence/openpi) under Apache 2.0. Gemma-related material is subject to [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt). [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), and [LeRobot](https://github.com/huggingface/lerobot) retain their respective licenses. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution. Anonymous release authors are listed without identifying metadata.
