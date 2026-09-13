# Release validation

Execution checks were run in an independently installed environment on Linux with Python 3.11, the committed `uv.lock`, and NVIDIA RTX PRO 6000 GPUs. Each GPU check used one visible GPU. These checks validate software execution and data contracts; they do not measure the paper's success rates.

| Check | Result |
| --- | --- |
| Locked core, CUDA, and simulator dependency installation | Passed |
| Objective and data-contract tests | 9 passed |
| Original demonstration inventory | 40 tasks, 2,000 episodes, 338,575 states |
| Original LIBERO HDF5 → paired MuJoCo render → LeRobot export | Passed on one demonstration, four states |
| Exported data → normalized paired training batch | Passed; actions `[B,2,10,32]`, identical labels within each pair |
| Compiled paired training on rendered data | Two updates, optimizer restore, third update, checkpoint reload, and single-view sampling passed |
| Nominal-only and mixed-camera compiled training | Training, optimizer restore, and single-view sampling passed |
| Full π0.5 base-checkpoint inference in nominal and perturbed-camera environments | Passed; each executed five commanded actions after settling |
| Evaluation CLI and resume | Nominal/camera output passed; camera resume retained one completed trial without duplication |
| Camera task enumeration | 1,599 tasks: 313 distance, 992 position, 294 orientation |
| Coverage episode selection | Exactly 100 / 500 of 2,000 episodes; nested membership |
| Robot interchange export and training loader | Passed on synthetic timestamped camera streams; action-chunk boundary checked |
| Ruff and source/wheel build | Passed |

The compiled training check uses small language/action modules with a regular, frozen vision backbone. Full-checkpoint rollout checks use the pretrained π0.5 base weights, not a submitted fine-tuned policy. Simulator assets for local checks were read from an existing copy of the official LIBERO-Plus assets; the setup script provides the pinned public download route.

Full 10,000-update, three-seed training; the complete 2,000/4,797-trial benchmarks; view-generator inference; and physical robot execution were not rerun for this release. Datasets, pretrained model downloads, and simulator assets are intentionally external to Git. The robot demonstrations and trained submission checkpoints are not bundled.
