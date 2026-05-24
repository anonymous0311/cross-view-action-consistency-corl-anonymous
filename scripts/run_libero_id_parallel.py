"""Launch parallel shard workers for original LIBERO nominal ID evaluation."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "pi05_libero_scene" / "pi05_libero_scene_01" / "10000")


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    config_name: str = "pi05_libero_scene"
    results_dir: str = "results/id_eval/libero_nominal_parallel"
    n_trials_per_task: int = 3
    gpu_ids: str = "0,1"
    suite_names: str = "libero_spatial,libero_object,libero_goal,libero_10"
    task_specs: str = ""
    resolution: int = 256
    seed: int = 7
    progress_every: int = 5
    image_flip_mode: str = "both"
    resume: bool = False
    allow_gpu_oversubscription: bool = False


def _parse_args() -> Args:
    try:
        import tyro  # noqa: PLC0415

        return tyro.cli(Args)
    except ModuleNotFoundError:
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint-dir", default=Args.checkpoint_dir)
        parser.add_argument("--config-name", default=Args.config_name)
        parser.add_argument("--results-dir", default=Args.results_dir)
        parser.add_argument("--n-trials-per-task", type=int, default=Args.n_trials_per_task)
        parser.add_argument("--gpu-ids", default=Args.gpu_ids)
        parser.add_argument("--suite-names", default=Args.suite_names)
        parser.add_argument("--task-specs", default=Args.task_specs)
        parser.add_argument("--resolution", type=int, default=Args.resolution)
        parser.add_argument("--seed", type=int, default=Args.seed)
        parser.add_argument("--progress-every", type=int, default=Args.progress_every)
        parser.add_argument("--image-flip-mode", default=Args.image_flip_mode, choices=("both", "flipud", "fliplr", "none"))
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--allow-gpu-oversubscription", action="store_true")
        ns = parser.parse_args()
        return Args(
            checkpoint_dir=ns.checkpoint_dir,
            config_name=ns.config_name,
            results_dir=ns.results_dir,
            n_trials_per_task=ns.n_trials_per_task,
            gpu_ids=ns.gpu_ids,
            suite_names=ns.suite_names,
            task_specs=ns.task_specs,
            resolution=ns.resolution,
            seed=ns.seed,
            progress_every=ns.progress_every,
            image_flip_mode=ns.image_flip_mode,
            resume=ns.resume,
            allow_gpu_oversubscription=ns.allow_gpu_oversubscription,
        )


def _python() -> str:
    candidates = [
        REPO_ROOT / ".venv-libero-plus" / "bin" / "python",
        REPO_ROOT / "venv-libero-plus" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find LIBERO python interpreter. "
        "Expected one of: .venv-libero-plus/bin/python or venv-libero-plus/bin/python"
    )


def _read_metadata(path: pathlib.Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not seconds < float("inf"):
        return "unknown"
    seconds_int = int(seconds)
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main(args: Args) -> None:
    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    if len(set(gpu_ids)) != len(gpu_ids) and not args.allow_gpu_oversubscription:
        raise ValueError(
            "--gpu-ids contains duplicates. This launcher starts one worker per listed id immediately. "
            "Use unique GPU ids, or pass --allow-gpu-oversubscription if intentional."
        )

    results_dir = pathlib.Path(args.results_dir)
    shard_root = results_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = (
        f"{REPO_ROOT}:"
        f"{REPO_ROOT / 'openpi/third_party/libero'}:"
        f"{REPO_ROOT / 'openpi/src'}:"
        f"{REPO_ROOT / 'openpi/packages/openpi-client/src'}"
    )
    env_base["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env_base["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    workers: list[tuple[int, subprocess.Popen[str], pathlib.Path, pathlib.Path]] = []
    started_at = time.time()
    for shard_index, gpu_id in enumerate(gpu_ids):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "worker.log"
        metadata_path = shard_dir / "metadata.json"
        cmd = [
            _python(),
            str(REPO_ROOT / "scripts" / "evaluate_libero_id.py"),
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--config-name",
            args.config_name,
            "--results-dir",
            str(shard_dir),
            "--n-trials-per-task",
            str(args.n_trials_per_task),
            "--suite-names",
            args.suite_names,
            "--resolution",
            str(args.resolution),
            "--seed",
            str(args.seed),
            "--progress-every",
            str(args.progress_every),
            "--image-flip-mode",
            args.image_flip_mode,
            "--num-shards",
            str(len(gpu_ids)),
            "--shard-index",
            str(shard_index),
        ]
        if args.task_specs:
            cmd += ["--task-specs", args.task_specs]
        if args.resume:
            cmd.append("--resume")

        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["LIBERO_CONFIG_PATH"] = str(shard_dir / ".libero")
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((shard_index, proc, log_path, metadata_path))
        print(f"[info] launched ID shard {shard_index}/{len(gpu_ids)} on GPU {gpu_id} -> pid={proc.pid}, log={log_path}")

    completed_workers: set[int] = set()
    failed = False
    while len(completed_workers) < len(workers):
        total_completed_tasks = 0
        total_tasks = 0
        shard_summaries: list[str] = []

        for shard_index, proc, log_path, metadata_path in workers:
            metadata = _read_metadata(metadata_path)
            completed = int(metadata.get("completed_tasks", 0) or 0)
            total = int(metadata.get("total_tasks", 0) or 0)
            status = str(metadata.get("status", "starting"))
            total_completed_tasks += completed
            total_tasks += total
            shard_summaries.append(f"s{shard_index}:{completed}/{total or '?'}:{status}")

            return_code = proc.poll()
            if return_code is not None and shard_index not in completed_workers:
                completed_workers.add(shard_index)
                if return_code != 0:
                    failed = True
                    print(f"[error] ID shard {shard_index} failed with code {return_code}. See {log_path}")
                else:
                    print(f"[info] ID shard {shard_index} completed successfully.")

        elapsed = time.time() - started_at
        eta_seconds = None
        if total_completed_tasks > 0 and total_tasks > total_completed_tasks:
            rate = total_completed_tasks / elapsed
            if rate > 0:
                eta_seconds = (total_tasks - total_completed_tasks) / rate
        print(
            "[progress]",
            f"{total_completed_tasks}/{total_tasks} tasks",
            f"elapsed={_format_eta(elapsed)}",
            f"eta={_format_eta(eta_seconds)}",
            "|",
            " ".join(shard_summaries),
        )

        if len(completed_workers) < len(workers):
            time.sleep(10)

    if failed:
        raise SystemExit(1)

    subprocess.run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "merge_libero_id_shards.py"),
            "--results-dir",
            str(results_dir),
            "--num-shards",
            str(len(gpu_ids)),
        ],
        cwd=REPO_ROOT,
        env=env_base,
        check=True,
    )
    print(f"[info] parallel original LIBERO ID eval finished in {time.time() - started_at:.1f}s")


if __name__ == "__main__":
    main(_parse_args())
