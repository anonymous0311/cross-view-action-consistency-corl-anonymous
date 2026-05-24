"""Evaluate a checkpoint on original LIBERO nominal-camera ID tasks.

This is the in-distribution companion to LIBERO-plus C1/C2/C3 camera eval.
It uses the original LIBERO benchmark from ``openpi/third_party/libero`` and
runs the 40 base tasks from spatial/object/goal/libero_10 by default.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any

import tqdm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for rel_path in ("openpi/third_party/libero", "openpi/src", "openpi/packages/openpi-client/src"):
    abs_path = REPO_ROOT / rel_path
    if abs_path.exists() and str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))

from canonical.eval import libero_policy_runner
from canonical.eval.libero_plus_eval import load_pi05_checkpoint


DEFAULT_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "pi05_libero_scene" / "pi05_libero_scene_01" / "10000")
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    config_name: str = "pi05_libero_scene"
    results_dir: str = "results/id_eval/libero_nominal"
    n_trials_per_task: int = 3
    suite_names: str = ",".join(DEFAULT_SUITES)
    task_specs: str = ""
    resolution: int = 256
    seed: int = 7
    progress_every: int = 5
    image_flip_mode: str = "both"
    resume: bool = False
    overwrite_results: bool = False
    num_shards: int = 1
    shard_index: int = 0


@dataclasses.dataclass(frozen=True)
class IDTrialResult:
    suite_name: str
    task_index: int
    task_spec: str
    task_name: str
    task_description: str
    trial_index: int
    rollout_seed: int
    success: bool
    done: bool
    steps_executed: int


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
        parser.add_argument("--suite-names", default=Args.suite_names)
        parser.add_argument("--task-specs", default=Args.task_specs)
        parser.add_argument("--resolution", type=int, default=Args.resolution)
        parser.add_argument("--seed", type=int, default=Args.seed)
        parser.add_argument("--progress-every", type=int, default=Args.progress_every)
        parser.add_argument("--image-flip-mode", default=Args.image_flip_mode, choices=("both", "flipud", "fliplr", "none"))
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--overwrite-results", action="store_true")
        parser.add_argument("--num-shards", type=int, default=Args.num_shards)
        parser.add_argument("--shard-index", type=int, default=Args.shard_index)
        ns = parser.parse_args()
        return Args(
            checkpoint_dir=ns.checkpoint_dir,
            config_name=ns.config_name,
            results_dir=ns.results_dir,
            n_trials_per_task=ns.n_trials_per_task,
            suite_names=ns.suite_names,
            task_specs=ns.task_specs,
            resolution=ns.resolution,
            seed=ns.seed,
            progress_every=ns.progress_every,
            image_flip_mode=ns.image_flip_mode,
            resume=ns.resume,
            overwrite_results=ns.overwrite_results,
            num_shards=ns.num_shards,
            shard_index=ns.shard_index,
        )


def _parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _wilson_ci(n_success: int, n_total: int) -> tuple[float, float]:
    if n_total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    phat = n_success / n_total
    denom = 1.0 + (z * z) / n_total
    center = (phat + (z * z) / (2.0 * n_total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n_total) + (z * z) / (4.0 * n_total * n_total))
    return max(0.0, center - margin), min(1.0, center + margin)


def _enumerate_task_specs(suite_names: Iterable[str]) -> list[str]:
    task_specs: list[str] = []
    for suite_name in suite_names:
        suite = libero_policy_runner.get_task_suite(suite_name)
        for task_index in range(int(suite.n_tasks)):
            task_specs.append(f"{suite_name}:{task_index}")
    return task_specs


def _load_trial_rows(path: pathlib.Path) -> list[IDTrialResult]:
    if not path.exists():
        return []
    rows: list[IDTrialResult] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(IDTrialResult(**json.loads(line)))
    return rows


def _write_trial_rows(path: pathlib.Path, rows: Iterable[IDTrialResult]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row), ensure_ascii=True) + "\n")


def _append_trial_rows(path: pathlib.Path, rows: Iterable[IDTrialResult]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row), ensure_ascii=True) + "\n")


def _sanitize_resumed_rows(
    rows: list[IDTrialResult],
    n_trials_per_task: int,
) -> tuple[list[IDTrialResult], set[str]]:
    grouped: dict[str, list[IDTrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.task_spec].append(row)

    kept: list[IDTrialResult] = []
    completed: set[str] = set()
    for task_spec, task_rows in grouped.items():
        if len(task_rows) == n_trials_per_task:
            kept.extend(sorted(task_rows, key=lambda row: row.trial_index))
            completed.add(task_spec)
    return kept, completed


def _validate_resume_metadata(metadata_path: pathlib.Path, args: Args, task_specs: list[str]) -> None:
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint_path": args.checkpoint_dir,
        "config_name": args.config_name,
        "n_trials_per_task": args.n_trials_per_task,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "image_flip_mode": args.image_flip_mode,
        "task_specs": task_specs,
    }
    mismatches: list[str] = []
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            mismatches.append(f"{key}: existing={metadata[key]!r}, requested={value!r}")
    if mismatches:
        raise ValueError(
            f"Refusing to resume incompatible ID eval directory {metadata_path.parent}. "
            + "; ".join(mismatches)
        )


def _aggregate_by_task(rows: Iterable[IDTrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[IDTrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.task_spec].append(row)

    out: list[dict[str, Any]] = []
    for task_spec, task_rows in sorted(grouped.items()):
        exemplar = task_rows[0]
        n_total = len(task_rows)
        n_success = sum(int(row.success) for row in task_rows)
        ci_low, ci_high = _wilson_ci(n_success, n_total)
        out.append(
            {
                "suite_name": exemplar.suite_name,
                "task_id": exemplar.task_index,
                "task_spec": task_spec,
                "task_name": exemplar.task_name,
                "task_description": exemplar.task_description,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def _aggregate_by_suite(rows: Iterable[IDTrialResult]) -> list[dict[str, Any]]:
    rows = list(rows)
    grouped: dict[str, list[IDTrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.suite_name].append(row)

    out: list[dict[str, Any]] = []
    for suite_name, suite_rows in sorted(grouped.items()):
        n_total = len(suite_rows)
        n_success = sum(int(row.success) for row in suite_rows)
        ci_low, ci_high = _wilson_ci(n_success, n_total)
        out.append(
            {
                "suite_name": suite_name,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    if rows:
        n_total = len(rows)
        n_success = sum(int(row.success) for row in rows)
        ci_low, ci_high = _wilson_ci(n_success, n_total)
        out.append(
            {
                "suite_name": "All",
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(aggregate_rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "# Original LIBERO Nominal ID Evaluation",
        "",
        "| Suite | Success Rate (95% Wilson CI) | n_success / n_total |",
        "|---|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['suite_name']} | "
            f"{100 * float(row['success_rate']):.1f}% "
            f"([{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]) | "
            f"{row['n_success']} / {row['n_total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(
    rows: list[IDTrialResult],
    results_dir: pathlib.Path,
    metadata_path: pathlib.Path,
    *,
    args: Args,
    task_specs: list[str],
    status: str,
    completed_tasks: int,
    started_at: float,
) -> None:
    per_task_rows = _aggregate_by_task(rows)
    aggregate_rows = _aggregate_by_suite(rows)
    _write_csv(per_task_rows, results_dir / "per_task_results.csv")
    _write_csv(aggregate_rows, results_dir / "aggregate_results.csv")
    _write_markdown(aggregate_rows, results_dir / "aggregate_summary.md")
    metadata = {
        "status": status,
        "checkpoint_path": args.checkpoint_dir,
        "config_name": args.config_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials_per_task": args.n_trials_per_task,
        "suite_names": _parse_csv_items(args.suite_names),
        "task_specs": task_specs,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "image_flip_mode": args.image_flip_mode,
        "completed_tasks": completed_tasks,
        "total_tasks": len(task_specs),
        "total_rollouts": len(rows),
        "wall_time_sec": time.time() - started_at,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _prefetch_libero_id_envs(task_spec: str, args: Args) -> list[Any]:
    """Create N independent LIBERO envs (one per trial) for the given task spec.

    Submitted to a ThreadPoolExecutor so env creation (MuJoCo/EGL init, asset loading)
    overlaps with policy rollout on the current task, hiding the cumulative init latency.
    Returns the list of env objects; caller must close each one.
    """
    suite_name, task_id_raw = task_spec.split(":", maxsplit=1)
    suite = libero_policy_runner.get_task_suite(suite_name)
    task = suite.get_task(int(task_id_raw))
    envs: list[Any] = []
    for _ in range(args.n_trials_per_task):
        env_i, _ = libero_policy_runner.create_libero_env(task, resolution=args.resolution, seed=args.seed)
        envs.append(env_i)
    return envs


def _run_single_task(policy: Any, task_spec: str, args: Args, *, prebuilt_envs: list[Any] | None = None) -> list[IDTrialResult]:
    """Run all trials for one task using batched lockstep rollout.

    Image orientation (args.image_flip_mode) is applied per-env inside the batched
    rollout's observation builder, exactly as in the sequential path; the critical
    image-flip bug fix is preserved. Env stepping remains serialized because
    LIBERO/robosuite MuJoCo+EGL stepping is not thread-safe across env instances.
    """
    suite_name, task_id_raw = task_spec.split(":", maxsplit=1)
    task_id = int(task_id_raw)
    suite = libero_policy_runner.get_task_suite(suite_name)
    task = suite.get_task(task_id)
    task_description = str(task.language)
    initial_states = suite.get_task_init_states(task_id)
    if args.n_trials_per_task > len(initial_states):
        raise ValueError(
            f"Requested {args.n_trials_per_task} trials for {task_spec}, "
            f"but only {len(initial_states)} init states exist."
        )

    if prebuilt_envs is not None:
        if len(prebuilt_envs) != args.n_trials_per_task:
            raise ValueError(
                f"prebuilt_envs has {len(prebuilt_envs)} entries but n_trials_per_task={args.n_trials_per_task}"
            )
        envs = list(prebuilt_envs)
    else:
        envs = []
        for _ in range(args.n_trials_per_task):
            env_i, _ = libero_policy_runner.create_libero_env(task, resolution=args.resolution, seed=args.seed)
            envs.append(env_i)

    rollout_cfg = libero_policy_runner.RolloutConfig(
        max_steps=libero_policy_runner.default_max_steps_for_suite(suite_name),
        num_steps_wait=10,
        replan_steps=5,
        resize_size=224,
    )

    rollout_seeds: list[int] = []
    for trial_index in range(args.n_trials_per_task):
        rs = _stable_seed(args.seed, task_spec, trial_index, "libero_id")
        envs[trial_index].seed(rs)
        rollout_seeds.append(rs)

    initial_states_used = [initial_states[t] for t in range(args.n_trials_per_task)]

    task_rows: list[IDTrialResult] = []
    try:
        results = libero_policy_runner.rollout_episode_batched(
            policy,
            envs,
            task_description,
            rollout_cfg,
            initial_states=initial_states_used,
            image_flip_mode=args.image_flip_mode,
            env_executor=None,
        )
        for trial_index, result in enumerate(results):
            task_rows.append(
                IDTrialResult(
                    suite_name=suite_name,
                    task_index=task_id,
                    task_spec=task_spec,
                    task_name=str(task.name),
                    task_description=task_description,
                    trial_index=trial_index,
                    rollout_seed=rollout_seeds[trial_index],
                    success=bool(result["success"]),
                    done=bool(result["done"]),
                    steps_executed=int(result["steps_executed"]),
                )
            )
    finally:
        for env_i in envs:
            try:
                env_i.close()
            except Exception:
                pass
    return task_rows


def main(args: Args) -> None:
    if args.n_trials_per_task <= 0:
        raise ValueError("--n-trials-per-task must be > 0.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")

    selected_task_specs = _parse_csv_items(args.task_specs)
    if not selected_task_specs:
        selected_task_specs = _enumerate_task_specs(_parse_csv_items(args.suite_names))
    selected_task_specs = [spec for idx, spec in enumerate(selected_task_specs) if idx % args.num_shards == args.shard_index]

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_rows_path = results_dir / "trial_results.jsonl"
    metadata_path = results_dir / "metadata.json"
    started_at = time.time()

    print(
        "[info] Original LIBERO ID eval config:",
        {
            "checkpoint_dir": args.checkpoint_dir,
            "config_name": args.config_name,
            "results_dir": str(results_dir),
            "n_trials_per_task": args.n_trials_per_task,
            "task_count": len(selected_task_specs),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "seed": args.seed,
            "resolution": args.resolution,
            "image_flip_mode": args.image_flip_mode,
            "resume": args.resume,
        },
        flush=True,
    )

    if raw_rows_path.exists() and raw_rows_path.stat().st_size > 0 and not args.resume:
        if not args.overwrite_results:
            raise RuntimeError(
                f"Refusing to append to existing results without --resume or --overwrite-results: {raw_rows_path}"
            )
        _write_trial_rows(raw_rows_path, [])

    rows: list[IDTrialResult] = []
    completed_task_specs: set[str] = set()
    if args.resume:
        _validate_resume_metadata(metadata_path, args, selected_task_specs)
        loaded_rows = _load_trial_rows(raw_rows_path)
        rows, completed_task_specs = _sanitize_resumed_rows(loaded_rows, args.n_trials_per_task)
        if len(rows) != len(loaded_rows):
            _write_trial_rows(raw_rows_path, rows)
        print(f"[info] Resume mode: found {len(completed_task_specs)} completed tasks.", flush=True)

    _write_outputs(
        rows,
        results_dir,
        metadata_path,
        args=args,
        task_specs=selected_task_specs,
        status="running",
        completed_tasks=len(completed_task_specs),
        started_at=started_at,
    )

    policy = load_pi05_checkpoint(args.checkpoint_dir, config_name=args.config_name)

    # Trigger JAX JIT compilation once before the task loop so the first real task
    # does not pay the 30-60 s compilation cost (which shows as a misleading GPU spike
    # followed by idle time in nvidia-smi).  Warmup with batch_size = n_trials_per_task
    # because the main loop uses batched inference.
    libero_policy_runner.warmup_policy(policy, batch_size=args.n_trials_per_task, resolution=args.resolution)

    task_bar = tqdm.tqdm(
        total=len(selected_task_specs),
        initial=len(completed_task_specs),
        dynamic_ncols=True,
        leave=True,
        unit="task",
        desc="Original LIBERO ID",
    )

    # Env prefetching: create the next task's N MuJoCo envs (one per trial) in a
    # background thread while the policy runs rollouts on the current task. Hides
    # the N*(2-5 s) of MuJoCo/EGL init behind rollout time.
    # Fall back to synchronous creation silently if the background thread fails.
    pending_specs = [s for s in selected_task_specs if s not in completed_task_specs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        prefetch_future: concurrent.futures.Future[Any] | None = None
        if pending_specs:
            prefetch_future = pool.submit(_prefetch_libero_id_envs, pending_specs[0], args)

        for i, task_spec in enumerate(selected_task_specs):
            if task_spec in completed_task_specs:
                continue

            # Retrieve pre-built envs (or fall back to synchronous creation).
            prebuilt_envs: list[Any] | None = None
            if prefetch_future is not None:
                try:
                    prebuilt_envs = prefetch_future.result()
                except Exception as exc:
                    print(f"[warn] env prefetch failed ({exc}); creating envs synchronously.", flush=True)
                prefetch_future = None

            # Start prefetching the next pending task's envs while we run this one.
            remaining = [s for s in selected_task_specs[i + 1:] if s not in completed_task_specs]
            if remaining:
                prefetch_future = pool.submit(_prefetch_libero_id_envs, remaining[0], args)

            task_rows = _run_single_task(policy, task_spec, args, prebuilt_envs=prebuilt_envs)
            rows.extend(task_rows)
            completed_task_specs.add(task_spec)
            _append_trial_rows(raw_rows_path, task_rows)
            _write_outputs(
                rows,
                results_dir,
                metadata_path,
                args=args,
                task_specs=selected_task_specs,
                status="running",
                completed_tasks=len(completed_task_specs),
                started_at=started_at,
            )
            n_success = sum(int(row.success) for row in task_rows)
            task_bar.set_postfix(task=task_spec, sr=f"{n_success}/{len(task_rows)}")
            task_bar.update(1)
            if len(completed_task_specs) % max(1, args.progress_every) == 0:
                print(
                    f"[progress] completed {len(completed_task_specs)}/{len(selected_task_specs)} tasks",
                    flush=True,
                )

    task_bar.close()

    _write_outputs(
        rows,
        results_dir,
        metadata_path,
        args=args,
        task_specs=selected_task_specs,
        status="completed",
        completed_tasks=len(completed_task_specs),
        started_at=started_at,
    )
    print(f"[info] Original LIBERO ID evaluation finished. Outputs written to: {results_dir}", flush=True)


if __name__ == "__main__":
    main(_parse_args())
