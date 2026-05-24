"""CLI entrypoint for LIBERO-plus camera evaluation with resume support."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pathlib
import sys
import time
from typing import Any

import tqdm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for rel_path in ("openpi/src", "openpi/packages/openpi-client/src", "LIBERO-plus"):
    abs_path = REPO_ROOT / rel_path
    if abs_path.exists() and str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))

from canonical.eval.libero_plus_eval import CanonicalInferenceExtractor
from canonical.eval.libero_plus_eval import LIBEROPlusCameraEvaluator
from canonical.eval.libero_plus_eval import NOMINAL_BASELINE_TASK_SPECS
from canonical.eval.libero_plus_eval import TrialResult
from canonical.eval.libero_plus_eval import create_libero_plus_env
from canonical.eval.libero_plus_eval import create_task_suite
from canonical.eval.libero_plus_eval import evaluate_with_scene_camera_masked
from canonical.eval.libero_plus_eval import load_pi05_checkpoint
from canonical.eval.libero_plus_eval import parse_task_spec
from canonical.eval.libero_plus_eval import prefetch_libero_plus_envs
from canonical.eval.libero_plus_eval import rollout_episode
from canonical.eval.libero_plus_eval import warmup_policy
from canonical.eval.libero_plus_eval import _stable_seed


DEFAULT_CHECKPOINT = str(REPO_ROOT / "checkpoints" / "pi05_libero_scene" / "pi05_libero_scene_01" / "10000")
TASK_KEY_FIELDS = ("condition", "suite_name", "task_index")


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    config_name: str = "pi05_libero_scene"
    n_trials_per_task: int = 3
    masking_trials: int = 3
    results_dir: str = "results/main_table/m0"
    resolution: int = 256
    seed: int = 7
    progress_every: int = 25
    use_wrist_image: bool = False
    image_flip_mode: str = "both"
    resume: bool = False
    num_shards: int = 1
    shard_index: int = 0
    skip_masking: bool = False
    only_masking: bool = False
    max_tasks: int | None = None
    # M5 canonical: if set, load Stage 1 encoder and inject canonical tokens each step
    stage1_checkpoint: str = ""
    stage1_config: str = "configs/stage1_v3_slotnce.yaml"
    canonical_token_mode: str = "matched"
    canonical_cache_h5: str = "data/libero_canonical_cache_v3/canonical_tokens.h5"
    canonical_shuffle_seed: int = 17
    canonical_pool_size: int = 200
    canonical_constant_pool_size: int = 100
    libero_parquet_dir: str = "data/libero/data/chunk-000"
    conditions: str = ""
    levels: str = ""
    task_specs: str = ""
    include_nominal: bool = False
    nominal_trials_per_task: int = 3
    max_tasks_per_condition: int | None = None
    m6_alpha_override: float | None = None
    overwrite_results: bool = False


def _parse_args() -> Args:
    try:
        import tyro  # noqa: PLC0415

        return tyro.cli(Args)
    except ModuleNotFoundError:
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint-dir", default=Args.checkpoint_dir)
        parser.add_argument("--config-name", default=Args.config_name)
        parser.add_argument("--n-trials-per-task", type=int, default=Args.n_trials_per_task)
        parser.add_argument("--masking-trials", type=int, default=Args.masking_trials)
        parser.add_argument("--results-dir", default=Args.results_dir)
        parser.add_argument("--resolution", type=int, default=Args.resolution)
        parser.add_argument("--seed", type=int, default=Args.seed)
        parser.add_argument("--progress-every", type=int, default=Args.progress_every)
        parser.add_argument("--use-wrist-image", action="store_true")
        parser.add_argument("--image-flip-mode", default=Args.image_flip_mode, choices=("both", "flipud", "fliplr", "none"))
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--num-shards", type=int, default=Args.num_shards)
        parser.add_argument("--shard-index", type=int, default=Args.shard_index)
        parser.add_argument("--skip-masking", action="store_true")
        parser.add_argument("--only-masking", action="store_true")
        parser.add_argument("--max-tasks", type=int, default=None)
        parser.add_argument("--stage1-checkpoint", default="")
        parser.add_argument("--stage1-config", default=Args.stage1_config)
        parser.add_argument("--canonical-token-mode", default=Args.canonical_token_mode)
        parser.add_argument("--canonical-cache-h5", default=Args.canonical_cache_h5)
        parser.add_argument("--canonical-shuffle-seed", type=int, default=Args.canonical_shuffle_seed)
        parser.add_argument("--canonical-pool-size", type=int, default=Args.canonical_pool_size)
        parser.add_argument("--canonical-constant-pool-size", type=int, default=Args.canonical_constant_pool_size)
        parser.add_argument("--libero-parquet-dir", default=Args.libero_parquet_dir)
        parser.add_argument("--conditions", default=Args.conditions)
        parser.add_argument("--levels", default=Args.levels)
        parser.add_argument("--task-specs", default=Args.task_specs)
        parser.add_argument("--include-nominal", action="store_true")
        parser.add_argument("--nominal-trials-per-task", type=int, default=Args.nominal_trials_per_task)
        parser.add_argument("--max-tasks-per-condition", type=int, default=None)
        parser.add_argument("--m6-alpha-override", type=float, default=None)
        parser.add_argument("--overwrite-results", action="store_true")
        ns = parser.parse_args()
        return Args(
            checkpoint_dir=ns.checkpoint_dir,
            config_name=ns.config_name,
            n_trials_per_task=ns.n_trials_per_task,
            masking_trials=ns.masking_trials,
            results_dir=ns.results_dir,
            resolution=ns.resolution,
            seed=ns.seed,
            progress_every=ns.progress_every,
            use_wrist_image=ns.use_wrist_image,
            image_flip_mode=ns.image_flip_mode,
            resume=ns.resume,
            num_shards=ns.num_shards,
            shard_index=ns.shard_index,
            skip_masking=ns.skip_masking,
            only_masking=ns.only_masking,
            max_tasks=ns.max_tasks,
            stage1_checkpoint=ns.stage1_checkpoint,
            stage1_config=ns.stage1_config,
            canonical_token_mode=ns.canonical_token_mode,
            canonical_cache_h5=ns.canonical_cache_h5,
            canonical_shuffle_seed=ns.canonical_shuffle_seed,
            canonical_pool_size=ns.canonical_pool_size,
            canonical_constant_pool_size=ns.canonical_constant_pool_size,
            libero_parquet_dir=ns.libero_parquet_dir,
            conditions=ns.conditions,
            levels=ns.levels,
            task_specs=ns.task_specs,
            include_nominal=ns.include_nominal,
            nominal_trials_per_task=ns.nominal_trials_per_task,
            max_tasks_per_condition=ns.max_tasks_per_condition,
            m6_alpha_override=ns.m6_alpha_override,
            overwrite_results=ns.overwrite_results,
        )


def _parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _filter_tasks(tasks: list[Any], args: Args) -> list[Any]:
    cond_items = {item.upper() for item in _parse_csv_items(args.conditions)}
    if cond_items:
        unknown = sorted(cond_items - {"C1", "C2", "C3"})
        if unknown:
            raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
        tasks = [task for task in tasks if task.condition in cond_items]

    level_items = _parse_csv_items(args.levels)
    if level_items:
        levels = {int(item) for item in level_items}
        tasks = [task for task in tasks if int(task.level) in levels]

    task_spec_items = _parse_csv_items(args.task_specs)
    if task_spec_items:
        wanted = {parse_task_spec(spec) for spec in task_spec_items}
        tasks = [task for task in tasks if (task.suite_name, int(task.task_index)) in wanted]

    return tasks


def _limit_tasks_per_condition(tasks: list[Any], max_tasks_per_condition: int | None) -> list[Any]:
    if max_tasks_per_condition is None:
        return tasks
    if max_tasks_per_condition < 0:
        raise ValueError("--max-tasks-per-condition must be >= 0.")
    counts: dict[str, int] = {}
    kept = []
    for task in tasks:
        count = counts.get(task.condition, 0)
        if count >= max_tasks_per_condition:
            continue
        kept.append(task)
        counts[task.condition] = count + 1
    return kept


def _task_key(row: TrialResult) -> tuple[str, str, int]:
    return row.condition, row.suite_name, row.task_index


def _load_trial_rows(path: pathlib.Path) -> list[TrialResult]:
    if not path.exists():
        return []
    rows: list[TrialResult] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(TrialResult(**json.loads(line)))
    return rows


def _write_trial_rows(path: pathlib.Path, rows: list[TrialResult]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row)) + "\n")


def _append_trial_rows(path: pathlib.Path, rows: list[TrialResult]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row)) + "\n")


def _validate_resume_metadata(metadata_path: pathlib.Path, args: Args) -> None:
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot resume with invalid metadata file: {metadata_path}") from exc

    expected = {
        "checkpoint_path": args.checkpoint_dir,
        "config_name": args.config_name,
        "n_trials_per_task": args.n_trials_per_task,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "image_flip_mode": args.image_flip_mode,
        "conditions": args.conditions,
        "levels": args.levels,
        "task_specs": args.task_specs,
        "max_tasks_per_condition": args.max_tasks_per_condition,
    }
    mismatches: list[str] = []
    for key, value in expected.items():
        if key not in metadata:
            if value not in ("", None):
                mismatches.append(f"{key}: existing=<missing>, requested={value!r}")
            continue
        if metadata[key] != value:
            mismatches.append(f"{key}: existing={metadata[key]!r}, requested={value!r}")
    if mismatches:
        raise ValueError(
            f"Refusing to resume incompatible eval directory {metadata_path.parent}. "
            + "; ".join(mismatches)
        )


def _sanitize_resumed_rows(rows: list[TrialResult], n_trials_per_task: int) -> tuple[list[TrialResult], set[tuple[str, str, int]]]:
    grouped: dict[tuple[str, str, int], list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault(_task_key(row), []).append(row)

    kept: list[TrialResult] = []
    completed: set[tuple[str, str, int]] = set()
    for key, group_rows in grouped.items():
        if len(group_rows) == n_trials_per_task:
            kept.extend(sorted(group_rows, key=lambda row: row.trial_index))
            completed.add(key)
    return kept, completed


def _write_masking_summary(masking: dict[str, Any], output_path: pathlib.Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                "# Scene Camera Masking Check",
                "",
                f"- Successes: {masking['n_success']} / {masking['n_total']}",
                f"- Success rate: {100 * float(masking['success_rate']):.1f}%",
                f"- 95% Wilson CI: [{100 * float(masking['ci_low']):.1f}, {100 * float(masking['ci_high']):.1f}]",
                "",
                "Interpretation:",
                "",
                "- This run is valid only if masking causes a large performance drop.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_nominal_tasks(evaluator: LIBEROPlusCameraEvaluator, n_trials_per_task: int) -> list[TrialResult]:
    raw_rows: list[TrialResult] = []
    for task_spec in NOMINAL_BASELINE_TASK_SPECS:
        suite_name, task_id = parse_task_spec(task_spec)
        suite = create_task_suite(suite_name)
        suite_task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = create_libero_plus_env(
            suite_task,
            resolution=evaluator.resolution,
            seed=evaluator.base_seed,
        )
        rollout_cfg = evaluator._rollout_config_for_suite(suite_name)
        try:
            for trial_index in range(min(n_trials_per_task, len(initial_states))):
                rollout_seed = _stable_seed(evaluator.base_seed, task_spec, trial_index, "nominal")
                env.seed(rollout_seed)
                result = rollout_episode(
                    evaluator.policy,
                    env,
                    task_description,
                    rollout_cfg,
                    initial_state=initial_states[trial_index],
                    use_wrist_image=evaluator.use_wrist_image,
                    mask_scene=False,
                    canonical_extractor=evaluator.canonical_extractor,
                    image_flip_mode=evaluator.image_flip_mode,
                )
                raw_rows.append(
                    TrialResult(
                        condition="nominal",
                        suite_name=suite_name,
                        task_index=task_id,
                        json_id=-1,
                        task_name=str(suite_task.name),
                        level=0,
                        trial_index=trial_index,
                        rollout_seed=rollout_seed,
                        success=bool(result["success"]),
                        done=bool(result["done"]),
                        steps_executed=int(result["steps_executed"]),
                        masked_scene=False,
                    )
                )
        finally:
            env.close()
    return raw_rows


def _write_outputs(
    evaluator: LIBEROPlusCameraEvaluator,
    raw_rows: list[TrialResult],
    *,
    metadata_path: pathlib.Path,
    checkpoint_dir: str,
    config_name: str,
    n_trials_per_task: int,
    masking_trials: int,
    image_flip_mode: str,
    conditions: str,
    levels: str,
    task_specs: str,
    max_tasks_per_condition: int | None,
    num_shards: int,
    shard_index: int,
    started_at: float,
    status: str,
    completed_tasks: int,
    total_tasks: int,
) -> None:
    per_task_rows = evaluator.compute_aggregate_with_ci(raw_rows)
    aggregate_rows = evaluator.aggregate_by_condition_and_level(raw_rows)
    evaluator.write_results_csv(per_task_rows, evaluator.results_dir / "per_task_results.csv")
    evaluator.write_markdown_summary(aggregate_rows, evaluator.results_dir / "aggregate_summary.md")

    metadata = {
        "status": status,
        "checkpoint_path": checkpoint_dir,
        "config_name": config_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials_per_task": n_trials_per_task,
        "masking_trials": masking_trials,
        "image_flip_mode": image_flip_mode,
        "conditions": conditions,
        "levels": levels,
        "task_specs": task_specs,
        "max_tasks_per_condition": max_tasks_per_condition,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "total_rollouts": len(raw_rows),
        "wall_time_sec": time.time() - started_at,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(args: Args) -> None:
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_rows_path = results_dir / "trial_results.jsonl"
    metadata_path = results_dir / "metadata.json"
    started_at = time.time()
    print(
        "[info] LIBERO-plus eval config:",
        {
            "checkpoint_dir": args.checkpoint_dir,
            "config_name": args.config_name,
            "n_trials_per_task": args.n_trials_per_task,
            "masking_trials": args.masking_trials,
            "results_dir": str(results_dir),
            "resolution": args.resolution,
            "seed": args.seed,
            "use_wrist_image": args.use_wrist_image,
            "image_flip_mode": args.image_flip_mode,
            "resume": args.resume,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "skip_masking": args.skip_masking,
            "only_masking": args.only_masking,
            "max_tasks": args.max_tasks,
            "canonical_token_mode": args.canonical_token_mode,
            "canonical_cache_h5": args.canonical_cache_h5,
            "canonical_shuffle_seed": args.canonical_shuffle_seed,
            "canonical_pool_size": args.canonical_pool_size,
            "canonical_constant_pool_size": args.canonical_constant_pool_size,
            "libero_parquet_dir": args.libero_parquet_dir,
            "conditions": args.conditions,
            "levels": args.levels,
            "task_specs": args.task_specs,
            "include_nominal": args.include_nominal,
            "nominal_trials_per_task": args.nominal_trials_per_task,
            "max_tasks_per_condition": args.max_tasks_per_condition,
            "m6_alpha_override": args.m6_alpha_override,
            "overwrite_results": args.overwrite_results,
        },
    )

    if raw_rows_path.exists() and raw_rows_path.stat().st_size > 0 and not args.resume and not args.only_masking:
        if not args.overwrite_results:
            raise RuntimeError(
                f"Refusing to append to existing raw results without --resume or --overwrite-results: {raw_rows_path}. "
                "Use a fresh --results-dir for a new run, --resume to continue a compatible partial run, "
                "or --overwrite-results to truncate the existing trial_results.jsonl first."
            )
        _write_trial_rows(raw_rows_path, [])

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if args.m6_alpha_override is not None:
        if not 0.0 <= args.m6_alpha_override <= 1.0:
            raise ValueError("--m6-alpha-override must be in [0, 1].")
        os.environ["OPENPI_M6_ALPHA_OVERRIDE"] = str(args.m6_alpha_override)

    policy = load_pi05_checkpoint(args.checkpoint_dir, config_name=args.config_name)

    # Trigger JAX JIT compilation before the task loop so the first real task
    # does not pay the 30-60 s compilation cost (which shows as a misleading GPU spike
    # followed by idle time in nvidia-smi).  Warmup with batch_size = n_trials_per_task
    # because the main loop uses batched inference; a batch=1 warmup would only prime
    # the XLA cache for the masking sanity-check (which is small and one-shot).
    warmup_policy(policy, batch_size=args.n_trials_per_task, resolution=args.resolution)

    canonical_extractor = None
    if args.stage1_checkpoint:
        print(f"[info] Loading Stage 1 canonical extractor from: {args.stage1_checkpoint}")
        canonical_extractor = CanonicalInferenceExtractor(
            stage1_checkpoint=args.stage1_checkpoint,
            stage1_config=args.stage1_config,
            token_mode=args.canonical_token_mode,
            canonical_cache_h5=args.canonical_cache_h5,
            shuffle_seed=args.canonical_shuffle_seed,
            libero_parquet_dir=args.libero_parquet_dir,
            token_pool_size=args.canonical_pool_size,
            constant_pool_size=args.canonical_constant_pool_size,
        )

    evaluator = LIBEROPlusCameraEvaluator(
        policy=policy,
        n_trials_per_task=args.n_trials_per_task,
        results_dir=results_dir,
        resolution=args.resolution,
        base_seed=args.seed,
        use_wrist_image=args.use_wrist_image,
        progress_every=args.progress_every,
        canonical_extractor=canonical_extractor,
        image_flip_mode=args.image_flip_mode,
    )

    if args.only_masking:
        masking = evaluate_with_scene_camera_masked(
            policy,
            evaluator,
            n_trials=args.masking_trials,
            condition="nominal",
        )
        _write_masking_summary(masking, results_dir / "masking_check.md")
        metadata = {
            "status": "masking_only_completed",
            "checkpoint_path": args.checkpoint_dir,
            "config_name": args.config_name,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "masking_trials": args.masking_trials,
            "image_flip_mode": args.image_flip_mode,
            "conditions": args.conditions,
            "levels": args.levels,
            "task_specs": args.task_specs,
            "max_tasks_per_condition": args.max_tasks_per_condition,
            "masking_check": {
                "n_success": masking["n_success"],
                "n_total": masking["n_total"],
                "success_rate": masking["success_rate"],
                "ci_low": masking["ci_low"],
                "ci_high": masking["ci_high"],
            },
        }
        existing = {}
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing.update(metadata)
        metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"[info] masking-only evaluation finished. Outputs written to: {results_dir}")
        return

    condition_order = {"C1": 0, "C2": 1, "C3": 2}
    all_tasks = sorted(evaluator.camera_tasks, key=lambda task: (condition_order[task.condition], task.suite_name, task.task_index))
    all_tasks = _filter_tasks(all_tasks, args)
    all_tasks = _limit_tasks_per_condition(all_tasks, args.max_tasks_per_condition)
    all_tasks = [task for idx, task in enumerate(all_tasks) if idx % args.num_shards == args.shard_index]
    if args.max_tasks is not None:
        all_tasks = all_tasks[: args.max_tasks]

    raw_rows: list[TrialResult] = []
    completed_task_keys: set[tuple[str, str, int]] = set()
    if args.resume:
        _validate_resume_metadata(metadata_path, args)
        loaded_rows = _load_trial_rows(raw_rows_path)
        raw_rows, completed_task_keys = _sanitize_resumed_rows(loaded_rows, args.n_trials_per_task)
        if len(raw_rows) != len(loaded_rows):
            _write_trial_rows(raw_rows_path, raw_rows)
        print(f"[info] Resume mode: found {len(completed_task_keys)} completed tasks from {raw_rows_path}")

    _write_outputs(
        evaluator,
        raw_rows,
        metadata_path=metadata_path,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        n_trials_per_task=args.n_trials_per_task,
        masking_trials=args.masking_trials,
        image_flip_mode=args.image_flip_mode,
        conditions=args.conditions,
        levels=args.levels,
        task_specs=args.task_specs,
        max_tasks_per_condition=args.max_tasks_per_condition,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        started_at=started_at,
        status="running",
        completed_tasks=len(completed_task_keys),
        total_tasks=len(all_tasks),
    )

    if args.include_nominal and args.shard_index == 0 and not any(row.condition == "nominal" for row in raw_rows):
        print("[info] Running nominal baseline tasks before displaced camera tasks.")
        nominal_rows = _run_nominal_tasks(evaluator, args.nominal_trials_per_task)
        raw_rows.extend(nominal_rows)
        _append_trial_rows(raw_rows_path, nominal_rows)
        _write_outputs(
            evaluator,
            raw_rows,
            metadata_path=metadata_path,
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config_name,
            n_trials_per_task=args.n_trials_per_task,
            masking_trials=args.masking_trials,
            image_flip_mode=args.image_flip_mode,
            conditions=args.conditions,
            levels=args.levels,
            task_specs=args.task_specs,
            max_tasks_per_condition=args.max_tasks_per_condition,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            started_at=started_at,
            status="running",
            completed_tasks=len(completed_task_keys),
            total_tasks=len(all_tasks),
        )

    task_bar = tqdm.tqdm(
        total=len(all_tasks),
        initial=len(completed_task_keys),
        dynamic_ncols=True,
        leave=True,
        unit="task",
        desc="LIBERO-plus M0",
    )

    # Env prefetching: create the next task's N MuJoCo envs (one per trial) in a
    # background thread while the policy runs rollouts on the current task.  MuJoCo/EGL
    # init is ~2-5 s per env; hiding N*init behind rollout time keeps the GPU busier.
    # Fall back to synchronous creation silently if the background thread fails.
    pending_tasks = [t for t in all_tasks if (t.condition, t.suite_name, t.task_index) not in completed_task_keys]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        prefetch_future: concurrent.futures.Future[Any] | None = None
        if pending_tasks:
            prefetch_future = pool.submit(
                prefetch_libero_plus_envs, pending_tasks[0],
                resolution=args.resolution, seed=args.seed, n=args.n_trials_per_task,
            )

        for i, task in enumerate(all_tasks):
            key = (task.condition, task.suite_name, task.task_index)
            if key in completed_task_keys:
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
            remaining = [t for t in all_tasks[i + 1:] if (t.condition, t.suite_name, t.task_index) not in completed_task_keys]
            if remaining:
                prefetch_future = pool.submit(
                    prefetch_libero_plus_envs, remaining[0],
                    resolution=args.resolution, seed=args.seed, n=args.n_trials_per_task,
                )

            task_rows = evaluator.run_single_task(task, n_trials=args.n_trials_per_task, prebuilt_envs=prebuilt_envs)
            raw_rows.extend(task_rows)
            completed_task_keys.add(key)
            _append_trial_rows(raw_rows_path, task_rows)
            _write_outputs(
                evaluator,
                raw_rows,
                metadata_path=metadata_path,
                checkpoint_dir=args.checkpoint_dir,
                config_name=args.config_name,
                n_trials_per_task=args.n_trials_per_task,
                masking_trials=args.masking_trials,
                image_flip_mode=args.image_flip_mode,
                conditions=args.conditions,
                levels=args.levels,
                task_specs=args.task_specs,
                max_tasks_per_condition=args.max_tasks_per_condition,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
                started_at=started_at,
                status="running",
                completed_tasks=len(completed_task_keys),
                total_tasks=len(all_tasks),
            )
            n_success = sum(int(row.success) for row in task_rows)
            task_bar.set_postfix(
                cond=task.condition,
                lvl=task.level,
                sr=f"{n_success}/{len(task_rows)}",
            )
            task_bar.update(1)

    task_bar.close()

    final_metadata = {
        "status": "completed",
        "checkpoint_path": args.checkpoint_dir,
        "config_name": args.config_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials_per_task": args.n_trials_per_task,
        "masking_trials": args.masking_trials,
        "image_flip_mode": args.image_flip_mode,
        "conditions": args.conditions,
        "levels": args.levels,
        "task_specs": args.task_specs,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "include_nominal": args.include_nominal,
        "nominal_trials_per_task": args.nominal_trials_per_task,
        "max_tasks_per_condition": args.max_tasks_per_condition,
        "m6_alpha_override": args.m6_alpha_override,
        "completed_tasks": len(completed_task_keys),
        "total_tasks": len(all_tasks),
        "total_rollouts": len(raw_rows),
        "wall_time_sec": time.time() - started_at,
    }
    if not args.skip_masking:
        masking = evaluate_with_scene_camera_masked(
            policy,
            evaluator,
            n_trials=args.masking_trials,
            condition="nominal",
        )
        _write_masking_summary(masking, results_dir / "masking_check.md")
        final_metadata["masking_check"] = {
            "n_success": masking["n_success"],
            "n_total": masking["n_total"],
            "success_rate": masking["success_rate"],
            "ci_low": masking["ci_low"],
            "ci_high": masking["ci_high"],
        }
    metadata_path.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"[info] LIBERO-plus evaluation finished. Outputs written to: {results_dir}")


if __name__ == "__main__":
    main(_parse_args())
