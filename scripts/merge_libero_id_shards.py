"""Merge shard outputs from original LIBERO nominal ID evaluation."""

from __future__ import annotations

from collections import defaultdict
import csv
import dataclasses
import json
import math
import pathlib
import time
from typing import Any


@dataclasses.dataclass
class Args:
    results_dir: str = "results/id_eval/libero_nominal_parallel"
    num_shards: int = 2
    allow_partial: bool = False


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
        parser.add_argument("--results-dir", default=Args.results_dir)
        parser.add_argument("--num-shards", type=int, default=Args.num_shards)
        parser.add_argument("--allow-partial", action="store_true")
        ns = parser.parse_args()
        return Args(results_dir=ns.results_dir, num_shards=ns.num_shards, allow_partial=ns.allow_partial)


def _load_rows(path: pathlib.Path, *, required: bool = False) -> list[IDTrialResult]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing trial results file: {path}")
        return []
    rows: list[IDTrialResult] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(IDTrialResult(**json.loads(line)))
    return rows


def _load_metadata(path: pathlib.Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing metadata file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if required:
            raise
        return {}


def _load_shard_rows(shard_dir: pathlib.Path, *, allow_partial: bool = False) -> tuple[list[IDTrialResult], int]:
    rows = _load_rows(shard_dir / "trial_results.jsonl", required=not allow_partial)
    metadata = _load_metadata(shard_dir / "metadata.json", required=not allow_partial)
    if metadata and not allow_partial:
        status = metadata.get("status")
        completed = metadata.get("completed_tasks")
        total = metadata.get("total_tasks")
        if status != "completed":
            raise ValueError(f"{shard_dir}: shard status is {status!r}, expected 'completed'.")
        if completed is not None and total is not None and int(completed) != int(total):
            raise ValueError(f"{shard_dir}: completed_tasks={completed}, total_tasks={total}.")

    expected = metadata.get("total_rollouts")
    if expected is None and not allow_partial:
        raise ValueError(f"{shard_dir}: metadata.json is missing total_rollouts.")
    if expected is None:
        return rows, 0
    expected = int(expected)
    if len(rows) < expected:
        raise ValueError(
            f"{shard_dir}: trial_results.jsonl has {len(rows)} rows, "
            f"but metadata.json reports total_rollouts={expected}."
        )
    stale_prefix = len(rows) - expected
    if stale_prefix > 0:
        rows = rows[-expected:] if expected else []
    return rows, stale_prefix


def _dedupe_rows(rows: list[IDTrialResult]) -> tuple[list[IDTrialResult], int]:
    seen: dict[tuple[str, int], IDTrialResult] = {}
    deduped: list[IDTrialResult] = []
    dropped = 0
    for row in rows:
        key = (row.task_spec, row.trial_index)
        if key in seen:
            old = seen[key]
            if (
                old.success != row.success
                or old.done != row.done
                or old.steps_executed != row.steps_executed
                or old.rollout_seed != row.rollout_seed
            ):
                raise ValueError(
                    "Conflicting duplicate ID rollout rows for "
                    f"{key}: first={dataclasses.asdict(old)} duplicate={dataclasses.asdict(row)}"
                )
            dropped += 1
            continue
        seen[key] = row
        deduped.append(row)
    return deduped, dropped


def _wilson_ci(n_success: int, n_total: int) -> tuple[float, float]:
    if n_total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    phat = n_success / n_total
    denom = 1.0 + (z * z) / n_total
    center = (phat + (z * z) / (2.0 * n_total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n_total) + (z * z) / (4.0 * n_total * n_total))
    return max(0.0, center - margin), min(1.0, center + margin)


def _aggregate_by_task(rows: list[IDTrialResult]) -> list[dict[str, Any]]:
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


def _aggregate_by_suite(rows: list[IDTrialResult]) -> list[dict[str, Any]]:
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
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
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


def main(args: Args) -> None:
    results_dir = pathlib.Path(args.results_dir)
    shard_root = results_dir / "shards"
    rows: list[IDTrialResult] = []
    stale_prefix_rollouts_dropped = 0
    for shard_idx in range(args.num_shards):
        shard_rows, stale_prefix = _load_shard_rows(
            shard_root / f"shard_{shard_idx:02d}",
            allow_partial=args.allow_partial,
        )
        rows.extend(shard_rows)
        stale_prefix_rollouts_dropped += stale_prefix

    rows, dropped_duplicates = _dedupe_rows(rows)
    per_task_rows = _aggregate_by_task(rows)
    aggregate_rows = _aggregate_by_suite(rows)
    _write_csv(per_task_rows, results_dir / "per_task_results.csv")
    _write_csv(aggregate_rows, results_dir / "aggregate_results.csv")
    _write_markdown(aggregate_rows, results_dir / "aggregate_summary.md")

    with (results_dir / "trial_results.jsonl").open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row), ensure_ascii=True) + "\n")

    metadata = {
        "status": "merged",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_shards": args.num_shards,
        "total_rollouts": len(rows),
        "completed_tasks": len(per_task_rows),
        "dropped_duplicate_rollouts": dropped_duplicates,
        "dropped_stale_prefix_rollouts": stale_prefix_rollouts_dropped,
    }
    (results_dir / "merged_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[info] Merged {len(rows)} ID trial rows into {results_dir} "
        f"(dropped {stale_prefix_rollouts_dropped} stale prefix rows, "
        f"{dropped_duplicates} exact duplicates)"
    )


if __name__ == "__main__":
    main(_parse_args())
