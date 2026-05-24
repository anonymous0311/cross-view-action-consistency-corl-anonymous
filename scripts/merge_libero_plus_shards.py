"""Merge shard outputs from parallel LIBERO-plus camera evaluation."""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import sys
import time
from typing import Any, Literal


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for rel_path in ("openpi/src", "openpi/packages/openpi-client/src", "LIBERO-plus"):
    abs_path = REPO_ROOT / rel_path
    if abs_path.exists() and str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))

from canonical.eval.libero_plus_eval import TrialResult
from canonical.eval.libero_plus_eval import wilson_ci


@dataclasses.dataclass
class Args:
    results_dir: str = "results/main_table/m0_parallel"
    num_shards: int = 6
    allow_partial: bool = False
    duplicate_policy: Literal["strict", "first", "last"] = "strict"
    stale_prefix_policy: Literal["drop", "keep"] = "drop"


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
        parser.add_argument(
            "--duplicate-policy",
            default=Args.duplicate_policy,
            choices=("strict", "first", "last"),
            help=(
                "How to handle duplicate rollout keys left by interrupted/resumed runs. "
                "'strict' raises on any metric-relevant conflict; 'first'/'last' keep that occurrence."
            ),
        )
        parser.add_argument(
            "--stale-prefix-policy",
            default=Args.stale_prefix_policy,
            choices=("drop", "keep"),
            help=(
                "How to handle shard files longer than metadata.total_rollouts. "
                "'drop' keeps the historical tail behavior; 'keep' keeps all raw rows and relies on duplicate de-duping."
            ),
        )
        ns = parser.parse_args()
        return Args(
            results_dir=ns.results_dir,
            num_shards=ns.num_shards,
            allow_partial=ns.allow_partial,
            duplicate_policy=ns.duplicate_policy,
            stale_prefix_policy=ns.stale_prefix_policy,
        )


def _load_rows(path: pathlib.Path, *, required: bool = False) -> list[TrialResult]:
    rows: list[TrialResult] = []
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing trial results file: {path}")
        return rows
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(TrialResult(**json.loads(line)))
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


def _load_shard_rows(
    shard_dir: pathlib.Path,
    *,
    allow_partial: bool = False,
    stale_prefix_policy: Literal["drop", "keep"] = "drop",
) -> tuple[list[TrialResult], int]:
    """Load one shard and drop stale prefix rows left by an older non-resumed run.

    `evaluate_libero_plus.py` appends trial rows as it runs. If a result
    directory is accidentally reused without `--resume`, stale rows can remain
    at the beginning of `trial_results.jsonl`, while the live worker's
    `metadata.json.total_rollouts` records only the current process rows.
    In that case, the current run is the tail of the file.
    """
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
    if expected < 0:
        raise ValueError(f"Invalid total_rollouts={expected} in {shard_dir / 'metadata.json'}")
    if len(rows) < expected:
        raise ValueError(
            f"{shard_dir}: trial_results.jsonl has {len(rows)} rows, "
            f"but metadata.json reports total_rollouts={expected}."
        )
    stale_prefix = len(rows) - expected
    if stale_prefix > 0:
        if stale_prefix_policy == "drop":
            rows = rows[-expected:] if expected else []
            return rows, stale_prefix
        if stale_prefix_policy == "keep":
            return rows, 0
        raise ValueError(f"Unknown stale_prefix_policy: {stale_prefix_policy}")
    return rows, 0


def _dedupe_rows(
    rows: list[TrialResult],
    *,
    duplicate_policy: Literal["strict", "first", "last"] = "strict",
) -> tuple[list[TrialResult], int, int]:
    deduped: list[TrialResult] = []
    seen: dict[tuple[str, str, int, int, bool], TrialResult] = {}
    seen_index: dict[tuple[str, str, int, int, bool], int] = {}
    dropped = 0
    conflicting = 0
    for row in rows:
        key = (row.condition, row.suite_name, row.task_index, row.trial_index, row.masked_scene)
        if key in seen:
            old = seen[key]
            is_conflict = (
                old.success != row.success
                or old.done != row.done
                or old.steps_executed != row.steps_executed
                or old.rollout_seed != row.rollout_seed
            )
            if is_conflict and duplicate_policy == "strict":
                raise ValueError(
                    "Conflicting duplicate rollout rows for "
                    f"{key}: first={dataclasses.asdict(old)} duplicate={dataclasses.asdict(row)}"
                )
            if is_conflict:
                conflicting += 1
            if duplicate_policy == "last":
                seen[key] = row
                deduped[seen_index[key]] = row
            dropped += 1
            continue
        seen[key] = row
        seen_index[key] = len(deduped)
        deduped.append(row)
    return deduped, dropped, conflicting


def _common_metadata(shard_metadatas: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        values = {metadata.get(key) for metadata in shard_metadatas if key in metadata}
        if len(values) == 1:
            out[key] = values.pop()
    return out


def _compute_per_task(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault((row.condition, row.suite_name, row.task_index), []).append(row)
    out: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        exemplar = group_rows[0]
        n_total = len(group_rows)
        n_success = sum(int(row.success) for row in group_rows)
        ci_low, ci_high = wilson_ci(n_success, n_total)
        out.append(
            {
                "condition": exemplar.condition,
                "suite_name": exemplar.suite_name,
                "task_id": exemplar.task_index,
                "json_id": exemplar.json_id,
                "task_name": exemplar.task_name,
                "level": exemplar.level,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def _compute_aggregate(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[TrialResult]] = {}
    condition_grouped: dict[str, list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault((row.condition, row.level), []).append(row)
        condition_grouped.setdefault(row.condition, []).append(row)

    out: list[dict[str, Any]] = []
    for (condition, level), group_rows in sorted(grouped.items()):
        n_total = len(group_rows)
        n_success = sum(int(row.success) for row in group_rows)
        ci_low, ci_high = wilson_ci(n_success, n_total)
        out.append(
            {
                "condition": condition,
                "level": level,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    for condition, group_rows in sorted(condition_grouped.items()):
        n_total = len(group_rows)
        n_success = sum(int(row.success) for row in group_rows)
        ci_low, ci_high = wilson_ci(n_success, n_total)
        out.append(
            {
                "condition": condition,
                "level": "Aggregate",
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
        ci_low, ci_high = wilson_ci(n_success, n_total)
        out.append(
            {
                "condition": "All",
                "level": "Aggregate",
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    fieldnames = [
        "condition",
        "suite_name",
        "task_id",
        "json_id",
        "task_name",
        "level",
        "n_success",
        "n_total",
        "success_rate",
        "ci_low",
        "ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "# LIBERO-plus Camera Evaluation",
        "",
        "| Condition | Level | Success Rate (95% Wilson CI) | n_success / n_total |",
        "|---|---:|---:|---:|",
    ]

    def _level_key(value: Any) -> tuple[int, str]:
        if isinstance(value, int):
            return (0, f"{value:02d}")
        return (1, str(value))

    for row in sorted(rows, key=lambda item: (str(item["condition"]), _level_key(item["level"]))):
        lines.append(
            "| "
            f"{row['condition']} | {row['level']} | "
            f"{100 * float(row['success_rate']):.1f}% "
            f"([{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]) | "
            f"{row['n_success']} / {row['n_total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    results_dir = pathlib.Path(args.results_dir)
    shard_root = results_dir / "shards"
    rows: list[TrialResult] = []
    shard_metadatas: list[dict[str, Any]] = []
    stale_prefix_rollouts_dropped = 0
    for shard_idx in range(args.num_shards):
        shard_dir = shard_root / f"shard_{shard_idx:02d}"
        shard_rows, stale_prefix = _load_shard_rows(
            shard_dir,
            allow_partial=args.allow_partial,
            stale_prefix_policy=args.stale_prefix_policy,
        )
        shard_metadatas.append(_load_metadata(shard_dir / "metadata.json"))
        rows.extend(shard_rows)
        stale_prefix_rollouts_dropped += stale_prefix
    rows, dropped_duplicates, conflicting_duplicates = _dedupe_rows(rows, duplicate_policy=args.duplicate_policy)

    per_task_rows = _compute_per_task(rows)
    aggregate_rows = _compute_aggregate(rows)
    _write_csv(per_task_rows, results_dir / "per_task_results.csv")
    _write_markdown(aggregate_rows, results_dir / "aggregate_summary.md")

    combined_raw_path = results_dir / "trial_results.jsonl"
    with combined_raw_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dataclasses.asdict(row)) + "\n")

    metadata = {
        "status": "merged",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_shards": args.num_shards,
        "total_rollouts": len(rows),
        "completed_tasks": len(per_task_rows),
        "dropped_duplicate_rollouts": dropped_duplicates,
        "conflicting_duplicate_rollouts": conflicting_duplicates,
        "duplicate_policy": args.duplicate_policy,
        "dropped_stale_prefix_rollouts": stale_prefix_rollouts_dropped,
    }
    metadata.update(
        _common_metadata(
            shard_metadatas,
            (
                "checkpoint_path",
                "config_name",
                "n_trials_per_task",
                "masking_trials",
                "image_flip_mode",
                "conditions",
                "levels",
                "task_specs",
                "max_tasks_per_condition",
            ),
        )
    )
    (results_dir / "merged_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[info] Merged {len(rows)} trial rows into {results_dir} "
        f"(dropped {stale_prefix_rollouts_dropped} stale prefix rows, "
        f"{dropped_duplicates} duplicates, "
        f"{conflicting_duplicates} conflicting duplicates, "
        f"policy={args.duplicate_policy})"
    )


if __name__ == "__main__":
    main(_parse_args())
