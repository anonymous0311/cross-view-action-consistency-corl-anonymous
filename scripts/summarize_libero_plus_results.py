"""Post-process LIBERO-plus camera evaluation outputs into analysis tables."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import re
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMERA_TASK_RE = re.compile(
    r"^(?P<base>.+)_view_(?P<hv>\d+)_(?P<vv>\d+)_(?P<scale>\d+)_(?P<rotz>\d+)_(?P<roty>\d+)_initstate_(?P<init>\d+)$"
)
SUMMARY_CONDITIONS = ("C1", "C2", "C3")
WILSON_Z = 1.959963984540054


@dataclasses.dataclass
class Args:
    results_dir: str = "results/main_table/m0_parallel"
    method_id: str = "m0"
    num_shards: int = 6
    sample_count: int = 3


def _resolve_task_classification_path() -> pathlib.Path:
    candidates = [
        REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "benchmark" / "task_classification.json",
        REPO_ROOT.parents[1] / "code" / "LIBERO-plus" / "libero" / "libero" / "benchmark" / "task_classification.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find LIBERO-plus task_classification.json in the current repo or sibling main repo."
    )


def _parse_args() -> Args:
    try:
        import tyro  # noqa: PLC0415

        return tyro.cli(Args)
    except ModuleNotFoundError:
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("--results-dir", default=Args.results_dir)
        parser.add_argument("--method-id", default=Args.method_id)
        parser.add_argument("--num-shards", type=int, default=Args.num_shards)
        parser.add_argument("--sample-count", type=int, default=Args.sample_count)
        ns = parser.parse_args()
        return Args(
            results_dir=ns.results_dir,
            method_id=ns.method_id,
            num_shards=ns.num_shards,
            sample_count=ns.sample_count,
        )


def _load_trial_rows(results_dir: pathlib.Path, num_shards: int) -> list[dict[str, object]]:
    merged_path = results_dir / "trial_results.jsonl"
    candidate_paths: list[pathlib.Path]
    if merged_path.exists():
        candidate_paths = [merged_path]
    else:
        candidate_paths = [
            results_dir / "shards" / f"shard_{shard_idx:02d}" / "trial_results.jsonl" for shard_idx in range(num_shards)
        ]

    rows: list[dict[str, object]] = []
    for path in candidate_paths:
        if not path.exists():
            continue
        metadata_path = path.parent / "metadata.json"
        expected_rollouts: int | None = None
        if path.name == "trial_results.jsonl" and metadata_path.exists() and path.parent.name.startswith("shard_"):
            try:
                expected_raw = json.loads(metadata_path.read_text(encoding="utf-8")).get("total_rollouts")
                expected_rollouts = int(expected_raw) if expected_raw is not None else None
            except (json.JSONDecodeError, TypeError, ValueError):
                expected_rollouts = None
        path_rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    path_rows.append(json.loads(line))
        if expected_rollouts is not None and len(path_rows) > expected_rollouts:
            path_rows = path_rows[-expected_rollouts:] if expected_rollouts else []
        rows.extend(path_rows)
    return rows


def _dedupe_trial_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: dict[tuple[str, str, int, int, bool], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["suite_name"]),
            int(row["task_index"]),
            int(row["trial_index"]),
            bool(row.get("masked_scene", False)),
        )
        if key in seen:
            old = seen[key]
            if (
                bool(old.get("success")) != bool(row.get("success"))
                or bool(old.get("done")) != bool(row.get("done"))
                or int(old.get("steps_executed", -1)) != int(row.get("steps_executed", -1))
                or int(old.get("rollout_seed", -1)) != int(row.get("rollout_seed", -1))
            ):
                raise ValueError(f"Conflicting duplicate rollout rows for {key}: first={old} duplicate={row}")
            continue
        seen[key] = row
        deduped.append(row)
    return deduped


def _load_metadata(results_dir: pathlib.Path) -> dict[str, Any]:
    for candidate in (results_dir / "metadata.json", results_dir / "merged_metadata.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _wilson_ci(n_success: int, n_total: int) -> tuple[float, float]:
    if n_total <= 0:
        return 0.0, 0.0
    phat = n_success / n_total
    denom = 1.0 + (WILSON_Z**2) / n_total
    center = (phat + (WILSON_Z**2) / (2.0 * n_total)) / denom
    margin = (WILSON_Z / denom) * math.sqrt((phat * (1.0 - phat) / n_total) + (WILSON_Z**2) / (4.0 * n_total**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def _signed_deg(raw_value: str | int) -> int:
    value = int(raw_value)
    return value - 360 if value > 180 else value


def _classify_condition(match: re.Match[str]) -> str | None:
    hv = int(match.group("hv"))
    vv = int(match.group("vv"))
    scale = int(match.group("scale"))
    rotz = int(match.group("rotz"))
    roty = int(match.group("roty"))
    if scale != 100:
        return "C1"
    if rotz != 0 or roty != 0:
        return "C3"
    if hv != 0 or vv != 0:
        return "C2"
    return None


def _aggregate_by_level(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    levels = [1, 2, 3, 4, 5]
    grouped = {(level, condition): {"n_success": 0, "n_total": 0} for level in levels for condition in SUMMARY_CONDITIONS}
    for row in rows:
        key = (int(row["level"]), str(row["condition"]))
        if key not in grouped:
            continue
        grouped[key]["n_total"] += 1
        grouped[key]["n_success"] += int(bool(row["success"]))

    out: list[dict[str, object]] = []
    for level in levels:
        record: dict[str, object] = {"level": level}
        for condition in SUMMARY_CONDITIONS:
            cell = grouped[(level, condition)]
            n_total = int(cell["n_total"])
            n_success = int(cell["n_success"])
            record[f"{condition}_n_success"] = n_success
            record[f"{condition}_n_total"] = n_total
            record[f"{condition}_success_rate"] = (n_success / n_total) if n_total else ""
        out.append(record)
    return out


def _write_level_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    fieldnames = [
        "level",
        "C1_success_rate",
        "C1_n_success",
        "C1_n_total",
        "C2_success_rate",
        "C2_n_success",
        "C2_n_total",
        "C3_success_rate",
        "C3_n_success",
        "C3_n_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_level_markdown(
    rows: list[dict[str, object]],
    path: pathlib.Path,
    *,
    method_id: str,
    total_rollouts: int,
) -> None:
    method_label = method_id.upper()
    lines = [
        f"# {method_label} By Level",
        "",
        "Snapshot over completed LIBERO-plus camera rollouts currently available under the result directory.",
        "",
        f"- Total completed rollouts included: `{total_rollouts}`",
        "",
        "| Level | C1 | C2 | C3 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        cells: list[str] = []
        for condition in SUMMARY_CONDITIONS:
            n_total = int(row[f"{condition}_n_total"])
            n_success = int(row[f"{condition}_n_success"])
            if n_total:
                cells.append(f"{100 * n_success / n_total:.1f}% ({n_success}/{n_total})")
            else:
                cells.append("N/A (0/0)")
        lines.append(f"| L{row['level']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_camera_parameters(sample_count: int) -> list[dict[str, object]]:
    raw = json.loads(_resolve_task_classification_path().read_text(encoding="utf-8"))
    buckets: dict[tuple[str, int], list[dict[str, object]]] = {}
    for suite_name, entries in raw.items():
        for task_index, entry in enumerate(entries):
            if entry.get("category") != "Camera Viewpoints":
                continue
            match = CAMERA_TASK_RE.match(str(entry["name"]))
            if match is None:
                continue
            condition = _classify_condition(match)
            if condition not in SUMMARY_CONDITIONS:
                continue
            level = int(entry["difficulty_level"])
            if level not in {1, 5}:
                continue
            buckets.setdefault((condition, level), []).append(
                {
                    "condition": condition,
                    "level": level,
                    "suite": suite_name,
                    "task_index": task_index,
                    "json_id": int(entry["id"]),
                    "task_name": str(entry["name"]),
                    "distance_multiplier": int(match.group("scale")) / 100.0,
                    "azimuth_offset_deg": _signed_deg(match.group("hv")),
                    "elevation_offset_deg": _signed_deg(match.group("vv")),
                    "yaw_offset_deg": _signed_deg(match.group("rotz")),
                    "pitch_offset_deg": _signed_deg(match.group("roty")),
                    "roll_offset_deg": 0,
                }
            )

    selected: list[dict[str, object]] = []
    for condition in SUMMARY_CONDITIONS:
        for level in (1, 5):
            group = sorted(
                buckets.get((condition, level), []),
                key=lambda row: (str(row["suite"]), int(row["task_index"]), str(row["task_name"])),
            )[:sample_count]
            selected.extend(group)
    return selected


def _write_samples_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    fieldnames = [
        "condition",
        "level",
        "suite",
        "task_index",
        "json_id",
        "task_name",
        "distance_multiplier",
        "azimuth_offset_deg",
        "elevation_offset_deg",
        "yaw_offset_deg",
        "pitch_offset_deg",
        "roll_offset_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_samples_markdown(rows: list[dict[str, object]], path: pathlib.Path, sample_count: int) -> None:
    lines = [
        "# Camera Parameter Samples",
        "",
        f"Deterministic sample: first {sample_count} tasks after sorting by `(suite, task_index, task_name)` within each `(condition, level)` bucket.",
        "",
        "Angle fields below are shown as signed degrees for readability. `roll_offset_deg` is `0` here because the task-name encoding only exposes two orientation offsets.",
        "",
    ]
    for condition in SUMMARY_CONDITIONS:
        for level in (1, 5):
            lines.extend(
                [
                    f"## {condition} L{level}",
                    "",
                    "| Suite | Task Index | JSON ID | Distance | Azimuth | Elevation | Yaw | Pitch | Roll | Task Name |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in [item for item in rows if item["condition"] == condition and item["level"] == level]:
                lines.append(
                    f"| {row['suite']} | {row['task_index']} | {row['json_id']} | "
                    f"{float(row['distance_multiplier']):.2f}x | "
                    f"{row['azimuth_offset_deg']} deg | {row['elevation_offset_deg']} deg | "
                    f"{row['yaw_offset_deg']} deg | {row['pitch_offset_deg']} deg | "
                    f"{row['roll_offset_deg']} deg | {row['task_name']} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_summary_groups(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, dict[str, int]] = {"All": {"n_success": 0, "n_total": 0}}
    for condition in SUMMARY_CONDITIONS:
        grouped[condition] = {"n_success": 0, "n_total": 0}

    for row in rows:
        condition = str(row["condition"])
        success = int(bool(row["success"]))
        if condition in grouped:
            grouped[condition]["n_success"] += success
            grouped[condition]["n_total"] += 1
        grouped["All"]["n_success"] += success
        grouped["All"]["n_total"] += 1

    out: dict[str, dict[str, float | int]] = {}
    for condition, counts in grouped.items():
        n_success = int(counts["n_success"])
        n_total = int(counts["n_total"])
        ci_low, ci_high = _wilson_ci(n_success, n_total)
        out[condition] = {
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": (n_success / n_total) if n_total else 0.0,
            "ci_low_percent": round(100.0 * ci_low, 1),
            "ci_high_percent": round(100.0 * ci_high, 1),
        }
    return out


def _write_main_table_row(
    results_dir: pathlib.Path,
    *,
    method_id: str,
    trial_rows: list[dict[str, object]],
    metadata: dict[str, Any],
) -> pathlib.Path:
    summary = _compute_summary_groups(trial_rows)
    output_path = results_dir.parent / f"{method_id}.csv"
    fieldnames = [
        "checkpoint",
        "config_name",
        "trials_per_task",
        "total_tasks",
        "total_rollouts",
        "aggregate_success_rate",
        "aggregate_ci_low",
        "aggregate_ci_high",
        "C1_success_rate",
        "C1_ci_low",
        "C1_ci_high",
        "C2_success_rate",
        "C2_ci_low",
        "C2_ci_high",
        "C3_success_rate",
        "C3_ci_low",
        "C3_ci_high",
    ]
    row = {
        "checkpoint": metadata.get("checkpoint_path", ""),
        "config_name": metadata.get("config_name", ""),
        "trials_per_task": metadata.get("n_trials_per_task", ""),
        "total_tasks": len({(str(item["condition"]), str(item["suite_name"]), int(item["task_index"])) for item in trial_rows}),
        "total_rollouts": len(trial_rows),
        "aggregate_success_rate": round(100.0 * float(summary["All"]["success_rate"]), 1),
        "aggregate_ci_low": summary["All"]["ci_low_percent"],
        "aggregate_ci_high": summary["All"]["ci_high_percent"],
        "C1_success_rate": round(100.0 * float(summary["C1"]["success_rate"]), 1),
        "C1_ci_low": summary["C1"]["ci_low_percent"],
        "C1_ci_high": summary["C1"]["ci_high_percent"],
        "C2_success_rate": round(100.0 * float(summary["C2"]["success_rate"]), 1),
        "C2_ci_low": summary["C2"]["ci_low_percent"],
        "C2_ci_high": summary["C2"]["ci_high_percent"],
        "C3_success_rate": round(100.0 * float(summary["C3"]["success_rate"]), 1),
        "C3_ci_low": summary["C3"]["ci_low_percent"],
        "C3_ci_high": summary["C3"]["ci_high_percent"],
    }
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return output_path


def main(args: Args) -> None:
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = _dedupe_trial_rows(_load_trial_rows(results_dir, args.num_shards))
    metadata = _load_metadata(results_dir)
    by_level_rows = _aggregate_by_level(trial_rows)
    sampled_rows = _sample_camera_parameters(args.sample_count)

    _write_level_csv(by_level_rows, results_dir / f"{args.method_id}_by_level.csv")
    _write_level_markdown(
        by_level_rows,
        results_dir / f"{args.method_id}_by_level.md",
        method_id=args.method_id,
        total_rollouts=len(trial_rows),
    )
    _write_samples_csv(sampled_rows, results_dir / "camera_parameter_samples.csv")
    _write_samples_markdown(sampled_rows, results_dir / "camera_parameter_samples.md", args.sample_count)
    main_table_path = _write_main_table_row(results_dir, method_id=args.method_id, trial_rows=trial_rows, metadata=metadata)
    print(f"[info] Wrote summary outputs to {results_dir}")
    print(f"[info] Wrote main-table row to {main_table_path}")


if __name__ == "__main__":
    main(_parse_args())
