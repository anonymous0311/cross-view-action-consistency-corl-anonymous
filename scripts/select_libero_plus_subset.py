"""Select a deterministic representative LIBERO-plus camera-eval subset."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
import csv
import dataclasses
import json
import math
import pathlib
import random
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for rel_path in ("openpi/src", "openpi/packages/openpi-client/src", "LIBERO-plus"):
    abs_path = REPO_ROOT / rel_path
    if abs_path.exists() and str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))

from canonical.eval.libero_plus_eval import CameraTask
from canonical.eval.libero_plus_eval import ensure_libero_plus_runtime
from canonical.eval.libero_plus_eval import enumerate_camera_tasks


CONDITION_ORDER = {"C1": 0, "C2": 1, "C3": 2}


@dataclasses.dataclass(frozen=True)
class Args:
    num_tasks: int = 120
    seed: int = 7
    output_dir: str = "results/libero_plus_subsets/screen_120_s7"
    min_one_per_stratum: bool = True


def _parse_args() -> Args:
    try:
        import tyro  # noqa: PLC0415

        return tyro.cli(Args)
    except ModuleNotFoundError:
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("--num-tasks", type=int, default=Args.num_tasks)
        parser.add_argument("--seed", type=int, default=Args.seed)
        parser.add_argument("--output-dir", default=Args.output_dir)
        parser.add_argument("--no-min-one-per-stratum", action="store_true")
        ns = parser.parse_args()
        return Args(
            num_tasks=ns.num_tasks,
            seed=ns.seed,
            output_dir=ns.output_dir,
            min_one_per_stratum=not ns.no_min_one_per_stratum,
        )


def _task_key(task: CameraTask) -> tuple[int, str, int, int]:
    return (CONDITION_ORDER[task.condition], task.suite_name, task.level, task.task_index)


def _stratum(task: CameraTask) -> tuple[str, str, int]:
    return (task.condition, task.suite_name, task.level)


def _allocate_counts(
    full_counts: dict[Any, int],
    *,
    target: int,
    min_one_per_stratum: bool,
) -> dict[Any, int]:
    if target <= 0:
        raise ValueError("--num-tasks must be positive.")
    total = sum(full_counts.values())
    if target > total:
        raise ValueError(f"--num-tasks={target} exceeds full task count {total}.")

    strata = sorted(full_counts)
    if min_one_per_stratum and target >= len(strata):
        allocation = {key: 1 for key in strata}
        remaining = target - len(strata)
        quotas = {key: (full_counts[key] / total) * remaining for key in strata}
    else:
        allocation = {key: 0 for key in strata}
        remaining = target
        quotas = {key: (full_counts[key] / total) * target for key in strata}

    floors = {key: int(math.floor(value)) for key, value in quotas.items()}
    for key, value in floors.items():
        allocation[key] += value
    used = sum(allocation.values())

    remainders = sorted(
        ((quotas[key] - floors[key], full_counts[key], key) for key in strata),
        key=lambda item: (-item[0], -item[1], str(item[2])),
    )
    for _frac, _count, key in remainders:
        if used >= target:
            break
        if allocation[key] < full_counts[key]:
            allocation[key] += 1
            used += 1

    if used != target:
        raise RuntimeError(f"Internal allocation error: requested {target}, allocated {used}.")
    return allocation


def _allocate_stratified_counts(tasks: list[CameraTask], target: int, min_one_per_stratum: bool) -> dict[tuple[str, str, int], int]:
    condition_counts = Counter(task.condition for task in tasks)
    per_condition = _allocate_counts(
        dict(condition_counts),
        target=target,
        min_one_per_stratum=False,
    )

    allocation: dict[tuple[str, str, int], int] = {}
    for condition, condition_target in per_condition.items():
        condition_tasks = [task for task in tasks if task.condition == condition]
        stratum_counts = dict(_counter(condition_tasks, "stratum"))
        allocation.update(
            _allocate_counts(
                stratum_counts,
                target=condition_target,
                min_one_per_stratum=min_one_per_stratum,
            )
        )
    if sum(allocation.values()) != target:
        raise RuntimeError("Internal allocation error across conditions.")
    return allocation


def _select_tasks(tasks: list[CameraTask], allocation: dict[tuple[str, str, int], int], seed: int) -> list[CameraTask]:
    grouped: dict[tuple[str, str, int], list[CameraTask]] = defaultdict(list)
    for task in tasks:
        grouped[_stratum(task)].append(task)

    selected: list[CameraTask] = []
    for key, count in allocation.items():
        candidates = sorted(grouped[key], key=_task_key)
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}:{key[2]}")
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    return sorted(selected, key=_task_key)


def _counter(tasks: list[CameraTask], field: str) -> Counter[Any]:
    if field == "condition":
        return Counter(task.condition for task in tasks)
    if field == "suite":
        return Counter(task.suite_name for task in tasks)
    if field == "level":
        return Counter(task.level for task in tasks)
    if field == "condition_suite":
        return Counter((task.condition, task.suite_name) for task in tasks)
    if field == "condition_level":
        return Counter((task.condition, task.level) for task in tasks)
    if field == "stratum":
        return Counter(_stratum(task) for task in tasks)
    raise ValueError(f"Unknown field: {field}")


def _write_count_table(
    lines: list[str],
    title: str,
    full: Counter[Any],
    subset: Counter[Any],
    *,
    full_total: int,
    subset_total: int,
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Key | Full n | Full % | Subset n | Subset % | Delta pp |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    keys = sorted(set(full) | set(subset), key=str)
    for key in keys:
        f_n = int(full.get(key, 0))
        s_n = int(subset.get(key, 0))
        f_pct = f_n / full_total if full_total else 0.0
        s_pct = s_n / subset_total if subset_total else 0.0
        lines.append(f"| `{key}` | {f_n} | {100*f_pct:.1f} | {s_n} | {100*s_pct:.1f} | {100*(s_pct-f_pct):+.1f} |")
    lines.append("")


def _write_outputs(tasks: list[CameraTask], selected: list[CameraTask], args: Args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_specs = [task.task_spec for task in selected]
    (output_dir / "task_specs_arg.txt").write_text(",".join(task_specs) + "\n", encoding="utf-8")
    (output_dir / "task_specs.txt").write_text("\n".join(task_specs) + "\n", encoding="utf-8")

    fieldnames = [
        "task_spec",
        "condition",
        "suite_name",
        "task_index",
        "json_id",
        "level",
        "task_name",
    ]
    with (output_dir / "subset_tasks.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for task in selected:
            writer.writerow(
                {
                    "task_spec": task.task_spec,
                    "condition": task.condition,
                    "suite_name": task.suite_name,
                    "task_index": task.task_index,
                    "json_id": task.json_id,
                    "level": task.level,
                    "task_name": task.task_name,
                }
            )

    with (output_dir / "subset_tasks.jsonl").open("w", encoding="utf-8") as file_obj:
        for task in selected:
            file_obj.write(json.dumps(dataclasses.asdict(task), sort_keys=True) + "\n")

    metadata = {
        "num_tasks": len(selected),
        "full_num_tasks": len(tasks),
        "seed": args.seed,
        "min_one_per_stratum": args.min_one_per_stratum,
        "stratification": ["condition", "suite_name", "level"],
        "task_specs_arg": str(output_dir / "task_specs_arg.txt"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# LIBERO-plus Representative Subset",
        "",
        f"- Full tasks: `{len(tasks)}`",
        f"- Subset tasks: `{len(selected)}`",
        f"- Seed: `{args.seed}`",
        "- Stratification: `(condition, suite_name, level)`",
        "- CLI task specs: `task_specs_arg.txt`",
        "",
    ]
    for title, field in (
        ("Condition", "condition"),
        ("Suite", "suite"),
        ("Difficulty Level", "level"),
        ("Condition x Suite", "condition_suite"),
        ("Condition x Level", "condition_level"),
    ):
        _write_count_table(
            lines,
            title,
            _counter(tasks, field),
            _counter(selected, field),
            full_total=len(tasks),
            subset_total=len(selected),
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: Args) -> None:
    ensure_libero_plus_runtime()
    tasks = sorted(enumerate_camera_tasks(), key=_task_key)
    allocation = _allocate_stratified_counts(
        tasks,
        target=args.num_tasks,
        min_one_per_stratum=args.min_one_per_stratum,
    )
    selected = _select_tasks(tasks, allocation, args.seed)
    _write_outputs(tasks, selected, args)
    print(f"[info] selected {len(selected)}/{len(tasks)} tasks -> {args.output_dir}")
    print(f"[info] task specs: {pathlib.Path(args.output_dir) / 'task_specs_arg.txt'}")


if __name__ == "__main__":
    main(_parse_args())
