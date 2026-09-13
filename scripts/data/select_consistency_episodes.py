"""Select nested 5% and 25% episode sets, stratified by task, for CV masking."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/episode_membership.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = defaultdict(list)
    for row in rows:
        tasks[tuple(row["tasks"])].append(int(row["episode_index"]))
    rng = np.random.default_rng(args.seed)
    orders = {task: rng.permutation(sorted(ids)).tolist() for task, ids in sorted(tasks.items())}
    settings = {}
    for percentage in [0, 5, 25, 100]:
        counts = {task: int(len(order) * percentage // 100) for task, order in orders.items()}
        target = round(len(rows) * percentage / 100)
        # Allocate rounding remainders across tasks to keep the global fraction exact.
        priority = list(orders)
        rng.shuffle(priority)
        priority.sort(key=lambda task: len(orders[task]) * percentage / 100 - counts[task], reverse=True)
        for task in priority[: target - sum(counts.values())]:
            counts[task] += 1
        selected = sorted(i for task, order in orders.items() for i in order[: counts[task]])
        settings[str(percentage)] = {
            "selected_export_episode_indices": selected,
            "selected_episode_count": len(selected),
        }
    output = {"status": "pass", "selection_seed": args.seed, "stratification": "task", "settings": settings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print({p: v["selected_episode_count"] for p, v in settings.items()})


if __name__ == "__main__":
    main()
