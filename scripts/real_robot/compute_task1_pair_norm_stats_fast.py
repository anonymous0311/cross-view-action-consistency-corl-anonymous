#!/usr/bin/env python3
"""Compute Task1_CoRL pair norm stats from parquet numeric columns only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq

REPO_DEFAULT = Path("data/real_robot/task1_pair")
OUTPUT_DEFAULT = Path("assets/pi05_realman_task1_b6b/anonymous/realman_task1_corl")


def _fixed_list_column_to_numpy(table, name: str) -> np.ndarray:
    values = table[name].combine_chunks().values
    width = table.schema.field(name).type.list_size
    return np.asarray(values).reshape(table.num_rows, width).astype(np.float32)


def compute_stats(repo: Path, output: Path, *, episodes: list[int], action_horizon: int) -> None:
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for ep_idx in episodes:
        table = pq.read_table(repo / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        state = _fixed_list_column_to_numpy(table, "observation.state")
        action = _fixed_list_column_to_numpy(table, "action")
        length = action.shape[0]
        chunk_indices = np.minimum(
            np.arange(length)[:, None] + np.arange(action_horizon)[None, :],
            length - 1,
        )
        action_chunks = action[chunk_indices]

        stats["state"].update(state)
        stats["actions"].update(action_chunks)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(output, norm_stats)
    print(f"Wrote norm stats: {output / 'norm_stats.json'}")
    print(f"episodes: {len(episodes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stop", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=10)
    args = parser.parse_args()

    compute_stats(
        args.repo,
        args.output,
        episodes=list(range(args.episode_start, args.episode_stop)),
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
