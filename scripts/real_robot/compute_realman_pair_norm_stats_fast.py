#!/usr/bin/env python3
"""Compute realman pair norm stats from parquet numeric columns only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq


REPO_DEFAULT = Path("data/real_robot/task123_pair")
OUTPUT_DEFAULT = Path("assets/pi05_realman_task123_b6b/anonymous/realman_task123_corl")


def _fixed_list_column_to_numpy(table, name: str) -> np.ndarray:
    values = table[name].combine_chunks().values
    width = table.schema.field(name).type.list_size
    return np.asarray(values).reshape(table.num_rows, width).astype(np.float32)


def _parse_episode_spec(spec: str | None) -> list[int] | None:
    if spec is None or not spec.strip():
        return None
    episodes: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_s, stop_s = item.split(":", 1)
            episodes.extend(range(int(start_s), int(stop_s)))
        else:
            episodes.append(int(item))
    return episodes


def _all_episodes(repo: Path) -> list[int]:
    path = repo / "meta" / "episodes.jsonl"
    episodes: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            import json

            episodes.append(int(json.loads(line)["episode_index"]))
    return episodes


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
    print(f"repo: {repo}")
    print(f"episodes: {len(episodes)}")
    print(f"episode range: {episodes[0]}..{episodes[-1]}" if episodes else "episode range: empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode ids and/or half-open ranges, e.g. '0:128,144:210,218:304'.",
    )
    parser.add_argument("--action-horizon", type=int, default=10)
    args = parser.parse_args()

    episodes = _parse_episode_spec(args.episodes)
    if episodes is None:
        episodes = _all_episodes(args.repo)
    compute_stats(
        args.repo,
        args.output,
        episodes=episodes,
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
