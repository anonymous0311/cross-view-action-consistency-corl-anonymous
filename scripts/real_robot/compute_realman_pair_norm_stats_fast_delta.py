#!/usr/bin/env python3
"""Compute Realman pair norm stats in DELTA-action space.

For each action chunk row ``actions[t : t+H]`` we subtract the current state
``state[t]`` from joints 0..6 only; the gripper dimension (index 7) is kept
absolute. This mirrors what
``openpi.transforms.DeltaActions(make_bool_mask(7, -1))`` does at runtime, so
training-time normalization sees the same distribution the live data pipeline
will produce.

Pair episodes 0:128, 144:210, 218:304 are the default train split (matches
``_REALMAN_TASK123_TRAIN_EPISODES`` in ``config.py``). Override with
``--episodes`` if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq

REPO_DEFAULT = Path("data/real_robot/task123_pair")
OUTPUT_DEFAULT = Path("assets/pi05_realman_task123_b6b_delta/anonymous/realman_task123_corl_delta")
TRAIN_SPLIT_DEFAULT = "0:128,144:210,218:304"
# Joints get delta; final dim (gripper) stays absolute. Length 8.
DELTA_MASK_DEFAULT = (True, True, True, True, True, True, True, False)


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
            episodes.append(int(json.loads(line)["episode_index"]))
    return episodes


def compute_stats(
    repo: Path,
    output: Path,
    *,
    episodes: list[int],
    action_horizon: int,
    delta_mask: tuple[bool, ...],
) -> None:
    delta_mask_arr = np.asarray(delta_mask, dtype=bool)
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for ep_idx in episodes:
        table = pq.read_table(repo / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        state = _fixed_list_column_to_numpy(table, "observation.state")          # (T, 8)
        action = _fixed_list_column_to_numpy(table, "action")                    # (T, 8)
        length = action.shape[0]
        chunk_indices = np.minimum(
            np.arange(length)[:, None] + np.arange(action_horizon)[None, :],
            length - 1,
        )
        action_chunks = action[chunk_indices]                                     # (T, H, 8)

        # Subtract state[t] from masked dims only — same semantics as the
        # runtime DeltaActions transform with use_quantiles enabled.
        dim = delta_mask_arr.shape[0]
        anchor = np.where(delta_mask_arr, state[:, :dim], 0.0)                    # (T, 8)
        action_chunks = action_chunks.copy()
        action_chunks[..., :dim] = action_chunks[..., :dim] - anchor[:, None, :]

        stats["state"].update(state)
        stats["actions"].update(action_chunks)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output.mkdir(parents=True, exist_ok=True)
    normalize.save(output, norm_stats)
    print(f"Wrote delta norm stats: {output / 'norm_stats.json'}")
    print(f"repo: {repo}")
    print(f"episodes: {len(episodes)} (range {episodes[0]}..{episodes[-1]})")
    print(f"delta_mask: {delta_mask_arr.tolist()}")
    # Quick sanity print.
    a = norm_stats["actions"]
    print("action.mean:", np.asarray(a.mean).round(4).tolist())
    print("action.std :", np.asarray(a.std).round(4).tolist())
    if a.q01 is not None and a.q99 is not None:
        print("action.q01 :", np.asarray(a.q01).round(4).tolist())
        print("action.q99 :", np.asarray(a.q99).round(4).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--episodes",
        default=TRAIN_SPLIT_DEFAULT,
        help=(
            "Comma-separated episode ids and/or half-open ranges, e.g. "
            "'0:128,144:210,218:304'. Use --episodes='' to default to all "
            "episodes in the repo."
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument(
        "--delta-mask",
        default=",".join("1" if b else "0" for b in DELTA_MASK_DEFAULT),
        help="Comma-separated 0/1 mask of length action_dim (default '1,1,1,1,1,1,1,0').",
    )
    args = parser.parse_args()

    delta_mask = tuple(bool(int(x)) for x in args.delta_mask.split(",") if x.strip())
    if not delta_mask:
        raise ValueError("delta_mask must contain at least one entry.")

    episodes = _parse_episode_spec(args.episodes)
    if episodes is None:
        episodes = _all_episodes(args.repo)
    compute_stats(
        args.repo,
        args.output,
        episodes=episodes,
        action_horizon=args.action_horizon,
        delta_mask=delta_mask,
    )


if __name__ == "__main__":
    main()
