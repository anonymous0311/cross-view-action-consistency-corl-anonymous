#!/usr/bin/env python3
"""Export Task1_CoRL into the pair layout used by cross-view training.

The source dataset has one episode with three synchronized scene-camera videos:

  observation.images.nominal_image
  observation.images.purturbated_c1_image
  observation.images.purturbated_c2_image

The cross-view pair dataloader expects one nominal/perturbed pair per row:

  observation.images.front
  observation.images.perturbed

For LeRobot video datasets, video frames are episode-indexed rather than stored
inside parquet rows. This exporter therefore creates two derived episodes per
source episode: one C0->C1 pair episode and one C0->C2 pair episode. Video files
are symlinked by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

SRC_DEFAULT = Path("data/real_robot/task1")
DST_DEFAULT = Path("data/real_robot/task1_pair")

CAMERA_PAIRS = (
    ("c1", "observation.images.purturbated_c1_image"),
    ("c2", "observation.images.purturbated_c2_image"),
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _replace_int64_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"Column {name!r} not found in parquet table.")
    return table.set_column(idx, name, pa.array(values, type=pa.int64()))


def _symlink_or_copy(src: Path, dst: Path, *, copy_videos: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_videos:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _image_feature_like(src_feature: dict) -> dict:
    feature = dict(src_feature)
    feature["names"] = ["height", "width", "channels"]
    if "info" in feature:
        feature["info"] = dict(feature["info"])
    return feature


def export_pair_dataset(src: Path, dst: Path, *, overwrite: bool, copy_videos: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "videos" / "chunk-000" / "observation.images.front").mkdir(parents=True)
    (dst / "videos" / "chunk-000" / "observation.images.perturbed").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = _read_jsonl(src / "meta" / "episodes.jsonl")
    stats_rows = _read_jsonl(src / "meta" / "episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row["stats"] for row in stats_rows}

    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    global_index = 0

    for ep in episodes:
        src_ep = int(ep["episode_index"])
        src_parquet = src / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
        table = pq.read_table(src_parquet)
        length = int(table.num_rows)

        for pair_offset, (_pair_name, perturbed_key) in enumerate(CAMERA_PAIRS):
            new_ep = src_ep * len(CAMERA_PAIRS) + pair_offset
            new_table = table
            new_table = _replace_int64_column(new_table, "episode_index", [new_ep] * length)
            new_table = _replace_int64_column(new_table, "index", list(range(global_index, global_index + length)))

            dst_parquet = dst / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet"
            pq.write_table(new_table, dst_parquet)

            src_nominal_video = (
                src
                / "videos"
                / "chunk-000"
                / "observation.images.nominal_image"
                / f"episode_{src_ep:06d}.mp4"
            )
            src_perturbed_video = src / "videos" / "chunk-000" / perturbed_key / f"episode_{src_ep:06d}.mp4"
            _symlink_or_copy(
                src_nominal_video,
                dst / "videos" / "chunk-000" / "observation.images.front" / f"episode_{new_ep:06d}.mp4",
                copy_videos=copy_videos,
            )
            _symlink_or_copy(
                src_perturbed_video,
                dst / "videos" / "chunk-000" / "observation.images.perturbed" / f"episode_{new_ep:06d}.mp4",
                copy_videos=copy_videos,
            )

            new_episodes.append({"episode_index": new_ep, "tasks": ep["tasks"], "length": length})

            src_stats = stats_by_episode[src_ep]
            dst_stats = {
                key: value
                for key, value in src_stats.items()
                if key
                not in {
                    "observation.images.nominal_image",
                    "observation.images.purturbated_c1_image",
                    "observation.images.purturbated_c2_image",
                }
            }
            dst_stats["episode_index"] = {
                "min": [new_ep],
                "max": [new_ep],
                "mean": [float(new_ep)],
                "std": [0.0],
                "count": [length],
            }
            dst_stats["index"] = {
                "min": [global_index],
                "max": [global_index + length - 1],
                "mean": [(2 * global_index + length - 1) / 2.0],
                "std": src_stats["frame_index"]["std"],
                "count": [length],
            }
            dst_stats["observation.images.front"] = src_stats["observation.images.nominal_image"]
            dst_stats["observation.images.perturbed"] = src_stats[perturbed_key]
            new_stats.append({"episode_index": new_ep, "stats": dst_stats})

            global_index += length

    tasks_text = (src / "meta" / "tasks.jsonl").read_text()
    (dst / "meta" / "tasks.jsonl").write_text(tasks_text)
    _write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(dst / "meta" / "episodes_stats.jsonl", new_stats)

    front_feature = _image_feature_like(info["features"]["observation.images.nominal_image"])
    perturbed_feature = _image_feature_like(info["features"]["observation.images.purturbated_c1_image"])
    new_features = {
        "action": info["features"]["action"],
        "observation.state": info["features"]["observation.state"],
        "observation.images.front": front_feature,
        "observation.images.perturbed": perturbed_feature,
        "timestamp": info["features"]["timestamp"],
        "frame_index": info["features"]["frame_index"],
        "episode_index": info["features"]["episode_index"],
        "index": info["features"]["index"],
        "task_index": info["features"]["task_index"],
    }
    new_info = dict(info)
    new_info.update(
        {
            "total_episodes": len(new_episodes),
            "total_frames": global_index,
            "total_videos": len(new_episodes) * 2,
            "total_tasks": info["total_tasks"],
            "features": new_features,
        }
    )
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2) + "\n")

    print(f"Wrote pair dataset: {dst}")
    print(f"episodes: {len(new_episodes)}")
    print(f"frames/pairs: {global_index}")
    print(f"videos: {len(new_episodes) * 2} ({'copied' if copy_videos else 'symlinked'})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC_DEFAULT)
    parser.add_argument("--dst", type=Path, default=DST_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking them.")
    args = parser.parse_args()

    export_pair_dataset(args.src.resolve(), args.dst.resolve(), overwrite=args.overwrite, copy_videos=args.copy_videos)


if __name__ == "__main__":
    main()
