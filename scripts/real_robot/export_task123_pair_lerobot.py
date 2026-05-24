#!/usr/bin/env python3
"""Export Task1/2/3 CoRL realman datasets into one cross-view pair LeRobot repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCES = (
    Path("data/real_robot/task1"),
    Path("data/real_robot/task2"),
    Path("data/real_robot/task3"),
)
DEFAULT_DST = Path("data/real_robot/task123_pair")

CAMERA_PAIRS = (
    ("c1", "observation.images.purturbated_c1_image"),
    ("c2", "observation.images.purturbated_c2_image"),
)

IMAGE_KEYS = {
    "observation.images.nominal_image",
    "observation.images.purturbated_c1_image",
    "observation.images.purturbated_c2_image",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _replace_int64_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"Column {name!r} not found in parquet table.")
    return table.set_column(idx, name, pa.array(values, type=pa.int64()))


def _symlink_or_copy(src: Path, dst: Path, *, copy_videos: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source video: {src}")
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


def _scalar_stats(value: int, count: int) -> dict:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": [count],
    }


def _index_stats(start: int, length: int, frame_std: list[float]) -> dict:
    return {
        "min": [start],
        "max": [start + length - 1],
        "mean": [(2 * start + length - 1) / 2.0],
        "std": frame_std,
        "count": [length],
    }


def _validate_sources(sources: tuple[Path, ...]) -> None:
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Source dataset does not exist: {source}")
        for rel in ("meta/info.json", "meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/tasks.jsonl"):
            if not (source / rel).exists():
                raise FileNotFoundError(f"Source dataset is missing {rel}: {source}")


def export_pair_dataset(
    sources: tuple[Path, ...],
    dst: Path,
    *,
    overwrite: bool,
    copy_videos: bool,
) -> None:
    _validate_sources(sources)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "videos" / "chunk-000" / "observation.images.front").mkdir(parents=True)
    (dst / "videos" / "chunk-000" / "observation.images.perturbed").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    first_info = json.loads((sources[0] / "meta" / "info.json").read_text(encoding="utf-8"))
    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    new_tasks: list[dict] = []
    global_episode = 0
    global_index = 0

    for global_task_index, source in enumerate(sources):
        info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        if set(info["features"]).issuperset(IMAGE_KEYS) is False:
            raise ValueError(f"{source}: expected camera keys {sorted(IMAGE_KEYS)}, got {sorted(info['features'])}")
        episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
        stats_rows = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
        stats_by_episode = {int(row["episode_index"]): row["stats"] for row in stats_rows}
        tasks = _read_jsonl(source / "meta" / "tasks.jsonl")
        if len(tasks) != 1:
            raise ValueError(f"{source}: expected exactly one task, got {len(tasks)}.")
        task_text = tasks[0]["task"]
        new_tasks.append({"task_index": global_task_index, "task": task_text})

        for ep in episodes:
            src_ep = int(ep["episode_index"])
            src_parquet = source / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"
            table = pq.read_table(src_parquet)
            length = int(table.num_rows)

            for _pair_name, perturbed_key in CAMERA_PAIRS:
                new_ep = global_episode
                new_table = table
                new_table = _replace_int64_column(new_table, "episode_index", [new_ep] * length)
                new_table = _replace_int64_column(new_table, "index", list(range(global_index, global_index + length)))
                new_table = _replace_int64_column(new_table, "task_index", [global_task_index] * length)

                dst_parquet = dst / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet"
                pq.write_table(new_table, dst_parquet)

                video_root = source / "videos" / "chunk-000"
                src_nominal = video_root / "observation.images.nominal_image" / f"episode_{src_ep:06d}.mp4"
                src_perturbed = video_root / perturbed_key / f"episode_{src_ep:06d}.mp4"
                _symlink_or_copy(
                    src_nominal,
                    dst / "videos" / "chunk-000" / "observation.images.front" / f"episode_{new_ep:06d}.mp4",
                    copy_videos=copy_videos,
                )
                _symlink_or_copy(
                    src_perturbed,
                    dst / "videos" / "chunk-000" / "observation.images.perturbed" / f"episode_{new_ep:06d}.mp4",
                    copy_videos=copy_videos,
                )

                new_episodes.append({"episode_index": new_ep, "tasks": [task_text], "length": length})

                src_stats = stats_by_episode[src_ep]
                dst_stats = {key: value for key, value in src_stats.items() if key not in IMAGE_KEYS}
                dst_stats["episode_index"] = _scalar_stats(new_ep, length)
                dst_stats["task_index"] = _scalar_stats(global_task_index, length)
                dst_stats["index"] = _index_stats(global_index, length, src_stats["frame_index"]["std"])
                dst_stats["observation.images.front"] = src_stats["observation.images.nominal_image"]
                dst_stats["observation.images.perturbed"] = src_stats[perturbed_key]
                new_stats.append({"episode_index": new_ep, "stats": dst_stats})

                global_episode += 1
                global_index += length

    front_feature = _image_feature_like(first_info["features"]["observation.images.nominal_image"])
    perturbed_feature = _image_feature_like(first_info["features"]["observation.images.purturbated_c1_image"])
    new_features = {
        "action": first_info["features"]["action"],
        "observation.state": first_info["features"]["observation.state"],
        "observation.images.front": front_feature,
        "observation.images.perturbed": perturbed_feature,
        "timestamp": first_info["features"]["timestamp"],
        "frame_index": first_info["features"]["frame_index"],
        "episode_index": first_info["features"]["episode_index"],
        "index": first_info["features"]["index"],
        "task_index": first_info["features"]["task_index"],
    }
    new_info = dict(first_info)
    new_info.update(
        {
            "total_episodes": len(new_episodes),
            "total_frames": global_index,
            "total_tasks": len(new_tasks),
            "total_videos": len(new_episodes) * 2,
            "splits": {"train": f"0:{len(new_episodes)}"},
            "features": new_features,
        }
    )

    _write_jsonl(dst / "meta" / "tasks.jsonl", new_tasks)
    _write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(dst / "meta" / "episodes_stats.jsonl", new_stats)
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote combined pair dataset: {dst}")
    print(f"sources: {len(sources)}")
    print(f"tasks: {len(new_tasks)}")
    print(f"episodes: {len(new_episodes)}")
    print(f"frames/pairs: {global_index}")
    print(f"videos: {len(new_episodes) * 2} ({'copied' if copy_videos else 'symlinked'})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking them.")
    args = parser.parse_args()

    export_pair_dataset(
        tuple(source.resolve() for source in args.sources),
        args.dst.resolve(),
        overwrite=args.overwrite,
        copy_videos=args.copy_videos,
    )


if __name__ == "__main__":
    main()
