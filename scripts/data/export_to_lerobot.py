"""Export exact paired observations and source-HDF5 action chunks to LeRobot v2."""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import datasets
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def action_chunk(actions, timestep, horizon=10):
    if not 0 <= timestep < len(actions):
        raise ValueError("Source timestep outside the demonstration")
    indices = np.minimum(np.arange(timestep, timestep + horizon), len(actions) - 1)
    return np.asarray(actions[indices], dtype=np.float32)


def policy_state(obs, t):
    return np.concatenate([obs["ee_pos"][t], obs["ee_ori"][t], obs["gripper_states"][t]]).astype(np.float32)


def features():
    return datasets.Features(
        {
            "observation.images.front": datasets.Image(),
            "observation.images.perturbed": datasets.Image(),
            "observation.state": datasets.Sequence(datasets.Value("float32"), length=8),
            "action": datasets.Sequence(datasets.Sequence(datasets.Value("float32"), length=7), length=10),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
            "source_timestep": datasets.Value("int64"),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/libero_pairs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.manifest.read_text().splitlines():
        r = json.loads(line)
        groups[(r["suite"], r["task_name"], r["hdf5_path"], r["demo_key"])].append(r)
    if not groups:
        raise ValueError("Manifest is empty")
    meta = args.output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    spec = {"manifest": str(args.manifest.resolve()), "episodes": len(groups), "horizon": 10}
    spec_path = meta / "export_config.json"
    if spec_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new output directory")
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("Resume specification differs")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    tasks = sorted({rows[0]["language"] for rows in groups.values()})
    episodes = []
    mapping = []
    offset = 0
    for episode, (key, rows) in enumerate(tqdm(sorted(groups.items()), desc="Export episodes", dynamic_ncols=True)):
        rows = sorted(rows, key=lambda r: r["timestep"])
        if len({r["timestep"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source timesteps in one episode")
        path = args.output_dir / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        task_index = tasks.index(rows[0]["language"])
        if not (args.resume and path.exists()):
            data = {k: [] for k in features()}
            with h5py.File(key[2]) as f:
                demo = f["data"][key[3]]
                actions = demo["actions"][:]
                for frame, r in enumerate(rows):
                    for column, source_key in [
                        ("observation.images.front", "img_a_path"),
                        ("observation.images.perturbed", "img_b_path"),
                    ]:
                        image = Path(r[source_key]).read_bytes()
                        with Image.open(io.BytesIO(image)) as im:
                            if im.size != (256, 256):
                                raise ValueError("Expected 256×256 upright rendered images")
                        data[column].append({"bytes": image, "path": None})
                    data["observation.state"].append(policy_state(demo["obs"], r["timestep"]))
                    data["action"].append(action_chunk(actions, r["timestep"]))
                    for column, value in [
                        ("timestamp", frame / 20),
                        ("frame_index", frame),
                        ("episode_index", episode),
                        ("index", offset + frame),
                        ("task_index", task_index),
                        ("source_timestep", r["timestep"]),
                    ]:
                        data[column].append(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            datasets.Dataset.from_dict(data, features=features()).to_parquet(str(tmp))
            tmp.replace(path)
        else:
            import pyarrow.parquet as pq

            if pq.read_metadata(path).num_rows != len(rows):
                raise ValueError("Incomplete saved episode")
        episodes.append({"episode_index": episode, "tasks": [tasks[task_index]], "length": len(rows)})
        mapping.append(
            {
                "episode_index": episode,
                "suite": key[0],
                "task_name": key[1],
                "demo_key": key[3],
                "task_index": task_index,
                "camera_params": rows[0]["camera_params_b"],
            }
        )
        offset += len(rows)
    spec_features = {k: {"dtype": "int64", "shape": [1], "names": None} for k in features()}
    for k in ["observation.images.front", "observation.images.perturbed"]:
        spec_features[k] = {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]}
    spec_features["observation.state"] = {"dtype": "float32", "shape": [8], "names": None}
    spec_features["action"] = {"dtype": "float32", "shape": [10, 7], "names": None}
    spec_features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episodes) + 999) // 1000,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": spec_features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name, records in [
        ("episodes", episodes),
        ("tasks", [{"task_index": i, "task": t} for i, t in enumerate(tasks)]),
        ("source_episodes", mapping),
    ]:
        (meta / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"Exported {len(episodes)} episodes, {offset} pairs, {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
