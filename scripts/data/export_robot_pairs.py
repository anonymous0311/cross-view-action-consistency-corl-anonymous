"""Export C1-anchored timestamp-aligned robot demonstrations to paired LeRobot data.

Input: one NPZ per demonstration; see docs/real_robot.md for the array contract.
"""

import argparse
import io
import json
from pathlib import Path

import datasets
import numpy as np
from export_to_lerobot import action_chunk, features
from PIL import Image
from tqdm import tqdm

from afcv.robot_data import nearest_timestamps


def encode_image(array):
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Camera frames must be upright uint8 RGB")
    image = Image.fromarray(array)
    image.thumbnail((256, 256))
    canvas = Image.new("RGB", (256, 256))
    canvas.paste(image, ((256 - image.width) // 2, (256 - image.height) // 2))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    paths = sorted(args.input_dir.glob("*.npz"))
    if not paths:
        raise ValueError("No demonstration NPZ files found")
    meta = args.output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    spec = {"files": [str(p.resolve()) for p in paths], "seed": args.seed, "max_offset_seconds": 0.067}
    spec_path = meta / "export_config.json"
    if spec_path.exists() and (not args.resume or json.loads(spec_path.read_text()) != spec):
        raise ValueError("Use --resume with unchanged inputs or a new output directory")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    episodes, tasks, offset = [], [], 0
    for episode, path in enumerate(tqdm(paths, desc="Robot episodes", dynamic_ncols=True)):
        with np.load(path, allow_pickle=False) as demo:
            times, actions, state = demo["timestamps_C1"], demo["actions"], demo["state"]
            if actions.shape != (len(times), 7) or state.shape != (len(times), 8):
                raise ValueError("Expected base-frame actions [N,7] and proprioception [N,8] on C1 timestamps")
            if not np.isfinite(actions).all() or not np.isfinite(state).all():
                raise ValueError("Nonfinite state/action values")
            for camera in ["C0", "C1", "C2"]:
                if len(demo[f"images_{camera}"]) != len(demo[f"timestamps_{camera}"]):
                    raise ValueError("Image/timestamp counts differ")
            task = str(demo["instruction"].item())
            if task not in tasks:
                tasks.append(task)
            matches = [nearest_timestamps(times, demo[f"timestamps_{camera}"]) for camera in ["C0", "C2"]]
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode]))
            output = {k: [] for k in features()}
            for t in range(len(times)):
                valid = [i for i, (_, mask) in enumerate(matches) if mask[t]]
                if not valid:
                    continue
                partner = int(rng.choice(valid))
                camera = ["C0", "C2"][partner]
                frame = len(output["index"])
                row = {
                    "observation.images.front": encode_image(demo["images_C1"][t]),
                    "observation.images.perturbed": encode_image(demo[f"images_{camera}"][matches[partner][0][t]]),
                    "observation.state": state[t],
                    "action": action_chunk(actions, t),
                    "timestamp": float(times[t] - times[0]),
                    "frame_index": frame,
                    "episode_index": episode,
                    "index": offset + frame,
                    "task_index": tasks.index(task),
                    "source_timestep": t,
                }
                for key, value in row.items():
                    output[key].append(value)
            length = len(output["index"])
            if not length:
                raise ValueError(f"No valid camera pairs in {path.name}")
            dest = args.output_dir / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            if not (args.resume and dest.exists()):
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix(".tmp")
                datasets.Dataset.from_dict(output, features=features()).to_parquet(str(temporary))
                temporary.replace(dest)
            episodes.append({"episode_index": episode, "tasks": [task], "length": length})
            offset += length
    spec_features = {k: {"dtype": "int64", "shape": [1], "names": None} for k in features()}
    for key in ["observation.images.front", "observation.images.perturbed"]:
        spec_features[key] = {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]}
    spec_features["observation.state"] = {"dtype": "float32", "shape": [8], "names": None}
    spec_features["action"] = {"dtype": "float32", "shape": [10, 7], "names": None}
    spec_features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.0",
        "robot_type": "rm75",
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episodes) + 999) // 1000,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": f"0:{len(episodes)}"},
        "video_path": None,
        "features": spec_features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name, rows in [("episodes", episodes), ("tasks", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])]:
        (meta / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Exported {len(episodes)} demonstrations, {offset} aligned pairs")


if __name__ == "__main__":
    main()
