"""LIBERO pair-data.5 — Export rendered LIBERO multiview pairs to LeRobot-compatible format.

Reads the merged pair manifests from LIBERO pair-data.3 and builds a LeRobot dataset
where each episode = one HDF5 demo, each frame has both a nominal and a
perturbed camera image.

Output directory layout (LeRobot v2.0 format):
    data/rendered_libero_multiview_lerobot/
    ├── data/
    │   └── chunk-{NNN}/
    │       └── episode_{NNNNNN}.parquet
    └── meta/
        ├── info.json
        ├── tasks.jsonl
        └── episodes.jsonl

Features stored per frame:
    observation.images.front:     struct<bytes:binary, path:string>  nominal view
    observation.images.perturbed: struct<bytes:binary, path:string>  C1/C2/C3 view
    observation.state:            fixed_size_list<float32>[8]        ee_pos + ee_ori + gripper_states
    action:                       fixed_size_list<float32>[7]        7D action
    timestamp:                    float32                            frame_idx / fps
    frame_index:                  int64                              position in episode
    episode_index:                int64                              global episode id
    index:                        int64                              global frame id
    task_index:                   int64                              task vocabulary index

Resumable: completed episode parquets are skipped on rerun (file-existence check).
Use --overwrite to start fresh.

Usage:
    venv-libero-plus/bin/python scripts/libero_pair_data/export_to_lerobot.py \\
        --manifest-train results/libero_pair_audit/libero_pair_manifest_train.jsonl \\
        --manifest-val   results/libero_pair_audit/libero_pair_manifest_val.jsonl \\
        --hdf5-root      data/libero_hdf5_original \\
        --repo-root      . \\
        --output-dir     data/rendered_libero_multiview_lerobot

After export, verify with:
    venv-libero-plus/bin/python scripts/libero_pair_data/export_to_lerobot.py --verify-only \\
        --output-dir data/rendered_libero_multiview_lerobot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]

FPS = 10
CHUNKS_SIZE = 1000

# PyArrow types matching the LeRobot v2.0 parquet schema
_IMG_STRUCT = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
_STATE_TYPE = pa.list_(pa.float32(), 8)
_ACTION_TYPE = pa.list_(pa.float32(), 7)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("observation.images.front",     _IMG_STRUCT),
        pa.field("observation.images.perturbed", _IMG_STRUCT),
        pa.field("observation.state",            _STATE_TYPE),
        pa.field("action",                       _ACTION_TYPE),
        pa.field("timestamp",                    pa.float32()),
        pa.field("frame_index",                  pa.int64()),
        pa.field("episode_index",                pa.int64()),
        pa.field("index",                        pa.int64()),
        pa.field("task_index",                   pa.int64()),
    ]
)

# HuggingFace datasets metadata embedded in each parquet file.
# This tells hf.datasets to decode image structs as PIL images.
_HF_FEATURES = {
    "observation.images.front":     {"_type": "Image"},
    "observation.images.perturbed": {"_type": "Image"},
    "observation.state": {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 8, "_type": "Sequence",
    },
    "action": {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 7, "_type": "Sequence",
    },
    "timestamp":      {"dtype": "float32", "_type": "Value"},
    "frame_index":    {"dtype": "int64",   "_type": "Value"},
    "episode_index":  {"dtype": "int64",   "_type": "Value"},
    "index":          {"dtype": "int64",   "_type": "Value"},
    "task_index":     {"dtype": "int64",   "_type": "Value"},
}

SCHEMA_WITH_META = PARQUET_SCHEMA.with_metadata(
    {b"huggingface": json.dumps({"info": {"features": _HF_FEATURES}}).encode()}
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _policy_state_from_obs(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_states: np.ndarray,
    t: int,
) -> np.ndarray:
    """Return the 8D state used by OpenPI LIBERO rollout inference.

    This must match canonical.eval.libero_policy_runner.build_policy_observation:
    end-effector position, end-effector axis-angle orientation, then gripper qpos.
    Raw HDF5 robot_states are ordered as gripper, ee_pos, ee_quat and are not a
    valid drop-in replacement for this policy state.
    """
    state = np.concatenate(
        [
            np.asarray(ee_pos[t], dtype=np.float32),
            np.asarray(ee_ori[t], dtype=np.float32),
            np.asarray(gripper_states[t], dtype=np.float32),
        ],
        axis=0,
    )
    if state.shape != (8,):
        raise ValueError(f"Expected 8D policy state at timestep {t}, got shape {state.shape}")
    return state


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_demo(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group manifest rows by (suite, task_name, demo_id), return sorted dict."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["suite"], row["task_name"], row["demo_id"])
        groups[key].append(row)
    return dict(sorted(groups.items()))


def _hdf5_path(hdf5_root: Path, suite: str, task_name: str) -> Path:
    return hdf5_root / suite / f"{task_name}_demo.hdf5"


def _write_episode_parquet(
    out_path: Path,
    rows: list[dict],
    repo_root: Path,
    hdf5_root: Path,
    episode_index: int,
    global_frame_offset: int,
    task_index: int,
) -> int:
    """Write one episode parquet. Returns number of frames written."""
    rows_sorted = sorted(rows, key=lambda r: r["timestep"])
    suite     = rows_sorted[0]["suite"]
    task_name = rows_sorted[0]["task_name"]
    demo_id   = rows_sorted[0]["demo_id"]

    hdf5_p = _hdf5_path(hdf5_root, suite, task_name)
    with h5py.File(hdf5_p, "r") as f:
        demo_grp   = f[f"data/demo_{demo_id}"]
        obs_grp = demo_grp["obs"]
        ee_pos = obs_grp["ee_pos"][:]                    # (T, 3)
        ee_ori = obs_grp["ee_ori"][:]                    # (T, 3), axis-angle
        gripper_states = obs_grp["gripper_states"][:]    # (T, 2)
        actions = demo_grp["actions"][:]                 # (T, 7)

    imgs_front     = []
    imgs_perturbed = []
    states         = []
    acts           = []
    timestamps     = []
    frame_indices  = []
    episode_idxs   = []
    global_indices = []
    task_indices   = []

    for fi, row in enumerate(rows_sorted):
        t = row["timestep"]

        img_a_abs = repo_root / row["img_a_path"]
        img_b_abs = repo_root / row["img_b_path"]
        front_bytes     = img_a_abs.read_bytes()
        perturbed_bytes = img_b_abs.read_bytes()

        imgs_front.append({"bytes": front_bytes, "path": None})
        imgs_perturbed.append({"bytes": perturbed_bytes, "path": None})
        states.append(_policy_state_from_obs(ee_pos, ee_ori, gripper_states, t).tolist())
        acts.append(actions[t].astype(np.float32).tolist())
        timestamps.append(np.float32(fi / FPS))
        frame_indices.append(fi)
        episode_idxs.append(episode_index)
        global_indices.append(global_frame_offset + fi)
        task_indices.append(task_index)

    n_frames = fi + 1
    table = pa.table(
        {
            "observation.images.front":     pa.array(imgs_front,     type=_IMG_STRUCT),
            "observation.images.perturbed": pa.array(imgs_perturbed, type=_IMG_STRUCT),
            "observation.state":            pa.array(states,          type=_STATE_TYPE),
            "action":                       pa.array(acts,            type=_ACTION_TYPE),
            "timestamp":                    pa.array(timestamps,      type=pa.float32()),
            "frame_index":                  pa.array(frame_indices,   type=pa.int64()),
            "episode_index":                pa.array(episode_idxs,    type=pa.int64()),
            "index":                        pa.array(global_indices,  type=pa.int64()),
            "task_index":                   pa.array(task_indices,    type=pa.int64()),
        },
        schema=SCHEMA_WITH_META,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return n_frames


def _episode_parquet_path(output_dir: Path, episode_index: int) -> Path:
    chunk = episode_index // CHUNKS_SIZE
    return output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


# ── Meta writers ───────────────────────────────────────────────────────────────

def _write_info(output_dir: Path, total_episodes: int, total_frames: int,
                total_tasks: int, n_train: int) -> None:
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {
            "train": f"0:{n_train}",
            "val":   f"{n_train}:{total_episodes}",
        },
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "image", "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.perturbed": {
                "dtype": "image", "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "eef_x",
                    "eef_y",
                    "eef_z",
                    "axisangle_x",
                    "axisangle_y",
                    "axisangle_z",
                    "gripper_0",
                    "gripper_1",
                ],
            },
            "action": {
                "dtype": "float32", "shape": [7], "names": ["action"],
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def _write_tasks(output_dir: Path, task_vocab: dict[str, int]) -> None:
    lines = [
        json.dumps({"task_index": idx, "task": task}) + "\n"
        for task, idx in sorted(task_vocab.items(), key=lambda x: x[1])
    ]
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        f.writelines(lines)


def _write_episodes(output_dir: Path, episode_meta: list[dict]) -> None:
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep) + "\n")


# ── Verify ─────────────────────────────────────────────────────────────────────

def verify(output_dir: Path) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not info_path.exists():
        print("ERROR: meta/info.json not found")
        sys.exit(1)
    with open(info_path) as f:
        info = json.load(f)

    ep_count = info["total_episodes"]
    frame_count = info["total_frames"]
    print(f"info.json: {ep_count} episodes, {frame_count} frames")
    print(f"  splits: {info['splits']}")

    # Check a sample parquet
    sample_path = _episode_parquet_path(output_dir, 0)
    if not sample_path.exists():
        print(f"ERROR: episode_000000.parquet not found at {sample_path}")
        sys.exit(1)

    table = pq.read_table(sample_path)
    print(f"\nSample parquet (episode 0):")
    print(f"  rows:    {len(table)}")
    print(f"  columns: {table.column_names}")
    print(f"  schema:  {table.schema}")

    # Decode first image to verify it's valid PNG
    front_bytes = table["observation.images.front"][0].as_py()["bytes"]
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(front_bytes))
    print(f"\n  observation.images.front[0]: {img.size} {img.mode}")

    pert_bytes = table["observation.images.perturbed"][0].as_py()["bytes"]
    img2 = Image.open(io.BytesIO(pert_bytes))
    print(f"  observation.images.perturbed[0]: {img2.size} {img2.mode}")

    state_val = table["observation.state"][0].as_py()
    print(f"  observation.state[0]: shape={len(state_val)}")
    action_val = table["action"][0].as_py()
    print(f"  action[0]: shape={len(action_val)}")

    print("\nVERIFY OK")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-train", type=Path,
                        default=REPO_ROOT / "results/libero_pair_audit/libero_pair_manifest_train.jsonl")
    parser.add_argument("--manifest-val",   type=Path,
                        default=REPO_ROOT / "results/libero_pair_audit/libero_pair_manifest_val.jsonl")
    parser.add_argument("--hdf5-root",      type=Path,
                        default=Path("data/libero_hdf5_original"))
    parser.add_argument("--repo-root",      type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir",     type=Path,
                        default=Path("data/libero_multiview_lerobot"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete output-dir and start fresh.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip export; just verify an existing dataset.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo_root / output_dir

    if args.verify_only:
        verify(output_dir)
        return

    if args.overwrite and output_dir.exists():
        print(f"--overwrite: deleting {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and group manifests ───────────────────────────────────────────────
    print("Loading manifests ...")
    train_rows = _load_manifest(args.manifest_train)
    val_rows   = _load_manifest(args.manifest_val)
    print(f"  train: {len(train_rows):,} pairs | val: {len(val_rows):,} pairs")

    train_groups = _group_by_demo(train_rows)
    val_groups   = _group_by_demo(val_rows)
    n_train_eps  = len(train_groups)
    n_val_eps    = len(val_groups)
    print(f"  train: {n_train_eps:,} episodes | val: {n_val_eps:,} episodes")

    # ── Build task vocabulary ──────────────────────────────────────────────────
    task_vocab: dict[str, int] = {}
    for rows in list(train_groups.values()) + list(val_groups.values()):
        lang = rows[0]["language"]
        if lang not in task_vocab:
            task_vocab[lang] = len(task_vocab)
    print(f"  unique tasks: {len(task_vocab)}")

    # ── Export episodes ────────────────────────────────────────────────────────
    episode_meta: list[dict] = []
    global_frame_offset = 0
    total_skipped = 0

    all_groups = list(train_groups.items()) + list(val_groups.items())

    with tqdm(total=len(all_groups), desc="episodes", unit="ep",
              dynamic_ncols=True, leave=True) as pbar:
        for ep_idx, (key, rows) in enumerate(all_groups):
            out_path = _episode_parquet_path(output_dir, ep_idx)

            if out_path.exists():
                # Resume: count frames from existing parquet
                existing = pq.read_metadata(out_path)
                n_frames = existing.num_rows
                global_frame_offset += n_frames
                episode_meta.append({
                    "episode_index": ep_idx,
                    "tasks": [rows[0]["language"]],
                    "length": n_frames,
                })
                total_skipped += 1
                pbar.set_postfix(skipped=total_skipped, frames=global_frame_offset)
                pbar.update(1)
                continue

            task_index = task_vocab[rows[0]["language"]]
            try:
                n_frames = _write_episode_parquet(
                    out_path=out_path,
                    rows=rows,
                    repo_root=args.repo_root,
                    hdf5_root=args.hdf5_root,
                    episode_index=ep_idx,
                    global_frame_offset=global_frame_offset,
                    task_index=task_index,
                )
            except Exception as exc:
                tqdm.write(f"[ERROR] episode {ep_idx} ({key}): {exc}")
                raise

            episode_meta.append({
                "episode_index": ep_idx,
                "tasks": [rows[0]["language"]],
                "length": n_frames,
            })
            global_frame_offset += n_frames
            pbar.set_postfix(ep=ep_idx, frames=global_frame_offset)
            pbar.update(1)

    total_frames = global_frame_offset
    total_eps    = len(all_groups)

    # ── Write meta ─────────────────────────────────────────────────────────────
    print("Writing meta/ ...")
    _write_info(output_dir, total_eps, total_frames, len(task_vocab), n_train_eps)
    _write_tasks(output_dir, task_vocab)
    _write_episodes(output_dir, episode_meta)

    print(f"\nDone.")
    print(f"  Episodes : {total_eps:,}  (train: {n_train_eps:,} | val: {n_val_eps:,})")
    print(f"  Frames   : {total_frames:,}")
    print(f"  Tasks    : {len(task_vocab):,}")
    print(f"  Output   : {output_dir}")
    if total_skipped:
        print(f"  Skipped  : {total_skipped:,} (already done)")
    print(f"\nNext: verify with --verify-only, then write LeRobotCrossViewPairDataConfig in config.py")


if __name__ == "__main__":
    main()
