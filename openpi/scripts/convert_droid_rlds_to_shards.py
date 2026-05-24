"""Convert the official DROID RLDS dataset into local HDF5 shards for the modern JAX training stack.

This script is intended to run in the legacy RLDS environment. The resulting shards can then be
consumed by the modern CUDA 13 / JAX training environment via `ShardedDroidDataset`.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Mapping
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import time
import subprocess
import sys
from typing import Literal

import numpy as np
import tqdm
import tyro

import openpi.shared.download as download
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.droid_sharded_dataset as droid_sharded_dataset

STATE_FILENAME = "conversion_state.json"
DEFAULT_FILTER_DICT = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"


@dataclasses.dataclass
class ConversionState:
    processed_success_episodes: int = 0
    total_samples: int = 0
    total_episodes: int = 0
    completed: bool = False


def _validate_legacy_rlds_env() -> None:
    numpy_version = np.lib.NumpyVersion(np.__version__)
    if numpy_version >= np.lib.NumpyVersion("2.0.0"):
        raise RuntimeError(
            "DROID RLDS conversion requires NumPy < 2 because tensorflow-cpu==2.15.0 and dlimp are not "
            f"compatible with NumPy {np.__version__}. "
            "Repair the active env with: `uv sync --active --no-default-groups --group rlds` "
            "and run conversion with: "
            "`uv run --active --no-sync --no-default-groups --group rlds "
            "openpi/scripts/convert_droid_rlds_to_shards.py ...`"
        )


def _load_state(state_path: pathlib.Path, *, allow_partial: bool = False) -> ConversionState:
    if not state_path.exists():
        return ConversionState()
    try:
        with state_path.open() as f:
            return ConversionState(**json.load(f))
    except json.JSONDecodeError:
        if allow_partial:
            return ConversionState()
        raise


def _save_state(state_path: pathlib.Path, state: ConversionState) -> None:
    tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(dataclasses.asdict(state), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(state_path)


def _resolve_cached_filter_path(cached_filter_path: pathlib.Path, original_filter_path: str) -> pathlib.Path:
    if not cached_filter_path.is_dir():
        return cached_filter_path

    # Backwards compatibility: older download logic could cache file URLs as directories.
    candidate = cached_filter_path / pathlib.Path(original_filter_path).name
    if candidate.exists():
        return candidate

    json_candidates = sorted(cached_filter_path.glob("*.json"))
    if len(json_candidates) == 1:
        return json_candidates[0]

    raise FileNotFoundError(
        f"Filter dict cache path {cached_filter_path} is a directory but no unique JSON file could be resolved."
    )


def _normalize_tfds_data_dir(data_dir: str, dataset_name: str, dataset_version: str) -> str:
    path = pathlib.Path(data_dir).resolve()
    if (path / dataset_name / dataset_version).exists():
        return str(path)

    # Common mistake: passing ".../<dataset_name>/<dataset_version>" instead of its parent directory.
    if path.name == dataset_version and path.parent.name == dataset_name:
        normalized = path.parent.parent
        logging.warning(
            "data_dir=%s points to %s/%s. Using parent directory %s for TFDS.",
            path,
            dataset_name,
            dataset_version,
            normalized,
        )
        return str(normalized)

    # Also support passing ".../<dataset_name>".
    if path.name == dataset_name and (path / dataset_version).exists():
        normalized = path.parent
        logging.warning(
            "data_dir=%s points to %s. Using parent directory %s for TFDS.",
            path,
            dataset_name,
            normalized,
        )
        return str(normalized)

    return str(path)


def _decode_image(image_value: object, tf) -> np.ndarray:
    image = np.asarray(image_value)
    if image.dtype == np.uint8 and image.ndim == 3:
        return image
    decoded = tf.io.decode_image(tf.convert_to_tensor(image_value), expand_animations=False, dtype=tf.uint8).numpy()
    if decoded.shape != (180, 320, 3):
        raise ValueError(f"Expected decoded DROID image to have shape (180, 320, 3), got {decoded.shape}")
    return decoded


def _normalize_gripper_position(gripper_position: np.ndarray) -> np.ndarray:
    gripper_position = np.asarray(gripper_position, dtype=np.float32)
    if gripper_position.ndim == 1:
        return gripper_position[:, None]
    return gripper_position


def _first_text_value(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return droid_sharded_dataset.decode_text(array.item())
    return droid_sharded_dataset.decode_text(array.reshape(-1)[0])


def _empty_batch(action_horizon: int) -> dict[str, np.ndarray | list[str]]:
    return {
        "exterior_image_1_left": np.zeros((0, 180, 320, 3), dtype=np.uint8),
        "exterior_image_2_left": np.zeros((0, 180, 320, 3), dtype=np.uint8),
        "wrist_image_left": np.zeros((0, 180, 320, 3), dtype=np.uint8),
        "joint_position": np.zeros((0, 7), dtype=np.float32),
        "gripper_position": np.zeros((0, 1), dtype=np.float32),
        "actions": np.zeros((0, action_horizon, 8), dtype=np.float32),
        "prompt": [],
        "step_id": [],
        "episode_key": [],
        "timestep": np.zeros((0,), dtype=np.int32),
    }


def _validate_worker_config(split_worker_count: int, split_worker_index: int | None, resume: bool) -> None:
    if split_worker_count < 1:
        raise ValueError(f"split_worker_count must be >= 1, got {split_worker_count}")
    if split_worker_index is not None and not (0 <= split_worker_index < split_worker_count):
        raise ValueError(
            f"split_worker_index must be in [0, {split_worker_count}), got {split_worker_index}"
        )
    if split_worker_count > 1 and resume:
        raise ValueError("resume=True is not supported with split_worker_count > 1.")


def _validate_parallel_output_path(output_path: pathlib.Path) -> None:
    existing_shards = sorted(output_path.glob("shard-*.h5"))
    if existing_shards:
        raise FileExistsError(
            f"Found existing shards in {output_path}. Use a fresh output_dir for parallel conversion."
        )
    if (output_path / droid_sharded_dataset.MANIFEST_FILENAME).exists():
        raise FileExistsError(
            f"Found existing manifest at {output_path / droid_sharded_dataset.MANIFEST_FILENAME}. "
            "Use a fresh output_dir for parallel conversion."
        )
    if (output_path / STATE_FILENAME).exists():
        raise FileExistsError(
            f"Found existing conversion state at {output_path / STATE_FILENAME}. "
            "Use a fresh output_dir for parallel conversion."
        )


def _worker_output_dir(output_path: pathlib.Path, worker_index: int) -> pathlib.Path:
    return output_path / "_worker_outputs" / f"worker-{worker_index:03d}"


def _build_worker_command(
    *,
    worker_index: int,
    split_worker_count: int,
    worker_output_dir: pathlib.Path,
    data_dir: str,
    dataset_name: str,
    dataset_version: str,
    filter_dict_path: str | None,
    action_space: str,
    action_horizon: int,
    records_per_shard: int,
    max_episodes: int | None,
    num_parallel_reads: int,
    decode_workers: int,
    tf_intra_op_threads: int | None,
    tf_inter_op_threads: int | None,
    show_progress: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--data-dir",
        data_dir,
        "--output-dir",
        str(worker_output_dir),
        "--dataset-name",
        dataset_name,
        "--dataset-version",
        dataset_version,
        "--action-space",
        action_space,
        "--action-horizon",
        str(action_horizon),
        "--records-per-shard",
        str(records_per_shard),
        "--num-parallel-reads",
        str(num_parallel_reads),
        "--decode-workers",
        str(decode_workers),
        "--split-worker-count",
        str(split_worker_count),
        "--split-worker-index",
        str(worker_index),
    ]
    if filter_dict_path is not None:
        command.extend(["--filter-dict-path", filter_dict_path])
    else:
        command.extend(["--filter-dict-path", "None"])
    if max_episodes is not None:
        command.extend(["--max-episodes", str(max_episodes)])
    if tf_intra_op_threads is not None:
        command.extend(["--tf-intra-op-threads", str(tf_intra_op_threads)])
    if tf_inter_op_threads is not None:
        command.extend(["--tf-inter-op-threads", str(tf_inter_op_threads)])
    if not show_progress:
        command.append("--no-show-progress")
    return command


def _read_rss_bytes(pid: int) -> int | None:
    """Best-effort RSS read from /proc. Returns None if unavailable."""
    status_path = pathlib.Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                # Format: VmRSS:\t  12345 kB
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _format_gb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def _run_parallel_workers(
    *,
    split_worker_count: int,
    output_path: pathlib.Path,
    data_dir: str,
    dataset_name: str,
    dataset_version: str,
    filter_dict_path: str | None,
    action_space: str,
    action_horizon: int,
    records_per_shard: int,
    max_episodes: int | None,
    num_parallel_reads: int,
    decode_workers: int,
    tf_intra_op_threads: int | None,
    tf_inter_op_threads: int | None,
    show_progress: bool,
    max_worker_rss_gb: float | None,
    max_total_rss_gb: float | None,
    max_concurrent_workers: int,
) -> list[pathlib.Path]:
    worker_dirs = [_worker_output_dir(output_path, i) for i in range(split_worker_count)]
    for worker_dir in worker_dirs:
        worker_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_path / "_worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    processes: dict[int, tuple[subprocess.Popen[str], pathlib.Path, object]] = {}
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    failed_workers: list[tuple[int, int, pathlib.Path]] = []
    next_worker_to_launch = 0
    completed_workers = 0

    total = None if max_episodes is None else (max_episodes * split_worker_count)
    overall_progress = (
        tqdm.tqdm(total=total, dynamic_ncols=True, desc="Converting DROID RLDS [total]") if show_progress else None
    )
    worker_rss_cap_bytes = None if max_worker_rss_gb is None else int(max_worker_rss_gb * (1024**3))
    total_rss_cap_bytes = None if max_total_rss_gb is None else int(max_total_rss_gb * (1024**3))
    over_total_cap_warned = False

    def _launch_worker(worker_index: int) -> None:
        worker_dir = worker_dirs[worker_index]
        log_path = logs_dir / f"worker-{worker_index:03d}.log"
        log_file = log_path.open("w")
        command = _build_worker_command(
            worker_index=worker_index,
            split_worker_count=split_worker_count,
            worker_output_dir=worker_dir,
            data_dir=data_dir,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filter_dict_path=filter_dict_path,
            action_space=action_space,
            action_horizon=action_horizon,
            records_per_shard=records_per_shard,
            max_episodes=max_episodes,
            num_parallel_reads=num_parallel_reads,
            decode_workers=decode_workers,
            tf_intra_op_threads=tf_intra_op_threads,
            tf_inter_op_threads=tf_inter_op_threads,
            show_progress=False,
        )
        logging.info(
            "Launching worker %d/%d (active %d/%d, log: %s)",
            worker_index + 1,
            split_worker_count,
            len(processes) + 1,
            max_concurrent_workers,
            log_path,
        )
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes[worker_index] = (process, log_path, log_file)

    try:
        while completed_workers < split_worker_count:
            processed_success = 0
            total_samples = 0
            total_kept_episodes = 0
            live_worker_rss_bytes = 0
            max_live_worker_rss_bytes = 0
            worker_over_cap_count = 0

            for worker_index, (process, _, _) in processes.items():
                return_code = process.poll()
                if return_code is not None:
                    continue
                worker_rss = _read_rss_bytes(process.pid)
                if worker_rss is None:
                    continue
                live_worker_rss_bytes += worker_rss
                max_live_worker_rss_bytes = max(max_live_worker_rss_bytes, worker_rss)
                if worker_rss_cap_bytes is not None and worker_rss > worker_rss_cap_bytes:
                    worker_over_cap_count += 1

            over_total_cap = total_rss_cap_bytes is not None and live_worker_rss_bytes > total_rss_cap_bytes
            if over_total_cap and not over_total_cap_warned:
                logging.warning(
                    "Current live worker RSS %.2f GiB is above --max-total-rss-gb=%.2f GiB. "
                    "Pausing new launches until memory drops.",
                    _format_gb(live_worker_rss_bytes),
                    max_total_rss_gb,
                )
                over_total_cap_warned = True
            if not over_total_cap:
                over_total_cap_warned = False

            while (
                next_worker_to_launch < split_worker_count
                and len(processes) < max_concurrent_workers
                and not over_total_cap
            ):
                _launch_worker(next_worker_to_launch)
                next_worker_to_launch += 1
                # With a total RSS target, ramp up one worker at a time so we can observe memory before launching more.
                if total_rss_cap_bytes is not None:
                    break

            finished_worker_indices: list[int] = []
            for worker_index, (process, log_path, log_file) in processes.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                finished_worker_indices.append(worker_index)
                completed_workers += 1
                log_file.close()
                if return_code != 0:
                    failed_workers.append((worker_index, return_code, log_path))

            for worker_index in finished_worker_indices:
                del processes[worker_index]

            if failed_workers:
                break

            for worker_dir in worker_dirs:
                state = _load_state(worker_dir / STATE_FILENAME, allow_partial=True)
                processed_success += state.processed_success_episodes
                total_samples += state.total_samples
                total_kept_episodes += state.total_episodes

            if overall_progress is not None:
                overall_progress.n = processed_success
                overall_progress.set_postfix(
                    kept_episodes=total_kept_episodes,
                    samples=total_samples,
                    done=f"{completed_workers}/{split_worker_count}",
                    active=f"{len(processes)}/{max_concurrent_workers}",
                    rss_gb=f"{_format_gb(live_worker_rss_bytes):.1f}",
                    max_worker_gb=f"{_format_gb(max_live_worker_rss_bytes):.1f}",
                    over_worker_cap=str(worker_over_cap_count),
                )
                overall_progress.refresh()

            if completed_workers == split_worker_count:
                break
            time.sleep(1.0)

        if failed_workers:
            details = ", ".join(
                f"worker {worker_index} exit={return_code} (log: {log_path})"
                for worker_index, return_code, log_path in failed_workers
            )
            raise RuntimeError(f"Parallel conversion failed: {details}")
    except (KeyboardInterrupt, RuntimeError) as exc:
        logging.warning("Stopping worker processes...")
        for process, _, _ in processes.values():
            if process.poll() is None:
                process.terminate()

        deadline = time.time() + 8.0
        for process, _, _ in processes.values():
            if process.poll() is not None:
                continue
            timeout = max(0.0, deadline - time.time())
            if timeout <= 0:
                continue
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

        for process, _, _ in processes.values():
            if process.poll() is None:
                process.kill()

        for _, _, log_file in processes.values():
            try:
                log_file.close()
            except (OSError, ValueError):
                pass

        if isinstance(exc, RuntimeError):
            raise
        raise KeyboardInterrupt("Parallel conversion interrupted by user.") from exc
    finally:
        if overall_progress is not None:
            overall_progress.close()

    return worker_dirs


def _merge_worker_outputs(
    *,
    output_path: pathlib.Path,
    worker_dirs: list[pathlib.Path],
    dataset_name: str,
    dataset_version: str,
    action_horizon: int,
    action_space: str,
    filter_dict_path: str | None,
    filter_dict_sha256: str | None,
    cleanup_worker_outputs: bool,
    completed: bool,
) -> None:
    shard_infos: list[droid_sharded_dataset.DroidShardInfo] = []
    total_samples = 0
    total_episodes = 0
    processed_success_episodes = 0

    for worker_dir in worker_dirs:
        worker_manifest = droid_sharded_dataset.load_manifest(worker_dir)
        if worker_manifest.action_horizon != action_horizon:
            raise ValueError(
                f"Worker manifest action_horizon mismatch at {worker_dir}: "
                f"{worker_manifest.action_horizon} vs expected {action_horizon}"
            )
        if worker_manifest.action_space != action_space:
            raise ValueError(
                f"Worker manifest action_space mismatch at {worker_dir}: "
                f"{worker_manifest.action_space} vs expected {action_space}"
            )

        worker_state = _load_state(worker_dir / STATE_FILENAME)
        processed_success_episodes += worker_state.processed_success_episodes
        total_episodes += worker_manifest.total_episodes

        for worker_shard in worker_manifest.shards:
            source_shard_path = worker_dir / worker_shard.path
            target_shard_name = droid_sharded_dataset.SHARD_FILENAME_TEMPLATE.format(index=len(shard_infos))
            target_shard_path = output_path / target_shard_name
            shutil.move(source_shard_path, target_shard_path)

            shard_infos.append(
                droid_sharded_dataset.DroidShardInfo(
                    path=target_shard_name,
                    num_samples=worker_shard.num_samples,
                    start_index=total_samples,
                    end_index=total_samples + worker_shard.num_samples,
                )
            )
            total_samples += worker_shard.num_samples

    merged_manifest = droid_sharded_dataset.DroidShardManifest(
        format_version=droid_sharded_dataset.MANIFEST_VERSION,
        source_dataset_name=dataset_name,
        source_dataset_version=dataset_version,
        action_horizon=action_horizon,
        action_space=action_space,
        image_shape=(180, 320, 3),
        filter_dict_path=filter_dict_path,
        filter_dict_sha256=filter_dict_sha256,
        total_samples=total_samples,
        total_episodes=total_episodes,
        shards=tuple(shard_infos),
    )
    droid_sharded_dataset.write_manifest(output_path, merged_manifest)
    _save_state(
        output_path / STATE_FILENAME,
        ConversionState(
            processed_success_episodes=processed_success_episodes,
            total_samples=total_samples,
            total_episodes=total_episodes,
            completed=completed,
        ),
    )

    if cleanup_worker_outputs:
        shutil.rmtree(output_path / "_worker_outputs", ignore_errors=True)


def _convert_trajectory(
    traj: Mapping[str, object],
    *,
    filter_dict: Mapping[str, list[list[int]]] | None,
    action_space: droid_rlds_dataset.DroidActionSpace,
    action_horizon: int,
    decode_workers: int,
    tf,
) -> tuple[str, dict[str, np.ndarray | list[str]]]:
    metadata = traj["traj_metadata"]["episode_metadata"]
    recording_folderpath = _first_text_value(metadata["recording_folderpath"])
    file_path = _first_text_value(metadata["file_path"])
    episode_key = droid_sharded_dataset.make_episode_key(recording_folderpath, file_path)

    action_key = "joint_position" if action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION else "joint_velocity"
    joint_actions = np.asarray(traj["action_dict"][action_key], dtype=np.float32)
    gripper_actions = _normalize_gripper_position(np.asarray(traj["action_dict"]["gripper_position"], dtype=np.float32))
    actions = np.concatenate([joint_actions, gripper_actions], axis=-1)
    chunked_actions = droid_sharded_dataset.chunk_actions(actions, action_horizon)

    traj_len = actions.shape[0]
    keep_mask = (
        np.ones(traj_len, dtype=bool)
        if filter_dict is None
        else droid_sharded_dataset.create_filter_mask(traj_len, filter_dict.get(episode_key))
    )
    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        return episode_key, _empty_batch(action_horizon)

    observation = traj["observation"]
    joint_position = np.asarray(observation["joint_position"], dtype=np.float32)
    gripper_position = _normalize_gripper_position(np.asarray(observation["gripper_position"], dtype=np.float32))
    prompt_candidates = tuple(
        _first_text_value(candidate) if candidate is not None else ""
        for candidate in (
            traj.get("language_instruction"),
            traj.get("language_instruction_2"),
            traj.get("language_instruction_3"),
        )
    )

    exterior_1 = np.zeros((keep_indices.size, 180, 320, 3), dtype=np.uint8)
    exterior_2 = np.zeros((keep_indices.size, 180, 320, 3), dtype=np.uint8)
    wrist = np.zeros((keep_indices.size, 180, 320, 3), dtype=np.uint8)
    prompts: list[str] = []
    step_ids: list[str] = []
    episode_keys: list[str] = []

    exterior_source_1 = observation["exterior_image_1_left"]
    exterior_source_2 = observation["exterior_image_2_left"]
    wrist_source = observation["wrist_image_left"]

    if decode_workers <= 1:
        for output_index, timestep in enumerate(keep_indices):
            exterior_1[output_index] = _decode_image(exterior_source_1[timestep], tf)
            exterior_2[output_index] = _decode_image(exterior_source_2[timestep], tf)
            wrist[output_index] = _decode_image(wrist_source[timestep], tf)

            step_id = droid_sharded_dataset.make_step_id(episode_key, int(timestep))
            step_ids.append(step_id)
            episode_keys.append(episode_key)
            prompts.append(droid_sharded_dataset.select_prompt(prompt_candidates, step_id))
    else:
        def _decode_for_step(step_index: int, timestep: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
            return (
                step_index,
                _decode_image(exterior_source_1[timestep], tf),
                _decode_image(exterior_source_2[timestep], tf),
                _decode_image(wrist_source[timestep], tf),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as executor:
            futures = [
                executor.submit(_decode_for_step, step_index, int(timestep))
                for step_index, timestep in enumerate(keep_indices)
            ]
            for future in concurrent.futures.as_completed(futures):
                step_index, img1, img2, wrist_img = future.result()
                exterior_1[step_index] = img1
                exterior_2[step_index] = img2
                wrist[step_index] = wrist_img

        for timestep in keep_indices:
            step_id = droid_sharded_dataset.make_step_id(episode_key, int(timestep))
            step_ids.append(step_id)
            episode_keys.append(episode_key)
            prompts.append(droid_sharded_dataset.select_prompt(prompt_candidates, step_id))

    batch = {
        "exterior_image_1_left": exterior_1,
        "exterior_image_2_left": exterior_2,
        "wrist_image_left": wrist,
        "joint_position": joint_position[keep_indices],
        "gripper_position": gripper_position[keep_indices],
        "actions": chunked_actions[keep_indices],
        "prompt": prompts,
        "step_id": step_ids,
        "episode_key": episode_keys,
        "timestep": keep_indices.astype(np.int32),
    }
    return episode_key, batch


def main(
    data_dir: str,
    output_dir: str,
    *,
    dataset_name: str = "droid",
    dataset_version: str = "1.0.1",
    filter_dict_path: str | None = DEFAULT_FILTER_DICT,
    action_space: Literal["joint_position", "joint_velocity"] = "joint_position",
    action_horizon: int = 16,
    records_per_shard: int = 10_000,
    max_episodes: int | None = None,
    resume: bool = False,
    num_parallel_reads: int = -1,
    decode_workers: int = 1,
    tf_intra_op_threads: int | None = None,
    tf_inter_op_threads: int | None = None,
    split_worker_count: int = 1,
    split_worker_index: int | None = None,
    cleanup_worker_outputs: bool = True,
    show_progress: bool = True,
    max_worker_rss_gb: float | None = None,
    max_total_rss_gb: float | None = None,
    max_concurrent_workers: int | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _validate_legacy_rlds_env()
    _validate_worker_config(split_worker_count, split_worker_index, resume)
    if max_worker_rss_gb is not None and max_worker_rss_gb <= 0:
        raise ValueError(f"max_worker_rss_gb must be > 0, got {max_worker_rss_gb}")
    if max_total_rss_gb is not None and max_total_rss_gb <= 0:
        raise ValueError(f"max_total_rss_gb must be > 0, got {max_total_rss_gb}")
    if max_concurrent_workers is not None and max_concurrent_workers <= 0:
        raise ValueError(f"max_concurrent_workers must be > 0, got {max_concurrent_workers}")
    # Keep RLDS conversion CPU-only and avoid TensorFlow probing CUDA in every worker process.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    output_path = pathlib.Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if split_worker_count > 1 and split_worker_index is None:
        _validate_parallel_output_path(output_path)

        effective_max_concurrent_workers = split_worker_count if max_concurrent_workers is None else max_concurrent_workers
        if max_concurrent_workers is None and max_total_rss_gb is not None and max_worker_rss_gb is not None:
            inferred = max(1, int(max_total_rss_gb // max_worker_rss_gb))
            effective_max_concurrent_workers = min(split_worker_count, inferred)
            logging.info(
                "Auto-selected max_concurrent_workers=%d from memory hints "
                "(max_total_rss_gb=%.2f, max_worker_rss_gb=%.2f).",
                effective_max_concurrent_workers,
                max_total_rss_gb,
                max_worker_rss_gb,
            )
        effective_max_concurrent_workers = min(effective_max_concurrent_workers, split_worker_count)
        if effective_max_concurrent_workers < split_worker_count:
            logging.info(
                "Running workers in waves: %d total splits, %d concurrent workers.",
                split_worker_count,
                effective_max_concurrent_workers,
            )

        cached_filter_dict_path = None
        filter_dict_sha256 = None
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            resolved_filter_dict_path = _resolve_cached_filter_path(pathlib.Path(cached_filter_dict_path), filter_dict_path)
            filter_dict_sha256 = droid_sharded_dataset.compute_file_sha256(resolved_filter_dict_path)
            cached_filter_dict_path = str(resolved_filter_dict_path)

        worker_dirs = _run_parallel_workers(
            split_worker_count=split_worker_count,
            output_path=output_path,
            data_dir=data_dir,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filter_dict_path=filter_dict_path,
            action_space=action_space,
            action_horizon=action_horizon,
            records_per_shard=records_per_shard,
            max_episodes=max_episodes,
            num_parallel_reads=num_parallel_reads,
            decode_workers=decode_workers,
            tf_intra_op_threads=tf_intra_op_threads,
            tf_inter_op_threads=tf_inter_op_threads,
            show_progress=show_progress,
            max_worker_rss_gb=max_worker_rss_gb,
            max_total_rss_gb=max_total_rss_gb,
            max_concurrent_workers=effective_max_concurrent_workers,
        )
        _merge_worker_outputs(
            output_path=output_path,
            worker_dirs=worker_dirs,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            action_horizon=action_horizon,
            action_space=action_space,
            filter_dict_path=cached_filter_dict_path if cached_filter_dict_path is not None else filter_dict_path,
            filter_dict_sha256=filter_dict_sha256,
            cleanup_worker_outputs=cleanup_worker_outputs,
            completed=max_episodes is None,
        )
        logging.info("Finished parallel conversion at %s", output_path)
        return

    # Import TensorFlow lazily so the modern training environment never needs it.
    try:
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except Exception as exc:
        raise RuntimeError(
            "Failed to import RLDS dependencies (tensorflow/dlimp). "
            "Make sure you are running in an RLDS-only environment. "
            "Use: `uv sync --active --no-default-groups --group rlds` and then "
            "`uv run --active --no-sync --no-default-groups --group rlds "
            "openpi/scripts/convert_droid_rlds_to_shards.py ...`."
        ) from exc

    if tf_intra_op_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(tf_intra_op_threads)
    if tf_inter_op_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(tf_inter_op_threads)

    tf.config.set_visible_devices([], "GPU")
    state_path = output_path / STATE_FILENAME
    state = _load_state(state_path) if resume else ConversionState()
    if state.completed and resume:
        logging.info("Previous conversion is already marked complete. Nothing to do.")
        return

    cached_filter_dict_path = None
    filter_dict = None
    filter_dict_sha256 = None
    if filter_dict_path is not None:
        cached_filter_dict_path = download.maybe_download(filter_dict_path)
        resolved_filter_dict_path = _resolve_cached_filter_path(pathlib.Path(cached_filter_dict_path), filter_dict_path)
        with resolved_filter_dict_path.open() as f:
            filter_dict = json.load(f)
        filter_dict_sha256 = droid_sharded_dataset.compute_file_sha256(resolved_filter_dict_path)
        cached_filter_dict_path = str(resolved_filter_dict_path)
        logging.info("Loaded filter dictionary with %d episodes from %s", len(filter_dict), resolved_filter_dict_path)

    action_space_enum = droid_rlds_dataset.DroidActionSpace.JOINT_POSITION
    if action_space == "joint_velocity":
        action_space_enum = droid_rlds_dataset.DroidActionSpace.JOINT_VELOCITY

    writer = droid_sharded_dataset.DroidShardWriter(
        output_path,
        records_per_shard=records_per_shard,
        action_horizon=action_horizon,
        action_dim=8,
        resume=resume,
    )

    resolved_data_dir = _normalize_tfds_data_dir(data_dir, dataset_name, dataset_version)
    builder = tfds.builder(dataset_name, data_dir=resolved_data_dir, version=dataset_version)
    split = "train"
    worker_desc = "single-worker"
    if split_worker_index is not None:
        split = tfds.even_splits("train", split_worker_count, drop_remainder=False)[split_worker_index]
        worker_desc = f"worker {split_worker_index + 1}/{split_worker_count}"
    logging.info("Reading TFDS split %s (%s)", split, worker_desc)
    dataset = dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*")
    )
    if state.processed_success_episodes:
        logging.info("Skipping %d already-converted success episodes", state.processed_success_episodes)
        dataset = dataset.skip(state.processed_success_episodes)

    iterator = dataset.as_numpy_iterator()
    progress = None
    if show_progress:
        progress_desc = "Converting DROID RLDS"
        if split_worker_index is not None:
            progress_desc = f"Converting DROID RLDS [worker {split_worker_index + 1}/{split_worker_count}]"
        progress = tqdm.tqdm(total=max_episodes, dynamic_ncols=True, desc=progress_desc)

    try:
        for episode_index, traj in enumerate(iterator, start=1):
            episode_key, batch = _convert_trajectory(
                traj,
                filter_dict=filter_dict,
                action_space=action_space_enum,
                action_horizon=action_horizon,
                decode_workers=decode_workers,
                tf=tf,
            )
            sample_count = int(np.asarray(batch["timestep"]).shape[0])
            if sample_count:
                writer.append_batch(batch)
                state.total_samples += sample_count
                state.total_episodes += 1
            state.processed_success_episodes += 1
            state.completed = False
            _save_state(state_path, state)
            writer.write_manifest(
                source_dataset_name=dataset_name,
                source_dataset_version=dataset_version,
                action_horizon=action_horizon,
                action_space=action_space,
                filter_dict_path=str(cached_filter_dict_path) if cached_filter_dict_path is not None else filter_dict_path,
                filter_dict_sha256=filter_dict_sha256,
                total_episodes=state.total_episodes,
            )

            if progress is not None:
                progress.update(1)
                progress.set_postfix(samples=state.total_samples, kept_episodes=state.total_episodes, last=episode_key)

            if max_episodes is not None and episode_index >= max_episodes:
                break
    finally:
        if progress is not None:
            progress.close()
        writer.close()

    state.completed = max_episodes is None
    _save_state(state_path, state)
    writer.write_manifest(
        source_dataset_name=dataset_name,
        source_dataset_version=dataset_version,
        action_horizon=action_horizon,
        action_space=action_space,
        filter_dict_path=str(cached_filter_dict_path) if cached_filter_dict_path is not None else filter_dict_path,
        filter_dict_sha256=filter_dict_sha256,
        total_episodes=state.total_episodes,
    )
    logging.info(
        "Finished conversion with %d samples across %d kept episodes at %s",
        state.total_samples,
        state.total_episodes,
        output_path,
    )


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        logging.warning("Conversion interrupted by user.")
        raise SystemExit(130)
