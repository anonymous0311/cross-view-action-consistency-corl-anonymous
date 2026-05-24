"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import os
import pathlib

# This script is data-loader bound and computes simple running stats. Force CPU JAX backend by default
# to avoid expensive/irrelevant CUDA plugin initialization checks in worker processes.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Suppress CUDA plugin version checks in this CPU-only stats script.
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.droid_sharded_dataset as droid_sharded_dataset
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute_sharded_droid_norm_stats(
    data_config: _config.DataConfig,
    *,
    max_frames: int | None,
    chunk_rows: int,
) -> dict[str, normalize.NormStats]:
    """Fast path for modern DROID shards.

    Reads only the numeric fields needed for norm stats from HDF5 shards, bypassing image loading
    and the PyTorch DataLoader pipeline.
    """
    if data_config.sharded_droid_dir is None:
        raise ValueError("Expected sharded_droid_dir to be set for sharded DROID stats computation.")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}.")

    import h5py

    shard_dir = pathlib.Path(data_config.sharded_droid_dir).resolve()
    manifest = droid_sharded_dataset.load_manifest(shard_dir)
    total_rows = manifest.total_samples
    target_rows = total_rows if max_frames is None else min(max_frames, total_rows)

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    processed_rows = 0
    with tqdm.tqdm(total=target_rows, desc="Computing stats [sharded fast]", unit="frames") as pbar:
        for shard_info in manifest.shards:
            if processed_rows >= target_rows:
                break
            remaining = target_rows - processed_rows
            rows_in_shard = min(shard_info.num_samples, remaining)
            if rows_in_shard <= 0:
                continue

            shard_path = shard_dir / shard_info.path
            with h5py.File(shard_path, "r") as shard:
                for start in range(0, rows_in_shard, chunk_rows):
                    end = min(start + chunk_rows, rows_in_shard)
                    joint = np.asarray(shard["joint_position"][start:end], dtype=np.float32)
                    gripper = np.asarray(shard["gripper_position"][start:end], dtype=np.float32)
                    actions = np.asarray(shard["actions"][start:end], dtype=np.float32)

                    state = np.concatenate([joint, gripper], axis=-1)
                    if data_config.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
                        actions = actions.copy()
                        actions[..., :7] -= state[:, None, :7]

                    stats["state"].update(state)
                    stats["actions"].update(actions)
                    delta = end - start
                    processed_rows += delta
                    pbar.update(delta)

    return {key: stats_obj.get_statistics() for key, stats_obj in stats.items()}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    seed: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config, seed=seed)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    num_workers: int | None = None,
    batch_size: int | None = None,
    jax_platform: str = "cpu",
    fast_sharded_droid: bool = True,
    shard_chunk_rows: int = 8192,
):
    if jax_platform:
        os.environ["JAX_PLATFORMS"] = jax_platform
        if jax_platform == "cpu":
            os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

    config = _config.get_config(config_name)
    if num_workers is not None:
        config = dataclasses.replace(config, num_workers=num_workers)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    data_config = config.data.create(config.assets_dirs, config.model)

    if fast_sharded_droid and data_config.sharded_droid_dir is not None and data_config.rlds_data_dir is None:
        norm_stats = compute_sharded_droid_norm_stats(
            data_config,
            max_frames=max_frames,
            chunk_rows=shard_chunk_rows,
        )
        output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            config.seed,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
