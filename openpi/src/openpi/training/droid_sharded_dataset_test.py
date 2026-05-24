import dataclasses
import importlib.util
import os
import pathlib

import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import droid_rlds_dataset as _droid_rlds_dataset
from openpi.training import droid_sharded_dataset as _droid_sharded_dataset


def _load_converter_module():
    script_path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "convert_droid_rlds_to_shards.py"
    spec = importlib.util.spec_from_file_location("convert_droid_rlds_to_shards", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_test_shards(tmp_path: pathlib.Path, *, num_samples: int = 5, records_per_shard: int = 2) -> pathlib.Path:
    output_dir = tmp_path / "droid_shards"
    writer = _droid_sharded_dataset.DroidShardWriter(output_dir, records_per_shard=records_per_shard, resume=False)

    images_1 = np.stack(
        [np.full((180, 320, 3), fill_value=index, dtype=np.uint8) for index in range(num_samples)],
        axis=0,
    )
    images_2 = np.stack(
        [np.full((180, 320, 3), fill_value=100 + index, dtype=np.uint8) for index in range(num_samples)],
        axis=0,
    )
    wrist = np.stack(
        [np.full((180, 320, 3), fill_value=200 + index, dtype=np.uint8) for index in range(num_samples)],
        axis=0,
    )
    writer.append_batch(
        {
            "exterior_image_1_left": images_1,
            "exterior_image_2_left": images_2,
            "wrist_image_left": wrist,
            "joint_position": np.tile(np.arange(7, dtype=np.float32), (num_samples, 1)) + np.arange(num_samples)[:, None],
            "gripper_position": np.arange(num_samples, dtype=np.float32)[:, None],
            "actions": np.tile(np.arange(16 * 8, dtype=np.float32).reshape(16, 8), (num_samples, 1, 1)),
            "prompt": [f"prompt-{index}" for index in range(num_samples)],
            "step_id": [f"episode-{index}--{index}" for index in range(num_samples)],
            "episode_key": [f"episode-{index}" for index in range(num_samples)],
            "timestep": np.arange(num_samples, dtype=np.int32),
        }
    )
    writer.write_manifest(
        source_dataset_name="droid",
        source_dataset_version="1.0.1",
        action_horizon=16,
        action_space="joint_position",
        filter_dict_path="/tmp/filter.json",
        filter_dict_sha256="deadbeef",
        total_episodes=num_samples,
    )
    writer.close()
    return output_dir


def test_sharded_droid_dataset_indexing(tmp_path: pathlib.Path):
    shard_dir = _make_test_shards(tmp_path)
    dataset = _droid_sharded_dataset.ShardedDroidDataset(shard_dir, exterior_sampling_strategy="first")

    assert len(dataset) == 5
    assert dataset.manifest.total_samples == 5

    raw_item = dataset.read_raw_item(3)
    assert raw_item["prompt"] == "prompt-3"
    assert raw_item["step_id"] == "episode-3--3"
    assert raw_item["episode_key"] == "episode-3"
    assert int(raw_item["timestep"]) == 3
    assert raw_item["observation"]["exterior_image_2_left"][0, 0, 0] == 103

    sample = dataset[3]
    assert sample["observation"]["image"][0, 0, 0] == 3
    assert sample["observation"]["wrist_image"][0, 0, 0] == 203


def test_sharded_droid_dataset_random_sampling_is_repeatable_with_seed(tmp_path: pathlib.Path):
    shard_dir = _make_test_shards(tmp_path)
    dataset_a = _droid_sharded_dataset.ShardedDroidDataset(shard_dir, seed=7, exterior_sampling_strategy="random")
    dataset_b = _droid_sharded_dataset.ShardedDroidDataset(shard_dir, seed=7, exterior_sampling_strategy="random")

    sequence_a = [int(dataset_a[index]["observation"]["image"][0, 0, 0]) for index in range(4)]
    sequence_b = [int(dataset_b[index]["observation"]["image"][0, 0, 0]) for index in range(4)]

    assert sequence_a == sequence_b


def test_sharded_droid_dataset_caps_open_files(tmp_path: pathlib.Path):
    shard_dir = _make_test_shards(tmp_path, num_samples=10, records_per_shard=1)
    dataset = _droid_sharded_dataset.ShardedDroidDataset(
        shard_dir,
        exterior_sampling_strategy="first",
        max_open_shards=2,
    )

    for index in range(len(dataset)):
        _ = dataset[index]
        assert len(dataset._shard_handles) <= 2  # noqa: SLF001

    dataset.close()


def test_sharded_droid_loader_integration(tmp_path: pathlib.Path):
    shard_dir = _make_test_shards(tmp_path)
    config = dataclasses.replace(
        _config.get_config("pi05_full_droid_modern_finetune"),
        batch_size=2,
        num_workers=0,
    )
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sharded_droid_dir=str(shard_dir)))

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2
    for observation, actions in batches:
        assert actions.shape == (2, 16, 32)
        assert observation.state.shape == (2, 32)


def test_sharded_droid_loader_parallel(tmp_path: pathlib.Path):
    shard_dir = _make_test_shards(tmp_path, num_samples=6, records_per_shard=3)
    config = dataclasses.replace(
        _config.get_config("pi05_full_droid_modern_finetune"),
        batch_size=2,
        num_workers=2,
    )
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sharded_droid_dir=str(shard_dir)))

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2
    for observation, actions in batches:
        assert actions.shape == (2, 16, 32)
        assert observation.state.shape == (2, 32)


@pytest.mark.manual
def test_rlds_conversion_parity_manual(tmp_path: pathlib.Path):
    data_dir = os.getenv("OPENPI_DROID_RLDS_DIR")
    if data_dir is None:
        pytest.skip("Set OPENPI_DROID_RLDS_DIR to a local DROID RLDS dataset to run this parity check.")

    dl = pytest.importorskip("dlimp")
    tf = pytest.importorskip("tensorflow")
    tfds = pytest.importorskip("tensorflow_datasets")

    output_dir = tmp_path / "converted_shards"
    convert_droid_rlds_to_shards.main(data_dir, str(output_dir), max_episodes=1, filter_dict_path=None, resume=False)

    dataset = _droid_sharded_dataset.ShardedDroidDataset(output_dir, exterior_sampling_strategy="first")
    assert len(dataset) > 0

    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    rlds_dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=1)
    rlds_dataset = rlds_dataset.filter(
        lambda traj: tf.strings.regex_full_match(traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*")
    )
    traj = next(rlds_dataset.as_numpy_iterator())
    _, converted_batch = convert_droid_rlds_to_shards._convert_trajectory(  # noqa: SLF001
        traj,
        filter_dict=None,
        action_space=_droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        action_horizon=16,
        tf=tf,
    )

    raw_item = dataset.read_raw_item(0)
    np.testing.assert_array_equal(raw_item["observation"]["exterior_image_1_left"], converted_batch["exterior_image_1_left"][0])
    np.testing.assert_array_equal(raw_item["actions"], converted_batch["actions"][0])
    assert raw_item["step_id"] == converted_batch["step_id"][0]
    convert_droid_rlds_to_shards = _load_converter_module()
