from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from collections import OrderedDict
import bisect
import dataclasses
import hashlib
import json
import operator
import pathlib
from typing import Literal

import h5py
import numpy as np
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
SHARD_FILENAME_TEMPLATE = "shard-{index:05d}.h5"
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
IMAGE_FIELDS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
NUMERIC_FIELDS = ("joint_position", "gripper_position", "actions", "timestep")
STRING_FIELDS = ("prompt", "step_id", "episode_key")
ALL_FIELDS = (*IMAGE_FIELDS, *NUMERIC_FIELDS, *STRING_FIELDS)
ExteriorSamplingStrategy = Literal["random", "first", "second"]


@dataclasses.dataclass(frozen=True)
class DroidShardInfo:
    path: str
    num_samples: int
    start_index: int
    end_index: int


@dataclasses.dataclass(frozen=True)
class DroidShardManifest:
    format_version: int
    source_dataset_name: str
    source_dataset_version: str
    action_horizon: int
    action_space: str
    image_shape: tuple[int, int, int]
    filter_dict_path: str | None
    filter_dict_sha256: str | None
    total_samples: int
    total_episodes: int
    shards: tuple[DroidShardInfo, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DroidShardManifest":
        shards = tuple(
            DroidShardInfo(
                path=str(shard["path"]),
                num_samples=int(shard["num_samples"]),
                start_index=int(shard["start_index"]),
                end_index=int(shard["end_index"]),
            )
            for shard in data["shards"]
        )
        return cls(
            format_version=int(data["format_version"]),
            source_dataset_name=str(data["source_dataset_name"]),
            source_dataset_version=str(data["source_dataset_version"]),
            action_horizon=int(data["action_horizon"]),
            action_space=str(data["action_space"]),
            image_shape=tuple(int(v) for v in data["image_shape"]),
            filter_dict_path=data.get("filter_dict_path"),
            filter_dict_sha256=data.get("filter_dict_sha256"),
            total_samples=int(data["total_samples"]),
            total_episodes=int(data["total_episodes"]),
            shards=shards,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "source_dataset_name": self.source_dataset_name,
            "source_dataset_version": self.source_dataset_version,
            "action_horizon": self.action_horizon,
            "action_space": self.action_space,
            "image_shape": list(self.image_shape),
            "filter_dict_path": self.filter_dict_path,
            "filter_dict_sha256": self.filter_dict_sha256,
            "total_samples": self.total_samples,
            "total_episodes": self.total_episodes,
            "shards": [dataclasses.asdict(shard) for shard in self.shards],
        }


def load_manifest(path: str | pathlib.Path) -> DroidShardManifest:
    manifest_path = pathlib.Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    with manifest_path.open() as f:
        manifest = DroidShardManifest.from_dict(json.load(f))
    if manifest.format_version != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported DROID shard format version: {manifest.format_version}. Expected {MANIFEST_VERSION}."
        )
    return manifest


def write_manifest(path: str | pathlib.Path, manifest: DroidShardManifest) -> pathlib.Path:
    manifest_path = pathlib.Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def compute_file_sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_text(value.item())
        if value.shape == (1,):
            return decode_text(value[0])
        raise ValueError(f"Expected scalar string value, got shape {value.shape}")
    if value is None:
        return ""
    return str(value)


def make_episode_key(recording_folderpath: str, file_path: str) -> str:
    return f"{recording_folderpath}--{file_path}"


def make_step_id(episode_key: str, timestep: int) -> str:
    return f"{episode_key}--{timestep}"


def chunk_actions(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to have shape [T, D], got {actions.shape}")
    if actions.shape[0] == 0:
        return np.zeros((0, action_horizon, actions.shape[-1]), dtype=np.float32)
    action_indices = np.arange(action_horizon, dtype=np.int64)[None, :] + np.arange(actions.shape[0])[:, None]
    action_indices = np.minimum(action_indices, actions.shape[0] - 1)
    return np.asarray(actions[action_indices], dtype=np.float32)


def select_prompt(candidates: Sequence[object], step_id: str) -> str:
    prompts = [prompt for candidate in candidates if (prompt := decode_text(candidate).strip())]
    if not prompts:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    digest = hashlib.blake2b(step_id.encode("utf-8"), digest_size=8).digest()
    prompt_index = int.from_bytes(digest, "little") % len(prompts)
    return prompts[prompt_index]


def create_filter_mask(traj_len: int, ranges: Sequence[Sequence[int]] | None) -> np.ndarray:
    mask = np.zeros(traj_len, dtype=bool)
    if ranges is None:
        return mask
    for start, end in ranges:
        if start >= traj_len or end <= 0:
            continue
        mask[max(start, 0) : min(end, traj_len)] = True
    return mask


class DroidShardWriter:
    """Append-only HDF5 writer for the sharded DROID training dataset."""

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        records_per_shard: int,
        *,
        image_shape: tuple[int, int, int] = (180, 320, 3),
        action_horizon: int = 16,
        action_dim: int = 8,
        resume: bool = False,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._records_per_shard = records_per_shard
        self._image_shape = image_shape
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._current_index = 0
        self._current_path: pathlib.Path | None = None
        self._current_count = 0
        self._current_file: h5py.File | None = None

        existing_shards = self._list_shards()
        if existing_shards and not resume:
            raise FileExistsError(
                f"Found existing shard files in {self._output_dir}. Pass resume=True to append to them."
            )
        if resume and existing_shards:
            last_index, last_path = existing_shards[-1]
            last_count = self._count_records(last_path)
            if last_count < self._records_per_shard:
                self._current_index = last_index
                self._current_path = last_path
                self._current_file = h5py.File(last_path, "a")
                self._current_count = last_count
            else:
                self._current_index = last_index + 1

    def close(self) -> None:
        if self._current_file is None:
            return
        try:
            self._current_file.attrs["num_samples"] = self._current_count
            self._current_file.close()
        except (AttributeError, TypeError, ValueError):
            pass
        finally:
            self._current_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def append_batch(self, batch: Mapping[str, np.ndarray | Sequence[str]]) -> None:
        batch_size = self._infer_batch_size(batch)
        if batch_size == 0:
            return

        offset = 0
        while offset < batch_size:
            shard = self._ensure_current_shard()
            space = self._records_per_shard - self._current_count
            write_size = min(space, batch_size - offset)
            self._append_slice(shard, batch, offset, offset + write_size)
            offset += write_size
            self._current_count += write_size
            shard.attrs["num_samples"] = self._current_count
            if self._current_count == self._records_per_shard:
                self.close()
                self._current_index += 1
                self._current_path = None
                self._current_count = 0

    def build_manifest(
        self,
        *,
        source_dataset_name: str,
        source_dataset_version: str,
        action_horizon: int,
        action_space: str,
        filter_dict_path: str | None,
        filter_dict_sha256: str | None,
        total_episodes: int,
    ) -> DroidShardManifest:
        self.close()
        shards: list[DroidShardInfo] = []
        start_index = 0
        for _, shard_path in self._list_shards():
            num_samples = self._count_records(shard_path)
            shards.append(
                DroidShardInfo(
                    path=shard_path.name,
                    num_samples=num_samples,
                    start_index=start_index,
                    end_index=start_index + num_samples,
                )
            )
            start_index += num_samples
        return DroidShardManifest(
            format_version=MANIFEST_VERSION,
            source_dataset_name=source_dataset_name,
            source_dataset_version=source_dataset_version,
            action_horizon=action_horizon,
            action_space=action_space,
            image_shape=self._image_shape,
            filter_dict_path=filter_dict_path,
            filter_dict_sha256=filter_dict_sha256,
            total_samples=start_index,
            total_episodes=total_episodes,
            shards=tuple(shards),
        )

    def write_manifest(self, **kwargs) -> pathlib.Path:
        return write_manifest(self._output_dir, self.build_manifest(**kwargs))

    def _ensure_current_shard(self) -> h5py.File:
        if self._current_file is not None:
            return self._current_file
        self._current_path = self._output_dir / SHARD_FILENAME_TEMPLATE.format(index=self._current_index)
        self._current_file = h5py.File(self._current_path, "a")
        self._current_count = self._count_records(self._current_path)
        if "timestep" not in self._current_file:
            self._initialize_shard(self._current_file)
        return self._current_file

    def _initialize_shard(self, shard: h5py.File) -> None:
        for field in IMAGE_FIELDS:
            shard.create_dataset(
                field,
                shape=(0, *self._image_shape),
                maxshape=(None, *self._image_shape),
                chunks=(1, *self._image_shape),
                compression="lzf",
                dtype=np.uint8,
            )
        shard.create_dataset(
            "joint_position",
            shape=(0, 7),
            maxshape=(None, 7),
            chunks=(64, 7),
            compression="lzf",
            dtype=np.float32,
        )
        shard.create_dataset(
            "gripper_position",
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=(64, 1),
            compression="lzf",
            dtype=np.float32,
        )
        shard.create_dataset(
            "actions",
            shape=(0, self._action_horizon, self._action_dim),
            maxshape=(None, self._action_horizon, self._action_dim),
            chunks=(8, self._action_horizon, self._action_dim),
            compression="lzf",
            dtype=np.float32,
        )
        shard.create_dataset("timestep", shape=(0,), maxshape=(None,), chunks=(256,), compression="lzf", dtype=np.int32)
        for field in STRING_FIELDS:
            shard.create_dataset(field, shape=(0,), maxshape=(None,), chunks=(256,), dtype=STRING_DTYPE)

    def _append_slice(
        self,
        shard: h5py.File,
        batch: Mapping[str, np.ndarray | Sequence[str]],
        start: int,
        end: int,
    ) -> None:
        for field in ALL_FIELDS:
            values = batch[field]
            if field in STRING_FIELDS:
                field_values = [decode_text(v) for v in values[start:end]]
            else:
                field_values = np.asarray(values[start:end])
            dataset = shard[field]
            dataset.resize(dataset.shape[0] + (end - start), axis=0)
            dataset[-(end - start) :] = field_values

    def _infer_batch_size(self, batch: Mapping[str, np.ndarray | Sequence[str]]) -> int:
        sizes = {len(values) for values in batch.values()}
        if len(sizes) != 1:
            raise ValueError(f"Expected every field in batch to have the same size, got {sizes}.")
        return sizes.pop()

    def _list_shards(self) -> list[tuple[int, pathlib.Path]]:
        shards: list[tuple[int, pathlib.Path]] = []
        for shard_path in sorted(self._output_dir.glob("shard-*.h5")):
            index = int(shard_path.stem.split("-")[-1])
            shards.append((index, shard_path))
        return shards

    def _count_records(self, shard_path: pathlib.Path) -> int:
        with h5py.File(shard_path, "r") as shard:
            if "timestep" not in shard:
                return 0
            return int(shard["timestep"].shape[0])


class ShardedDroidDataset:
    """Random-access DROID dataset backed by local HDF5 shards."""

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        *,
        seed: int = 0,
        exterior_sampling_strategy: ExteriorSamplingStrategy = "random",
        max_open_shards: int = 32,
    ) -> None:
        self._data_dir = pathlib.Path(data_dir).resolve()
        self._manifest = load_manifest(self._data_dir)
        self._seed = seed
        self._exterior_sampling_strategy = exterior_sampling_strategy
        if max_open_shards < 1:
            raise ValueError(f"max_open_shards must be >= 1, got {max_open_shards}")
        self._max_open_shards = max_open_shards
        self._shard_paths = tuple(self._data_dir / shard.path for shard in self._manifest.shards)
        self._shard_end_indices = tuple(shard.end_index for shard in self._manifest.shards)
        self._shard_handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self._manifest.total_samples

    @property
    def manifest(self) -> DroidShardManifest:
        return self._manifest

    def read_raw_item(self, index: int) -> dict[str, object]:
        shard_index, local_index = self._locate(index)
        shard = self._get_shard(shard_index)
        return {
            "observation": {
                "exterior_image_1_left": np.asarray(shard["exterior_image_1_left"][local_index], dtype=np.uint8),
                "exterior_image_2_left": np.asarray(shard["exterior_image_2_left"][local_index], dtype=np.uint8),
                "wrist_image_left": np.asarray(shard["wrist_image_left"][local_index], dtype=np.uint8),
                "joint_position": np.asarray(shard["joint_position"][local_index], dtype=np.float32),
                "gripper_position": np.asarray(shard["gripper_position"][local_index], dtype=np.float32),
            },
            "actions": np.asarray(shard["actions"][local_index], dtype=np.float32),
            "prompt": decode_text(shard["prompt"][local_index]),
            "step_id": decode_text(shard["step_id"][local_index]),
            "episode_key": decode_text(shard["episode_key"][local_index]),
            "timestep": np.asarray(shard["timestep"][local_index], dtype=np.int32),
        }

    def read_exterior_views(self, index: int) -> dict[str, np.ndarray]:
        raw_item = self.read_raw_item(index)
        observation = raw_item["observation"]
        return {
            "exterior_image_1_left": observation["exterior_image_1_left"],
            "exterior_image_2_left": observation["exterior_image_2_left"],
            "wrist_image_left": observation["wrist_image_left"],
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        raw_item = self.read_raw_item(index)
        observation = raw_item["observation"]
        exterior_field = self._select_exterior_field()
        return {
            "observation": {
                "image": observation[exterior_field],
                "wrist_image": observation["wrist_image_left"],
                "joint_position": observation["joint_position"],
                "gripper_position": observation["gripper_position"],
            },
            "actions": raw_item["actions"],
            "prompt": raw_item["prompt"],
            "step_id": raw_item["step_id"],
        }

    def close(self) -> None:
        for shard in self._shard_handles.values():
            try:
                shard.close()
            except (AttributeError, TypeError, ValueError):
                pass
        self._shard_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_shard_handles"] = OrderedDict()
        state["_rng"] = None
        return state

    def _locate(self, index: int) -> tuple[int, int]:
        normalized_index = operator.index(index)
        if normalized_index < 0:
            normalized_index += len(self)
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        shard_index = bisect.bisect_right(self._shard_end_indices, normalized_index)
        shard_start = 0 if shard_index == 0 else self._manifest.shards[shard_index - 1].end_index
        return shard_index, normalized_index - shard_start

    def _get_shard(self, shard_index: int) -> h5py.File:
        shard = self._shard_handles.get(shard_index)
        if shard is None:
            shard = h5py.File(self._shard_paths[shard_index], "r")
            self._shard_handles[shard_index] = shard
            self._evict_old_shards()
        else:
            self._shard_handles.move_to_end(shard_index)
        return shard

    def _evict_old_shards(self) -> None:
        while len(self._shard_handles) > self._max_open_shards:
            _, evicted = self._shard_handles.popitem(last=False)
            try:
                evicted.close()
            except (AttributeError, TypeError, ValueError):
                pass

    def _select_exterior_field(self) -> str:
        if self._exterior_sampling_strategy == "first":
            return "exterior_image_1_left"
        if self._exterior_sampling_strategy == "second":
            return "exterior_image_2_left"
        if self._get_rng().integers(0, 2) == 0:
            return "exterior_image_1_left"
        return "exterior_image_2_left"

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker_info = None if torch is None else torch.utils.data.get_worker_info()
            seed = worker_info.seed if worker_info is not None else self._seed
            self._rng = np.random.default_rng(seed)
        return self._rng
