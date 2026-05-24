from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib

import numpy as np
import pandas as pd

RAW_CAMERA_FEATURE_COLS: tuple[str, ...] = (
    "label_horizon",
    "label_vertical",
    "label_scale",
    "label_end_rot",
    "label_end_vert",
)

_CIRCULAR_CAMERA_FEATURE_COLS = frozenset(
    {
        "label_horizon",
        "label_end_rot",
        "label_end_vert",
    }
)

ENCODED_CAMERA_FEATURE_COLS: tuple[str, ...] = (
    "label_horizon_cos",
    "label_horizon_sin",
    "label_vertical",
    "label_scale",
    "label_end_rot_cos",
    "label_end_rot_sin",
    "label_end_vert_cos",
    "label_end_vert_sin",
)

_EPS = 1e-8
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_CAMERA_BINS_PATH = _REPO_ROOT / "assets" / "camera_bins_k8.parquet"
DEFAULT_CAMERA_EPISODE_PARAMS_PATH = _REPO_ROOT / "assets" / "m7_boundary" / "episode_camera_params.parquet"


def default_camera_scaler_path(camera_bins_path: str | pathlib.Path) -> pathlib.Path:
    camera_bins_path = pathlib.Path(camera_bins_path)
    return camera_bins_path.with_name(camera_bins_path.name.replace("camera_bins", "camera_scaler"))


def encode_camera_params(camera_params: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Encode raw camera labels into a wrap-safe feature space for clustering."""
    raw = np.asarray(camera_params, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[-1] != len(RAW_CAMERA_FEATURE_COLS):
        raise ValueError(
            f"Expected last dimension {len(RAW_CAMERA_FEATURE_COLS)} for camera params, got {raw.shape[-1]}"
        )

    parts: list[np.ndarray] = []
    for idx, feature in enumerate(RAW_CAMERA_FEATURE_COLS):
        values = raw[..., idx]
        if feature in _CIRCULAR_CAMERA_FEATURE_COLS:
            radians = np.deg2rad(np.mod(values, 360.0))
            parts.append(np.cos(radians))
            parts.append(np.sin(radians))
        else:
            parts.append(values)
    return np.stack(parts, axis=-1).astype(np.float32)


def circular_mean_degrees(values: np.ndarray | Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float32)
    radians = np.deg2rad(np.mod(values, 360.0))
    sin_mean = np.nanmean(np.sin(radians))
    cos_mean = np.nanmean(np.cos(radians))
    return float(np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0))


def summarize_camera_bins(raw_camera_params: np.ndarray, labels: np.ndarray, num_bins: int) -> pd.DataFrame:
    raw = np.asarray(raw_camera_params, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != len(RAW_CAMERA_FEATURE_COLS):
        raise ValueError(f"Expected raw camera params with shape [N, {len(RAW_CAMERA_FEATURE_COLS)}], got {raw.shape}")

    rows: list[dict[str, float | int]] = []
    for bin_id in range(num_bins):
        mask = labels == bin_id
        if not np.any(mask):
            raise ValueError(f"Cluster {bin_id} is empty.")
        members = raw[mask]
        row: dict[str, float | int] = {"bin_id": int(bin_id), "count": int(mask.sum())}
        for idx, feature in enumerate(RAW_CAMERA_FEATURE_COLS):
            values = members[:, idx]
            row[feature] = (
                circular_mean_degrees(values) if feature in _CIRCULAR_CAMERA_FEATURE_COLS else float(np.mean(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_frame_to_array(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    missing = [feature for feature in feature_cols if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return frame.loc[:, list(feature_cols)].to_numpy(dtype=np.float32)


def _prepare_runtime_features(
    camera_params: np.ndarray | Sequence[Sequence[float]],
    runtime_feature_cols: Sequence[str],
) -> np.ndarray:
    if tuple(runtime_feature_cols) == RAW_CAMERA_FEATURE_COLS:
        features = np.asarray(camera_params, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        return features
    if tuple(runtime_feature_cols) == ENCODED_CAMERA_FEATURE_COLS:
        return encode_camera_params(camera_params)
    raise ValueError(f"Unsupported camera feature space: {tuple(runtime_feature_cols)}")


@dataclasses.dataclass(frozen=True)
class CameraBinAssigner:
    centroids: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    feature_cols: tuple[str, ...]

    @classmethod
    def from_dataframes(cls, bins_df: pd.DataFrame, scaler_df: pd.DataFrame) -> "CameraBinAssigner":
        if "feature" not in scaler_df.columns:
            raise ValueError('Scaler parquet must contain a "feature" column.')
        feature_cols = tuple(str(v) for v in scaler_df["feature"].tolist())
        centroids = _feature_frame_to_array(bins_df, feature_cols)
        scaler_index = scaler_df.set_index("feature")
        mean = scaler_index.loc[list(feature_cols), "mean"].to_numpy(dtype=np.float32)
        std = scaler_index.loc[list(feature_cols), "std"].to_numpy(dtype=np.float32)
        return cls(centroids=centroids, scaler_mean=mean, scaler_std=std, feature_cols=feature_cols)

    @classmethod
    def from_parquet(
        cls,
        camera_bins_path: str | pathlib.Path,
        camera_scaler_path: str | pathlib.Path | None = None,
    ) -> "CameraBinAssigner":
        camera_bins_path = pathlib.Path(camera_bins_path)
        camera_scaler_path = default_camera_scaler_path(camera_bins_path) if camera_scaler_path is None else pathlib.Path(camera_scaler_path)
        return cls.from_dataframes(
            pd.read_parquet(camera_bins_path),
            pd.read_parquet(camera_scaler_path),
        )

    def assign(self, camera_params: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        features = _prepare_runtime_features(camera_params, self.feature_cols)
        if np.isnan(features).any():
            raise ValueError("Camera params contain NaNs; cannot assign camera bins.")
        scaled = (features - self.scaler_mean) / (self.scaler_std + _EPS)
        centroids_scaled = (self.centroids - self.scaler_mean) / (self.scaler_std + _EPS)
        dists = np.linalg.norm(scaled[:, None, :] - centroids_scaled[None, :, :], axis=-1)
        return np.argmin(dists, axis=1).astype(np.int32)


def assign_camera_bin_ids(
    camera_params: np.ndarray | Sequence[Sequence[float]],
    bins_df: pd.DataFrame,
    scaler_df: pd.DataFrame,
) -> np.ndarray:
    return CameraBinAssigner.from_dataframes(bins_df, scaler_df).assign(camera_params)


def build_episode_camera_bin_lookup(
    episode_params_path: str | pathlib.Path = DEFAULT_CAMERA_EPISODE_PARAMS_PATH,
    camera_bins_path: str | pathlib.Path = DEFAULT_CAMERA_BINS_PATH,
    camera_scaler_path: str | pathlib.Path | None = None,
) -> Mapping[int, int]:
    episode_params = pd.read_parquet(episode_params_path)
    if "episode_index" not in episode_params.columns:
        raise ValueError('Episode camera params parquet must contain "episode_index".')
    raw = _feature_frame_to_array(episode_params, RAW_CAMERA_FEATURE_COLS)
    bin_ids = CameraBinAssigner.from_parquet(camera_bins_path, camera_scaler_path).assign(raw)
    return {
        int(episode_index): int(bin_id)
        for episode_index, bin_id in zip(episode_params["episode_index"].to_numpy(), bin_ids, strict=True)
    }
