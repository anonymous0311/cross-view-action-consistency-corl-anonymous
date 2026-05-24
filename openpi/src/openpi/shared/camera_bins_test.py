import numpy as np
import pandas as pd

from openpi.shared import camera_bins as _camera_bins


def test_encode_camera_params_handles_wraparound():
    camera_params = np.asarray(
        [
            [359.0, 0.0, 100.0, 358.0, 2.0],
            [1.0, 0.0, 100.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    encoded = _camera_bins.encode_camera_params(camera_params)

    assert encoded.shape == (2, len(_camera_bins.ENCODED_CAMERA_FEATURE_COLS))
    assert np.linalg.norm(encoded[0] - encoded[1]) < 0.1


def test_camera_bin_assigner_uses_wrapsafe_feature_space():
    raw_centroids = np.asarray(
        [
            [359.0, 0.0, 100.0, 358.0, 2.0],
            [180.0, 15.0, 200.0, 180.0, 180.0],
        ],
        dtype=np.float32,
    )
    encoded_centroids = _camera_bins.encode_camera_params(raw_centroids)
    bins_df = pd.DataFrame(
        {
            "bin_id": [0, 1],
            "count": [10, 10],
            **{
                feature: raw_centroids[:, idx]
                for idx, feature in enumerate(_camera_bins.RAW_CAMERA_FEATURE_COLS)
            },
            **{
                feature: encoded_centroids[:, idx]
                for idx, feature in enumerate(_camera_bins.ENCODED_CAMERA_FEATURE_COLS)
            },
        }
    )
    scaler_df = pd.DataFrame(
        {
            "feature": _camera_bins.ENCODED_CAMERA_FEATURE_COLS,
            "mean": np.zeros(len(_camera_bins.ENCODED_CAMERA_FEATURE_COLS), dtype=np.float32),
            "std": np.ones(len(_camera_bins.ENCODED_CAMERA_FEATURE_COLS), dtype=np.float32),
        }
    )

    assigner = _camera_bins.CameraBinAssigner.from_dataframes(bins_df, scaler_df)
    assigned = assigner.assign(np.asarray([[1.0, 0.0, 100.0, 0.0, 0.0]], dtype=np.float32))

    assert assigned.tolist() == [0]
