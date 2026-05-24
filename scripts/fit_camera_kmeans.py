"""
Fit K-means camera bins from LIBERO-plus episode camera parameters.

Usage:
    uv run python scripts/fit_camera_kmeans.py [--k 8] [--seed 42]

Outputs (relative to repo root):
    assets/camera_bins_k{k}.parquet   -- raw centroids + wrap-safe encoded centroids + bin_id
    assets/camera_scaler_k{k}.parquet -- StandardScaler params for the runtime feature space
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
for rel in ("openpi/src", "openpi/packages/openpi-client/src"):
    abs_path = REPO_ROOT / rel
    if abs_path.exists() and str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))

from openpi.shared import camera_bins as _camera_bins

INPUT_PARQUET = REPO_ROOT / "assets" / "m7_boundary" / "episode_camera_params.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Fit K-means camera bins from LIBERO-plus episode camera params."
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    k = args.k
    seed = args.seed

    out_bins = REPO_ROOT / "assets" / f"camera_bins_k{k}.parquet"
    out_scaler = REPO_ROOT / "assets" / f"camera_scaler_k{k}.parquet"

    print(f"Loading {INPUT_PARQUET} ...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Loaded {len(df)} episodes, {len(df.columns)} columns.")

    raw_camera = df.loc[:, list(_camera_bins.RAW_CAMERA_FEATURE_COLS)].to_numpy(dtype=np.float32)
    encoded_camera = _camera_bins.encode_camera_params(raw_camera)
    print(f"  Raw feature matrix shape: {raw_camera.shape}")
    print(f"  Encoded feature matrix shape: {encoded_camera.shape}")

    scaler = StandardScaler()
    encoded_scaled = scaler.fit_transform(encoded_camera)

    print(f"\nFitting KMeans(n_clusters={k}, n_init=20, random_state={seed}) ...")
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed)
    kmeans.fit(encoded_scaled)
    print(f"  Inertia: {kmeans.inertia_:.4f}")

    labels = kmeans.labels_
    centers_encoded = scaler.inverse_transform(kmeans.cluster_centers_)

    reassigned = _camera_bins.CameraBinAssigner(
        centroids=centers_encoded.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_std=scaler.scale_.astype(np.float32),
        feature_cols=_camera_bins.ENCODED_CAMERA_FEATURE_COLS,
    ).assign(raw_camera)
    assert np.array_equal(labels, reassigned), "Re-assignment mismatch: helper assignment disagrees with kmeans.labels_."
    print("  Validation passed: helper re-assignment matches kmeans.labels_.")

    bin_sizes = np.bincount(labels, minlength=k)
    max_bin = int(bin_sizes.max())
    min_bin = int(bin_sizes.min())
    ratio = max_bin / max(min_bin, 1)
    print("\nCluster sizes (episodes per bin):")
    for i, size in enumerate(bin_sizes):
        print(f"  bin {i:2d}: {size:4d} episodes")
    print(f"\nMin bin: {min_bin}, Max bin: {max_bin}, Ratio: {ratio:.2f}x")
    if ratio >= 10.0:
        print(
            f"WARNING: max/min bin size ratio {ratio:.2f}x >= 10x. "
            "Camera distribution is highly skewed."
        )
    else:
        print("  OK: ratio < 10x.")

    bins_df = _camera_bins.summarize_camera_bins(raw_camera, labels, k)
    for idx, feature in enumerate(_camera_bins.ENCODED_CAMERA_FEATURE_COLS):
        bins_df[feature] = centers_encoded[:, idx].astype(np.float32)
    bins_df.to_parquet(out_bins, index=False)
    print(f"\nSaved centroids to {out_bins}")

    scaler_df = pd.DataFrame(
        {
            "feature": _camera_bins.ENCODED_CAMERA_FEATURE_COLS,
            "mean": scaler.mean_.astype(np.float32),
            "std": scaler.scale_.astype(np.float32),
        }
    )
    scaler_df.to_parquet(out_scaler, index=False)
    print(f"Saved scaler params to {out_scaler}")

    print("\nCentroids in original label space:")
    printable_cols = ["bin_id", "count", *_camera_bins.RAW_CAMERA_FEATURE_COLS]
    print(bins_df.loc[:, printable_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
