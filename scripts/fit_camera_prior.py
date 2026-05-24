"""Fit empirical camera-bin prior on a given sparse subset.

Reads the K=8 camera-bin parquet file and the kcenter_30 (or other named)
sparse subset, counts bin frequencies for the selected episodes, and saves
a normalised float32 prior vector to a .npy file.

Usage:
  python scripts/fit_camera_prior.py \\
      --subset kcenter_30 \\
      --output assets/camera_prior_kcenter30.npy

  # Optionally override the camera-bins parquet path:
  python scripts/fit_camera_prior.py \\
      --subset kcenter_30 \\
      --camera-bins assets/camera_bins_k8.parquet \\
      --output assets/camera_prior_kcenter30.npy

Output: 1D float32 array of shape [K] summing to 1.

Expected values for kcenter_30:
  bin 0: 0.106  bin 4: 0.190
  bin 1: 0.082  bin 5: 0.111
  bin 2: 0.226  bin 6: 0.074
  bin 3: 0.077  bin 7: 0.134
  sum = 1.000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "openpi" / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subset",
        required=True,
        choices=["kcenter_30", "random_30"],
        help="Which sparse subset to compute the prior over.",
    )
    parser.add_argument(
        "--camera-bins",
        default=None,
        help="Path to camera_bins_k8.parquet. Defaults to the path in sparse_subsets.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy path for the prior vector.",
    )
    args = parser.parse_args()

    # Load subset spec
    from canonical.data.sparse_subsets import (
        KCENTER_PRIMARY_30,
        RANDOM_CONTROL_30,
        CAMERA_BINS_K8_PATH,
    )
    if args.subset == "kcenter_30":
        spec = KCENTER_PRIMARY_30
    elif args.subset == "random_30":
        spec = RANDOM_CONTROL_30
    else:
        raise ValueError(f"Unknown subset: {args.subset!r}")

    camera_bins_path = args.camera_bins or CAMERA_BINS_K8_PATH

    print(f"Subset: {spec.name} ({spec.num_episodes} episodes)")
    print(f"Camera bins centroid path: {camera_bins_path}")

    # Build episode→bin_id lookup using the existing helper (applies scaler + KMeans).
    # camera_bins_k8.parquet stores centroids (8 rows), not per-episode assignments,
    # so we must go through build_episode_camera_bin_lookup.
    from openpi.shared.camera_bins import build_episode_camera_bin_lookup, DEFAULT_CAMERA_EPISODE_PARAMS_PATH
    episode_to_bin = build_episode_camera_bin_lookup(
        episode_params_path=DEFAULT_CAMERA_EPISODE_PARAMS_PATH,
        camera_bins_path=camera_bins_path,
    )
    print(f"Total episodes in lookup: {len(episode_to_bin)}")

    # Filter to subset and count per-bin frequencies
    subset_ep = set(spec.episode_indices)
    selected_bins = [episode_to_bin[ep] for ep in subset_ep if ep in episode_to_bin]
    matched = len(selected_bins)
    print(f"Episodes matched to subset: {matched} / {spec.num_episodes}")

    if matched == 0:
        raise ValueError("No subset episodes found in the camera bin lookup.")

    K = max(episode_to_bin.values()) + 1
    prior = np.zeros(K, dtype=np.float32)
    for b in selected_bins:
        prior[int(b)] += 1.0

    total = prior.sum()
    if total == 0:
        raise ValueError("All bin counts are zero — something is wrong with the data.")
    prior /= total

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), prior)

    print(f"\nSaved prior of shape {prior.shape}, sum = {prior.sum():.6f}")
    print("Per-bin frequencies:")
    for i, p in enumerate(prior):
        n = int(round(p * matched))
        print(f"  bin {i}: {p:.3f}  ({n}/{matched})")

    # Sanity checks
    assert prior.shape == (K,), f"Expected shape ({K},), got {prior.shape}"
    assert abs(prior.sum() - 1.0) < 1e-5, f"Prior does not sum to 1: sum={prior.sum()}"
    assert (prior >= 0).all(), "Prior contains negative values"
    print(f"\nSanity checks passed. Prior saved to: {output_path}")


if __name__ == "__main__":
    main()
