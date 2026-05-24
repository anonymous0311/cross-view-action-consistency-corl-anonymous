#!/usr/bin/env python3
"""Smoke test Phase 0B pair-aware augmentation on CPU."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

# This smoke is intentionally CPU-only so it cannot interfere with active GPU jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import v4_cv
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from PIL import Image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def _save_pair_grid(path: Path, images: np.ndarray, *, max_pairs: int) -> None:
    images = np.asarray(images[:max_pairs])
    if images.ndim != 5 or images.shape[1] != 2:
        raise ValueError(f"Expected images [B, 2, H, W, C], got {images.shape}")
    rows, _, height, width, channels = images.shape
    if channels != 3:
        raise ValueError(f"Expected RGB images, got channel count {channels}")
    canvas = np.zeros((rows * height, 2 * width, 3), dtype=np.uint8)
    for row in range(rows):
        canvas[row * height : (row + 1) * height, :width] = _to_uint8(images[row, 0])
        canvas[row * height : (row + 1) * height, width:] = _to_uint8(images[row, 1])
    Image.fromarray(canvas).save(path)


def _shared_spatial_debug(rng: jax.Array, *, batch_size: int, height: int, width: int) -> dict:
    spatial_rng, _ = jax.random.split(rng)
    row_rng = jax.random.split(spatial_rng, batch_size)[0]
    geometric_chain_rng = jax.random.split(row_rng, 1)[0]
    crop_key, _, rotate_key = jax.random.split(geometric_chain_rng, 3)
    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)
    limit_y = (height - crop_height) / 2
    limit_x = (width - crop_width) / 2
    crop_center_yx = jax.random.uniform(
        crop_key,
        [2],
        minval=jnp.asarray([-limit_y, -limit_x]),
        maxval=jnp.asarray([limit_y, limit_x]),
    )
    rotate_rad = jax.random.uniform(rotate_key, minval=math.radians(-5), maxval=math.radians(5))
    return {
        "first_pair_crop_center_offset_yx": [float(crop_center_yx[0]), float(crop_center_yx[1])],
        "first_pair_rotate_degrees": float(rotate_rad * 180.0 / math.pi),
    }


def _shared_photometric_debug(rng: jax.Array, *, batch_size: int) -> dict:
    _, photometric_rng = jax.random.split(rng)
    row_rng = jax.random.split(photometric_rng, batch_size)[0]
    colorspace_chain_rng = jax.random.split(row_rng, 1)[0]
    color_jitter_rng = jax.random.split(colorspace_chain_rng, 1)[0]
    brightness_key, contrast_key, hue_key, saturation_key, _ = jax.random.split(color_jitter_rng, 5)
    return {
        "first_pair_brightness_delta": float(jax.random.uniform(brightness_key, minval=-0.3, maxval=0.3)),
        "first_pair_contrast_delta": float(jax.random.uniform(contrast_key, minval=-0.4, maxval=0.4)),
        "first_pair_hue_delta": float(jax.random.uniform(hue_key, minval=-0.1, maxval=0.1)),
        "first_pair_saturation_delta": float(jax.random.uniform(saturation_key, minval=-0.5, maxval=0.5)),
        "note": "augmax samples these inside ColorJitter; values shown for the first shared pair RNG.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_v4_pair_fm_only")
    parser.add_argument("--pair-spatial-aug-mode", default="none", choices=["current", "none", "shared"])
    parser.add_argument(
        "--pair-photometric-aug-mode",
        default="independent",
        choices=["current", "none", "shared", "independent"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs-to-save", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/v4_plus/pair_aug_smoke")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=0)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True, num_batches=1, shuffle=False)
    observation, actions = next(iter(loader))

    image_key = "base_0_rgb"
    if image_key not in observation.images:
        raise AssertionError(f"Missing {image_key}; image keys are {tuple(observation.images)}")
    before = np.asarray(jax.device_get(observation.images[image_key]))
    if before.ndim != 5 or before.shape[1] != 2:
        raise AssertionError(f"Pair axis is missing before augmentation: {before.shape}")

    wrist_keys = [key for key in observation.images if "wrist" in key]
    wrist_ok = not wrist_keys or all(not bool(np.asarray(observation.image_masks[key]).any()) for key in wrist_keys)
    if not wrist_ok:
        raise AssertionError(f"Wrist image unexpectedly active: {wrist_keys}")

    rng = jax.random.key(args.seed)
    after_obs = v4_cv.preprocess_pair_observation(
        rng,
        observation,
        train=True,
        image_keys=(image_key,),
        pair_spatial_aug_mode=args.pair_spatial_aug_mode,
        pair_photometric_aug_mode=args.pair_photometric_aug_mode,
    )
    after_flat = np.asarray(jax.device_get(after_obs.images[image_key]))
    after = after_flat.reshape(before.shape[0], 2, *after_flat.shape[1:])

    _save_pair_grid(out_dir / "before_pairs.png", before, max_pairs=args.max_pairs_to_save)
    _save_pair_grid(out_dir / "after_pairs.png", after, max_pairs=args.max_pairs_to_save)

    shared_params = {}
    if args.pair_spatial_aug_mode == "shared":
        shared_params["spatial"] = _shared_spatial_debug(
            rng,
            batch_size=before.shape[0],
            height=before.shape[2],
            width=before.shape[3],
        )
    if args.pair_photometric_aug_mode == "shared":
        shared_params["photometric"] = _shared_photometric_debug(rng, batch_size=before.shape[0])

    report = {
        "config_name": args.config_name,
        "pair_spatial_aug_mode": args.pair_spatial_aug_mode,
        "pair_photometric_aug_mode": args.pair_photometric_aug_mode,
        "cpu_only": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_CUDA_VISIBLE_DEVICES": os.environ.get("JAX_CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "jax_default_backend": jax.default_backend(),
        },
        "batch": {
            "before_base_shape": list(before.shape),
            "after_flat_base_shape": list(after_flat.shape),
            "actions_shape": list(np.asarray(jax.device_get(actions)).shape),
            "pair_axis_present_before_aug": before.ndim == 5 and before.shape[1] == 2,
            "wrist_keys": wrist_keys,
            "wrist_inactive_or_absent": wrist_ok,
        },
        "outputs": {
            "before_grid": str(out_dir / "before_pairs.png"),
            "after_grid": str(out_dir / "after_pairs.png"),
        },
        "shared_params": shared_params,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Pair Augmentation Smoke",
        "",
        f"- config: `{args.config_name}`",
        f"- pair_spatial_aug_mode: `{args.pair_spatial_aug_mode}`",
        f"- pair_photometric_aug_mode: `{args.pair_photometric_aug_mode}`",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- before shape: `{list(before.shape)}`",
        f"- after flat shape: `{list(after_flat.shape)}`",
        f"- pair axis before augmentation: `{before.ndim == 5 and before.shape[1] == 2}`",
        f"- wrist inactive or absent: `{wrist_ok}`",
        f"- before grid: `{out_dir / 'before_pairs.png'}`",
        f"- after grid: `{out_dir / 'after_pairs.png'}`",
    ]
    if shared_params:
        lines += ["", "## Shared Parameters", "", "```json", json.dumps(shared_params, indent=2), "```"]
    (Path("results/v4_plus/pair_aug_smoke.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
