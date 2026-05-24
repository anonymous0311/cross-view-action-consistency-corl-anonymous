"""Smoke-test cross-view paired LeRobot dataloader output shapes.

This does not start training. It loads one batch and verifies the cross-view pair axis,
wrist exclusion, token pair replication, and padded action/state dimensions.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_cross_view_action_consistency")
    parser.add_argument("--batch-size", type=int, default=max(1, jax.device_count()))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size, num_workers=args.num_workers)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True, num_batches=1, shuffle=False)
    obs, actions = next(iter(loader))

    expected_actions = (args.batch_size, 2, cfg.model.action_horizon, cfg.model.action_dim)
    expected_state = (args.batch_size, 2, cfg.model.action_dim)
    if actions.shape != expected_actions:
        raise AssertionError(f"actions shape {actions.shape} != {expected_actions}")
    if obs.state.shape != expected_state:
        raise AssertionError(f"state shape {obs.state.shape} != {expected_state}")
    if obs.images["base_0_rgb"].shape[:2] != (args.batch_size, 2):
        raise AssertionError(f"base image shape {obs.images['base_0_rgb'].shape} missing pair axis")
    if obs.tokenized_prompt is None or obs.tokenized_prompt.shape[:2] != (args.batch_size, 2):
        raise AssertionError(f"tokenized_prompt shape {None if obs.tokenized_prompt is None else obs.tokenized_prompt.shape}")
    unexpected_wrist_keys = {"left_wrist_0_rgb", "right_wrist_0_rgb"} & set(obs.images)
    if unexpected_wrist_keys:
        raise AssertionError(f"wrist images should be omitted for cross-view action consistency: {sorted(unexpected_wrist_keys)}")

    print(f"config={args.config_name}")
    print(f"actions={actions.shape}")
    print(f"state={obs.state.shape}")
    print(f"base_image={obs.images['base_0_rgb'].shape}")
    print(f"tokenized_prompt={obs.tokenized_prompt.shape}")
    print("wrist_images=omitted")


if __name__ == "__main__":
    main()
