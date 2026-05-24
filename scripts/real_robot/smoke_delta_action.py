#!/usr/bin/env python3
"""CPU smoke tests for the Realman delta-action pipeline.

Three subcommands, each runnable independently:

  delta_sanity   — load one episode, apply DeltaActions, verify joint deltas
                   are near-zero mean with small std and gripper stays absolute.
  roundtrip      — DeltaActions then AbsoluteActions returns the original
                   action chunk (within float tolerance).
  norm_stats     — load the freshly-written delta norm_stats and sanity-check
                   shapes/ranges (joint means near zero; gripper preserved).

Run with::

  JAX_PLATFORMS=cpu uv run python scripts/real_robot/smoke_delta_action.py delta_sanity
  JAX_PLATFORMS=cpu uv run python scripts/real_robot/smoke_delta_action.py roundtrip
  JAX_PLATFORMS=cpu uv run python scripts/real_robot/smoke_delta_action.py norm_stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpi.shared.normalize as _normalize
import pyarrow.parquet as pq

from openpi import transforms as _transforms

REPO_DEFAULT = Path("data/real_robot/task123_pair")
NORM_STATS_DEFAULT = Path("assets/pi05_realman_task123_cross_view_action_consistency_delta/anonymous/realman_task123_corl_delta")
ACTION_HORIZON = 10
MASK_SPEC = (7, -1)  # 7 True, 1 False — joints get delta, gripper stays absolute.


def _read_episode(repo: Path, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(repo / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
    state = (
        table["observation.state"].combine_chunks().values.to_numpy()
        .reshape(table.num_rows, 8).astype(np.float32)
    )
    action = (
        table["action"].combine_chunks().values.to_numpy()
        .reshape(table.num_rows, 8).astype(np.float32)
    )
    return state, action


def _make_chunks(action: np.ndarray, horizon: int) -> np.ndarray:
    length = action.shape[0]
    idx = np.minimum(
        np.arange(length)[:, None] + np.arange(horizon)[None, :],
        length - 1,
    )
    return action[idx]


def cmd_delta_sanity(args: argparse.Namespace) -> None:
    state, action = _read_episode(args.repo, args.episode)
    chunks = _make_chunks(action, ACTION_HORIZON)
    print(f"[delta_sanity] episode={args.episode} frames={state.shape[0]} chunks={chunks.shape}")

    mask = _transforms.make_bool_mask(*MASK_SPEC)
    print(f"[delta_sanity] mask={list(mask)} len={len(mask)}")
    delta_fn = _transforms.DeltaActions(mask)

    # The DeltaActions transform expects per-timestep (state, actions[H,D]).
    # Mimic the runtime: for each timestep t, anchor is state[t].
    deltas = []
    for t in range(state.shape[0]):
        out = delta_fn({"state": state[t].copy(), "actions": chunks[t].copy()})
        deltas.append(out["actions"])
    deltas = np.stack(deltas)  # (T, H, 8)
    print(f"[delta_sanity] deltas shape: {deltas.shape}")

    # Joint dims (0..6) should be small, mean ≈ 0.
    for d in range(7):
        col = deltas[..., d].flatten()
        abs_col = action[..., d]
        print(
            f"  d[{d}] delta: mean={col.mean():+.5f} std={col.std():.5f} "
            f"absmax={np.max(np.abs(col)):.4f}  | abs action: mean={abs_col.mean():+.3f} std={abs_col.std():.3f}"
        )
    # Gripper (dim 7) should be UNTOUCHED (matches original action).
    g_after = deltas[..., 7].flatten()
    g_before = chunks[..., 7].flatten()
    diff = np.max(np.abs(g_after - g_before))
    print(f"  d[7] gripper: max|diff|={diff:.6e}  (must be 0; if not, mask is wrong)")
    assert diff < 1e-6, "gripper must be untouched by delta transform"

    # First step of each chunk should be close to zero (because actions[t] ≈ state[t]
    # in the dataset).
    first_step = deltas[:, 0, :7]
    print(f"  first-step joint delta (a[0]-state): |mean|={np.mean(np.abs(first_step)):.5f}")

    # End of chunk should be substantially larger than start (since the chunk
    # advances 0.67 s into the future). This is the structure we WANT the model
    # to predict; under absolute targets, the model could ignore it.
    last_step = deltas[:, -1, :7]
    ratio = np.mean(np.abs(last_step)) / max(np.mean(np.abs(first_step)), 1e-8)
    print(f"  last-step joint delta (a[H-1]-state): |mean|={np.mean(np.abs(last_step)):.5f}")
    print(f"  ratio |Δ_last| / |Δ_first| = {ratio:.2f}  (delta target carries forward-motion structure if >> 1)")
    print("[delta_sanity] OK")


def cmd_roundtrip(args: argparse.Namespace) -> None:
    state, action = _read_episode(args.repo, args.episode)
    chunks = _make_chunks(action, ACTION_HORIZON)

    mask = _transforms.make_bool_mask(*MASK_SPEC)
    delta_fn = _transforms.DeltaActions(mask)
    abs_fn = _transforms.AbsoluteActions(mask)

    max_err = 0.0
    for t in range(state.shape[0]):
        d = delta_fn({"state": state[t].copy(), "actions": chunks[t].copy()})
        a = abs_fn({"state": state[t].copy(), "actions": d["actions"].copy()})
        err = np.max(np.abs(a["actions"] - chunks[t]))
        if err > max_err:
            max_err = float(err)
    print(f"[roundtrip] episode={args.episode} frames={state.shape[0]} max|round-trip err|={max_err:.3e}")
    assert max_err < 1e-5, f"roundtrip must be identity, got {max_err}"
    print("[roundtrip] OK — DeltaActions → AbsoluteActions is identity")


def cmd_norm_stats(args: argparse.Namespace) -> None:
    path = args.norm_stats / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run compute_realman_pair_norm_stats_fast_delta.py first."
        )
    norm = _normalize.load(args.norm_stats)
    a = norm["actions"]
    s = norm["state"]
    print(f"[norm_stats] loaded: {path}")
    print(f"  action.mean ({np.asarray(a.mean).shape}): {np.asarray(a.mean).round(4).tolist()}")
    print(f"  action.std  : {np.asarray(a.std).round(4).tolist()}")
    if a.q01 is not None:
        print(f"  action.q01  : {np.asarray(a.q01).round(4).tolist()}")
        print(f"  action.q99  : {np.asarray(a.q99).round(4).tolist()}")
    print(f"  state.mean  : {np.asarray(s.mean).round(4).tolist()}")
    print(f"  state.std   : {np.asarray(s.std).round(4).tolist()}")

    a_mean = np.asarray(a.mean)
    a_std = np.asarray(a.std)
    s_std = np.asarray(s.std)
    # Joint action means should be very close to zero (deltas).
    joint_mean_abs = float(np.max(np.abs(a_mean[:7])))
    if joint_mean_abs > 0.1:
        print(f"  WARNING: joint action mean abs={joint_mean_abs:.3f} > 0.1 — delta wasn't applied? (or chunk-mean is biased)")
    else:
        print(f"  joint action |mean| max = {joint_mean_abs:.4f}  ✓ near zero")

    # Joint action std must be << absolute-action std → deltas have less spread
    # than absolute targets. We sanity-check joint std < joint state std.
    print(
        f"  joint action std vs state std (per joint):\n    {(a_std[:7] / np.maximum(s_std[:7], 1e-9)).round(3).tolist()}"
        f"\n    (should be << 1 if delta computation was applied)"
    )

    # Gripper should be untouched: action[7] dist should match state[7] dist
    # within sampling noise (since action[t] ≈ state[t+1] for gripper too in
    # this dataset).
    print(f"  gripper action mean={a_mean[7]:.4f} std={a_std[7]:.4f}  (state mean={s.mean[7]:.4f} std={s.std[7]:.4f}; expect similar)")
    print("[norm_stats] OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("delta_sanity")
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    p.add_argument("--episode", type=int, default=0)
    p.set_defaults(func=cmd_delta_sanity)

    p = sub.add_parser("roundtrip")
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    p.add_argument("--episode", type=int, default=0)
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("norm_stats")
    p.add_argument("--norm-stats", type=Path, default=NORM_STATS_DEFAULT)
    p.set_defaults(func=cmd_norm_stats)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
