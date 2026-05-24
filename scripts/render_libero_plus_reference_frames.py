"""Render LIBERO-plus reference first-frames for every camera-view target.

For each row in `assets/m7_boundary/camera_view_targets.parquet`, construct a
fresh `OffScreenRenderEnv` at the encoded (h, v, s, er, ev, init_state)
configuration, render the agentview image at reset, and save it as PNG.

Multi-process: each worker owns one env at a time; envs are short-lived
(constructor parses BDDL + builds scene; cheap to recreate).

Usage:
  .venv-libero-plus/bin/python scripts/render_libero_plus_reference_frames.py [--workers 32] [--limit N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LIBERO_PLUS_ROOT = REPO / "LIBERO-plus"
BENCHMARK_ROOT = LIBERO_PLUS_ROOT / "libero" / "libero"
TARGETS_PATH = REPO / "assets" / "m7_boundary" / "camera_view_targets.parquet"
OUT_DIR = REPO / "assets" / "m7_boundary" / "reference_frames"
LOG_DIR = REPO / "assets" / "m7_boundary" / "render_logs"


def reference_filename(row: pd.Series) -> str:
    return (
        f"{row['suite']}__{row['base_task']}__"
        f"h{int(row['horizon'])}_v{int(row['vertical'])}_s{int(row['scale'])}_"
        f"er{int(row['end_rot'])}_ev{int(row['end_vert'])}__init{int(row['init_state'])}.png"
    )


def init_worker() -> None:
    """One-time worker setup: configure LIBERO_CONFIG_PATH + sys.path."""
    sys.path.insert(0, str(LIBERO_PLUS_ROOT))
    sys.path.insert(0, str(REPO))
    config_dir = Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        config_text = (
            f"benchmark_root: {BENCHMARK_ROOT.resolve()}\n"
            f"bddl_files: {(BENCHMARK_ROOT / 'bddl_files').resolve()}\n"
            f"init_states: {(BENCHMARK_ROOT / 'init_files').resolve()}\n"
            f"datasets: {(LIBERO_PLUS_ROOT / 'datasets').resolve()}\n"
            f"assets: {(BENCHMARK_ROOT / 'assets').resolve()}\n"
        )
        config_path.write_text(config_text)


def _render_one(args: tuple[dict, Path, int]) -> tuple[str, bool, str]:
    """Render a single reference frame.

    Returns (filename, ok, error_msg).
    """
    row_dict, out_path, seed = args
    fname = out_path.name
    if out_path.exists():
        return (fname, True, "skipped (exists)")

    try:
        from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415
        import imageio.v2 as imageio  # noqa: PLC0415

        suite = row_dict["suite"]
        base = row_dict["base_task"]
        h = int(row_dict["horizon"])
        v = int(row_dict["vertical"])
        s = int(row_dict["scale"])
        er = int(row_dict["end_rot"])
        ev = int(row_dict["end_vert"])
        init = int(row_dict["init_state"])

        synthetic = (
            BENCHMARK_ROOT
            / "bddl_files"
            / suite
            / f"{base}_view_{h}_{v}_{s}_{er}_{ev}_initstate_{init}"
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(synthetic),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(seed)
        try:
            obs = env.reset()
            img = obs["agentview_image"]
            # robosuite's offscreen image is upside-down relative to display convention
            img_rgb = np.flipud(img).astype(np.uint8)
            imageio.imwrite(str(out_path), img_rgb)
        finally:
            env.close()

        return (fname, True, "")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return (fname, False, msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only render first N targets (for smoke test).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = pd.read_parquet(args.targets)
    print(f"Loaded {len(df)} targets from {args.targets}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Limiting to first {len(df)} targets")

    work: list[tuple[dict, Path, int]] = []
    for _, row in df.iterrows():
        out_path = args.out_dir / reference_filename(row)
        work.append((row.to_dict(), out_path, args.seed))

    print(f"Rendering {len(work)} frames with {args.workers} workers ...")
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    n_err = 0
    err_log: list[str] = []

    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        for i, (fname, ok, msg) in enumerate(pool.imap_unordered(_render_one, work, chunksize=1), 1):
            if ok:
                if "skipped" in msg:
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_err += 1
                err_log.append(f"{fname}\n{msg}\n---\n")
                print(f"[ERR] {fname}: {msg.splitlines()[0]}")
            if i % 50 == 0 or i == len(work):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(work) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(work)}]  ok={n_ok} skip={n_skip} err={n_err}  rate={rate:.1f}/s  eta={eta:.0f}s")

    if err_log:
        log_path = LOG_DIR / "render_errors.log"
        log_path.write_text("\n".join(err_log))
        print(f"Wrote {len(err_log)} error reports to {log_path}")

    print(f"\nDONE: ok={n_ok} skip={n_skip} err={n_err}  total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
