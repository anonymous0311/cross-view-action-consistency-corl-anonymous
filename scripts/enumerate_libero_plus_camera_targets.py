"""Enumerate every (suite, base_task, h, v, s, er, ev, init_state) combo that
LIBERO-plus's `camera_view` perturbation enumerates.

Inputs:
- data/libero_plus_data_4suite/task_classification.json

Output:
- assets/m7_boundary/camera_view_targets.parquet

Usage:
  uv run python scripts/enumerate_libero_plus_camera_targets.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TASK_CLASSIFICATION = Path(
    "data/libero_plus_data_4suite/task_classification.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "m7_boundary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A name like:
#   pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_2_4_initstate_0
# Parses to:
#   base = pick_up_the_alphabet_soup_and_place_it_in_the_basket
#   h, v, s, er, ev = 0, 0, 100, 2, 4
#   init_state = 0
NAME_RE = re.compile(
    r"^(?P<base>.+?)_view_"
    r"(?P<h>-?\d+)_(?P<v>-?\d+)_(?P<s>-?\d+)_(?P<er>-?\d+)_(?P<ev>-?\d+)"
    r"_initstate_(?P<init>\d+)$"
)


def parse_name(name: str) -> dict | None:
    m = NAME_RE.match(name)
    if not m:
        return None
    return {
        "base_task": m.group("base"),
        "horizon": int(m.group("h")),
        "vertical": int(m.group("v")),
        "scale": int(m.group("s")),
        "end_rot": int(m.group("er")),
        "end_vert": int(m.group("ev")),
        "init_state": int(m.group("init")),
    }


def main() -> None:
    with TASK_CLASSIFICATION.open("r") as f:
        cls = json.load(f)

    rows: list[dict] = []
    skipped_non_camera = 0
    skipped_unparseable = 0
    for suite, items in cls.items():
        for it in items:
            if it.get("category") != "Camera Viewpoints":
                skipped_non_camera += 1
                continue
            parsed = parse_name(it["name"])
            if parsed is None:
                skipped_unparseable += 1
                continue
            row = {
                "suite": suite,
                "task_id": it["id"],
                "task_name": it["name"],
                "difficulty_level": it.get("difficulty_level"),
                **parsed,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Camera-viewpoint task entries: {len(df)}")
    print(f"Skipped non-camera-viewpoint entries: {skipped_non_camera}")
    print(f"Skipped unparseable entries: {skipped_unparseable}")
    print()
    print("Per-suite counts:")
    print(df.groupby("suite").size())
    print()
    print("Unique 5-tuples (h, v, s, er, ev) per suite:")
    print(
        df.groupby("suite").apply(
            lambda g: g[["horizon", "vertical", "scale", "end_rot", "end_vert"]].drop_duplicates().shape[0],
            include_groups=False,
        )
    )
    print()
    print("Unique 5-tuples across all suites:")
    print(df[["horizon", "vertical", "scale", "end_rot", "end_vert"]].drop_duplicates().shape[0])
    print()
    print("Distinct values per axis:")
    for col in ["horizon", "vertical", "scale", "end_rot", "end_vert", "init_state"]:
        vals = sorted(df[col].unique().tolist())
        print(f"  {col} ({len(vals)} unique): {vals[:10]}{'...' if len(vals) > 10 else ''} max={max(vals)} min={min(vals)}")
    print()
    print("Distinct base tasks:", df["base_task"].nunique())

    out_path = OUTPUT_DIR / "camera_view_targets.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
