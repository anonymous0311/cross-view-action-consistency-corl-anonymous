"""Compute state/action normalization without decoding images."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from openpi.shared import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("assets/libero"))
    args = p.parse_args()
    files = sorted((args.dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError("No episode parquet files found")
    state, action = normalize.RunningStats(), normalize.RunningStats()
    for path in tqdm(files, desc="Episode statistics", dynamic_ncols=True):
        table = pq.read_table(path, columns=["observation.state", "action"])
        state.update(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        values = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        action.update(values.reshape(-1, values.shape[-1]))
    normalize.save(args.output_dir, {"state": state.get_statistics(), "actions": action.get_statistics()})
    print(args.output_dir / "norm_stats.json")


if __name__ == "__main__":
    main()
