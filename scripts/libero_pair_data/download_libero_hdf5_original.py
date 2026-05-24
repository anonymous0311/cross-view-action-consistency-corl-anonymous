"""Download the original LIBERO HDF5 datasets from utexas.box.com.

Replicates the official LIBERO download_utils.py logic (same URLs, same ZIP
extraction) with project-standard tqdm (dynamic_ncols=True, leave=True) and
no dependency on the libero package.

Expected output layout:
  <download_dir>/
    libero_spatial/   — 10 .hdf5 task files
    libero_object/    — 10 .hdf5 task files
    libero_goal/      — 10 .hdf5 task files
    libero_10/        — 10 .hdf5 task files  (from libero_100.zip)
    libero_90/        — 90 .hdf5 task files  (from libero_100.zip)

Usage:
  python scripts/libero_pair_data/download_libero_hdf5_original.py \
    --download-dir data/libero_hdf5_original \
    [--datasets all|libero_spatial|libero_object|libero_goal|libero_100]
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Official LIBERO dataset URLs (from download_utils.py in LIBERO repo) ──────
DATASET_LINKS: dict[str, str] = {
    "libero_object":  "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal":    "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100":     "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

# libero_100.zip extracts into libero_10/ (10 files) and libero_90/ (90 files)
EXPECTED_COUNTS: dict[str, int] = {
    "libero_object":  10,
    "libero_goal":    10,
    "libero_spatial": 10,
    "libero_10":      10,
    "libero_90":      90,
}

CHUNK = 1 << 20  # 1 MiB read chunks


def download_and_extract(url: str, download_dir: Path, dataset_name: str) -> Path:
    """Download one ZIP from Box.com with requests (handles Box.com redirects)
    and extract it. Returns the download_dir path."""
    zip_path = download_dir / f"{dataset_name}.zip"

    if zip_path.exists():
        print(f"  ZIP already exists at {zip_path} — skipping download.", flush=True)
    else:
        # Box.com uses multi-step redirects that urllib cannot follow but requests can.
        with requests.get(url, stream=True, allow_redirects=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            with (
                open(zip_path, "wb") as fout,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    dynamic_ncols=True,
                    leave=True,
                    desc=f"{dataset_name}.zip",
                ) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    fout.write(chunk)
                    pbar.update(len(chunk))

    print(f"  Extracting {zip_path.name} …", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        with tqdm(
            total=len(members),
            unit="file",
            dynamic_ncols=True,
            leave=False,
            desc=f"  {dataset_name} extract",
        ) as pbar:
            for member in members:
                zf.extract(member, path=str(download_dir))
                pbar.update(1)

    print(f"  Removing {zip_path.name} …")
    zip_path.unlink()
    return download_dir


def check_dataset(download_dir: Path) -> bool:
    """Verify expected HDF5 counts for each suite (mirrors check_libero_dataset)."""
    print("\n── Dataset integrity check ─────────────────────────────────────────")
    all_ok = True
    for suite, expected in EXPECTED_COUNTS.items():
        suite_dir = download_dir / suite
        if not suite_dir.exists():
            print(f"  [ ] {suite:<20} — directory not found")
            all_ok = False
            continue
        h5_files = sorted(suite_dir.glob("*.hdf5"))
        found = len(h5_files)
        if found == expected:
            print(f"  [✓] {suite:<20} — {found}/{expected} HDF5 files")
        else:
            print(f"  [✗] {suite:<20} — {found}/{expected} HDF5 files (incomplete)")
            all_ok = False
    print("────────────────────────────────────────────────────────────────────")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Download original LIBERO HDF5 datasets.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/libero_hdf5_original"),
        help="Destination directory (default: data/libero_hdf5_original)",
    )
    parser.add_argument(
        "--datasets",
        choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_100"],
        default="all",
        help="Which dataset(s) to download.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the post-download integrity check.",
    )
    args = parser.parse_args()

    download_dir: Path = args.download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    to_download = (
        list(DATASET_LINKS.keys()) if args.datasets == "all" else [args.datasets]
    )

    print(f"Download dir : {download_dir}")
    print(f"Datasets     : {to_download}")
    print()

    for dataset_name in to_download:
        url = DATASET_LINKS[dataset_name]
        print(f"─── {dataset_name} ───────────────────────────────────────────────")
        try:
            download_and_extract(url, download_dir, dataset_name)
            print(f"  Done: {dataset_name}\n")
        except Exception as exc:
            print(f"  ERROR downloading {dataset_name}: {exc}", file=sys.stderr)
            sys.exit(1)

    if not args.skip_check:
        ok = check_dataset(download_dir)
        if ok:
            print("\nAll datasets downloaded and verified successfully.")
            print(f"\nRe-run audit with:")
            print(f"  .venv/bin/python scripts/libero_pair_data/audit_libero_hdf5_original.py \\")
            print(f"    --libero-root {download_dir} \\")
            print(f"    --suite-dirs libero_spatial libero_object libero_goal libero_90 libero_10 \\")
            print(f"    --output-dir results/libero_pair_audit")
        else:
            print("\nSome datasets are incomplete. Check errors above.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
