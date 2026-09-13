"""Fetch the pinned official LIBERO-Plus asset archive (approximately 6.4 GB)."""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path("third_party/libero-plus/libero/libero"))
    args = p.parse_args()
    marker = args.destination / "assets/.afcv_assets_revision"
    if marker.exists() and marker.read_text().strip() == REVISION:
        print("Pinned simulator assets are ready")
        return
    if (args.destination / "assets").is_symlink():
        raise ValueError("Replace the asset symlink with a directory before installing official assets")
    archive = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", revision=REVISION)
    with zipfile.ZipFile(archive) as zf:
        for member in tqdm(zf.infolist(), desc="Simulator assets", dynamic_ncols=True):
            target = (args.destination / member.filename).resolve()
            if not target.is_relative_to(args.destination.resolve()):
                raise ValueError("Archive member outside destination")
            zf.extract(member, args.destination)
    if not (args.destination / "assets/scenes").is_dir():
        raise ValueError("Archive must contain assets/scenes")
    marker.write_text(REVISION + "\n")


if __name__ == "__main__":
    main()
