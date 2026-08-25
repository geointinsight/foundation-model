"""Download model checkpoints (not committed to the repo — see .gitignore).

    geoint-insight setup

Fetches checkpoints-geoint-insight.zip from GEOINT Insight's shared storage
and extracts it into ./checkpoints/. Kept separate from `pip install` itself
since pip has no reliable way to run a large post-install download step.
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

CHECKPOINTS_ZIP_FILE_ID = "1W9kraLERsSGGnfMFp6djG7dA7OVAZtLs"

# Canonical checkpoint filenames this package's models look for.
SAR_CHECKPOINT_NAME = "sar_geoint_insight.cp"
MULTISENSOR_CHECKPOINT_NAME = "multisensor_geoint_insight.ckpt"
EXPECTED_CHECKPOINTS = [SAR_CHECKPOINT_NAME, MULTISENSOR_CHECKPOINT_NAME]


def download_checkpoints(dest_dir="checkpoints", force=False):
    """Download and extract the shared checkpoints archive into dest_dir.
    Skips re-downloading if all expected checkpoints are already present,
    unless force=True.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Downloading checkpoints requires the 'gdown' package: pip install gdown "
            "(already included if you installed this package normally)."
        ) from exc

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        missing = [name for name in EXPECTED_CHECKPOINTS if not (dest_dir / name).exists()]
        if not missing:
            print(f"All checkpoints already present in {dest_dir}/ — nothing to do (use --force to re-download).")
            return dest_dir
        print(f"Missing: {missing}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "checkpoints-geoint-insight.zip"
        print("Downloading checkpoints-geoint-insight.zip ...")
        gdown.download(id=CHECKPOINTS_ZIP_FILE_ID, output=str(zip_path), quiet=False)

        if not zip_path.exists():
            raise RuntimeError("Download failed — no file was written.")

        print(f"Extracting into {dest_dir}/ ...")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                # Skip macOS metadata cruft (.DS_Store, AppleDouble ._* resource
                # forks) that ends up in zips created on a Mac — not real content.
                if name == ".DS_Store" or name.startswith("._") or "__MACOSX" in member.filename:
                    continue
                # Flatten any subfolder structure inside the zip — we only care
                # about the checkpoint files themselves, not where the archive
                # happened to nest them.
                target = dest_dir / name
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  {target.name}")

    found = [name for name in EXPECTED_CHECKPOINTS if (dest_dir / name).exists()]
    missing = [name for name in EXPECTED_CHECKPOINTS if name not in found]
    if missing:
        print(
            f"WARNING: expected checkpoint(s) not found after extraction: {missing}. "
            f"Check the archive contents — files present: {sorted(p.name for p in dest_dir.iterdir())}"
        )
    else:
        print("All expected checkpoints present.")

    return dest_dir


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Download GEOINT Insight model checkpoints")
    parser.add_argument("--dest", default="checkpoints", help="Where to place checkpoint files")
    parser.add_argument("--force", action="store_true", help="Re-download even if checkpoints already exist")
    args = parser.parse_args(argv)

    download_checkpoints(args.dest, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
