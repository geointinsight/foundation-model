"""Bundled sample data — lets a fresh install produce a result immediately,
without the user needing to source their own Sentinel-1 scene first."""

from pathlib import Path

_SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_s1.tif"


def sample_path():
    """Path to a small bundled real Sentinel-1 VV/VH GeoTIFF (500x500px, UTM 47N)."""
    if not _SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Bundled sample scene missing: {_SAMPLE_PATH}")
    return _SAMPLE_PATH
