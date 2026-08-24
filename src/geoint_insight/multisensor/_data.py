"""Bundled sample data — lets a fresh install produce a result immediately,
without the user needing to source their own S1+S2 scene pair first."""

from pathlib import Path

_SAMPLE_DIR = Path(__file__).resolve().parent / "data"
_SAMPLE_S1_PATH = _SAMPLE_DIR / "sample_s1.tif"
_SAMPLE_S2_PATH = _SAMPLE_DIR / "sample_s2.tif"


def sample_s1_path():
    """Path to a small bundled real Sentinel-1 VV/VH GeoTIFF (500x500px, UTM 47N)."""
    if not _SAMPLE_S1_PATH.exists():
        raise FileNotFoundError(f"Bundled sample S1 scene missing: {_SAMPLE_S1_PATH}")
    return _SAMPLE_S1_PATH


def sample_s2_path():
    """Path to a small bundled real Sentinel-2 13-band GeoTIFF, on the same grid
    as sample_s1_path()."""
    if not _SAMPLE_S2_PATH.exists():
        raise FileNotFoundError(f"Bundled sample S2 scene missing: {_SAMPLE_S2_PATH}")
    return _SAMPLE_S2_PATH
