"""Bundled sample data — lets a fresh install run the rice pipeline end to end
without sourcing a real multitemporal Sentinel-2 stack first.

Three real acquisition dates are bundled here (2026-03-13, 03-26, 05-05) —
not the full 6-date stack the checkpoint was fine-tuned on (see
training_config.json's "timestamps"), but enough for the model's temporal
pooling to work meaningfully: unlike a single timestamp (which this package
tested and found detects ~0 rice even over visibly green paddy fields), 3
dates produce real, spatially coherent rice detections.
"""

from pathlib import Path

_SAMPLE_DIR = Path(__file__).resolve().parent / "data"
_SAMPLE_STACK_PATHS = [
    _SAMPLE_DIR / "S2_20260313_STACK_10M.tif",
    _SAMPLE_DIR / "S2_20260326_STACK_10M.tif",
    _SAMPLE_DIR / "S2_20260505_STACK_10M.tif",
]


def sample_stack_paths():
    """Ordered list of bundled sample Sentinel-2 stack GeoTIFF paths (small
    real crop, Prathumthanee, Thailand, 2026-03-13/03-26/05-05)."""
    for path in _SAMPLE_STACK_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Bundled sample rice stack missing: {path}")
    return list(_SAMPLE_STACK_PATHS)
