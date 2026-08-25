"""Bundled sample data — lets a fresh install run the rice pipeline end to end
without sourcing a real multitemporal Sentinel-2 stack first.

Only one real acquisition date (2026-03-13) is bundled here, not the full
multitemporal stack the checkpoint was actually fine-tuned on (6 dates
spanning dry season through early wet season — see training_config.json's
"timestamps"). Tested against the real checkpoint: a single timestamp is
enough to exercise every step of the pipeline (band mapping, tiling,
inference, mask export, preview), but the model itself needs multiple
timestamps to detect rice reliably — expect ~0 rice area from --sample even
over visibly green paddy fields. Use it to confirm the tool runs, not to
judge detection quality; for that, run against a real multitemporal stack
(see predict_stacks_folder / README).
"""

from pathlib import Path

_SAMPLE_DIR = Path(__file__).resolve().parent / "data"
_SAMPLE_STACK_PATHS = [_SAMPLE_DIR / "S2_20260313_STACK_10M.tif"]


def sample_stack_paths():
    """Ordered list of bundled sample Sentinel-2 stack GeoTIFF path(s) (small
    real crop, Prathumthanee, Thailand, 2026-03-13). Single-timestamp — see
    module docstring for why that means near-zero detected rice area."""
    for path in _SAMPLE_STACK_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Bundled sample rice stack missing: {path}")
    return list(_SAMPLE_STACK_PATHS)
