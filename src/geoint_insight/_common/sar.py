"""Generic Sentinel-1 dB conversion — shared by every model that consumes SAR."""

import numpy as np

S1_SCALE_FACTOR = 10000.0
S1_VV_MEAN_DB = -12.599
S1_VH_MEAN_DB = -20.293
VH_OFFSET_DB = S1_VH_MEAN_DB - S1_VV_MEAN_DB


def convert_sar_band_to_db(band, scale_factor=S1_SCALE_FACTOR, clip_min=-50.0, clip_max=1.0):
    """Convert one SAR band to dB, auto-detecting whether it's already dB, linear
    power, or scaled linear power from its own value range (no format flag needed
    from the caller). clip_min/clip_max vary per model — see each model's own
    preprocessing module for the range it expects."""
    band = band.astype(np.float32)
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        raise ValueError("S1 band has no usable values")

    eps = np.float32(1e-8)
    minimum = float(np.min(finite))

    if minimum < -1.0:
        # Already dB: negative values are normal here (typical VV/VH means are
        # around -11 to -19 dB), unlike the linear-power branches below where a
        # value <= 0 really does mean invalid data.
        mode = "already_db"
        out = band.copy()
        invalid = ~np.isfinite(band)
    else:
        positive = finite[finite > 0]
        if positive.size == 0:
            raise ValueError("S1 band has no usable values")
        # Median (not max) decides scaling: a single bright pixel (corner
        # reflector, metal roof) can push max above 2.0 even when the band is
        # already correctly-scaled linear power with a small median.
        reference = float(np.median(positive))
        if reference <= 2.0:
            mode = "linear_power"
            out = 10.0 * np.log10(np.maximum(band, eps))
        else:
            mode = f"scaled_linear_power/{scale_factor:g}"
            out = 10.0 * np.log10(np.maximum(band / float(scale_factor), eps))
        invalid = (~np.isfinite(band)) | (band <= 0)

    out[invalid] = clip_min
    return np.clip(out, clip_min, clip_max).astype(np.float32), mode
