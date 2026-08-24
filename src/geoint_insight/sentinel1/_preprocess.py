"""Sentinel-1 conditioning: band ordering, dB conversion, speckle filtering."""

import numpy as np
import rasterio
from scipy import ndimage

from .._common.geo import make_safe_profile

# dB clip range and VV/VH training-set means the Sen1Floods11 baseline expects.
S1_SCALE_FACTOR = 10000.0
S1_VV_MEAN_DB = -12.599
S1_VH_MEAN_DB = -20.293
VH_OFFSET_DB = S1_VH_MEAN_DB - S1_VV_MEAN_DB
CLIP_MIN = -50.0
CLIP_MAX = 1.0


def convert_sar_band_to_db(band, scale_factor=S1_SCALE_FACTOR, clip_min=CLIP_MIN, clip_max=CLIP_MAX):
    """Convert one SAR band to dB, auto-detecting whether it's already dB, linear
    power, or scaled linear power from its own value range (no format flag needed
    from the caller)."""
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


def prepare_s1(input_path, output_path, speckle_filter_size=3):
    """Load a 1- or 2-band Sentinel-1 GeoTIFF, order/derive VV+VH, convert to dB,
    optionally speckle-filter (median), and write a clean 2-band dB raster ready
    for inference."""
    input_path, output_path = str(input_path), str(output_path)

    with rasterio.open(input_path) as src:
        raw = src.read().astype(np.float32)
        profile = src.profile.copy()
        descriptions = [str(v).upper() if v is not None else "" for v in src.descriptions]
        width, height = src.width, src.height

    if raw.shape[0] not in (1, 2):
        raise ValueError(f"{input_path}: expected 1 or 2 bands, got {raw.shape[0]}")

    if raw.shape[0] == 1:
        vv_db, mode = convert_sar_band_to_db(raw[0])
        vh_db = np.clip(vv_db + VH_OFFSET_DB, CLIP_MIN, CLIP_MAX).astype(np.float32)
        vh_db[vv_db <= CLIP_MIN] = CLIP_MIN
        scale_mode = mode
    else:
        medians = []
        for band in raw:
            valid = band[np.isfinite(band) & (band > 0)]
            medians.append(float(np.median(valid)) if valid.size else -np.inf)

        if "VV" in descriptions[0] and "VH" in descriptions[1]:
            vv_idx, vh_idx = 0, 1
        elif "VH" in descriptions[0] and "VV" in descriptions[1]:
            vv_idx, vh_idx = 1, 0
        elif medians[1] > medians[0]:
            vv_idx, vh_idx = 1, 0
        else:
            vv_idx, vh_idx = 0, 1

        vv_db, vv_mode = convert_sar_band_to_db(raw[vv_idx])
        vh_db, vh_mode = convert_sar_band_to_db(raw[vh_idx])
        scale_mode = f"VV={vv_mode}; VH={vh_mode}"

    if speckle_filter_size and speckle_filter_size > 1:
        vv_db = ndimage.median_filter(vv_db, size=speckle_filter_size)
        vh_db = ndimage.median_filter(vh_db, size=speckle_filter_size)

    output = np.stack([vv_db, vh_db], axis=0).astype(np.float32)
    out_profile = make_safe_profile(profile, width, height, 2, "float32", CLIP_MIN)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(output)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.update_tags(SOURCE_FILE=str(input_path), SCALE_MODE=scale_mode)

    return output_path, scale_mode
