"""Sentinel-1 conditioning: band ordering, dB conversion, speckle filtering."""

import numpy as np
import rasterio
from scipy import ndimage

from .._common.geo import make_safe_profile
from .._common.sar import S1_SCALE_FACTOR, VH_OFFSET_DB, convert_sar_band_to_db

# dB clip range this model (Sen1Floods11 baseline) expects — different from
# TerraMind's [-35, 5], see ../multisensor/_preprocess.py.
CLIP_MIN = -50.0
CLIP_MAX = 1.0


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
        vv_db, mode = convert_sar_band_to_db(raw[0], S1_SCALE_FACTOR, CLIP_MIN, CLIP_MAX)
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

        vv_db, vv_mode = convert_sar_band_to_db(raw[vv_idx], S1_SCALE_FACTOR, CLIP_MIN, CLIP_MAX)
        vh_db, vh_mode = convert_sar_band_to_db(raw[vh_idx], S1_SCALE_FACTOR, CLIP_MIN, CLIP_MAX)
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
