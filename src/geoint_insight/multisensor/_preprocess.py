"""Sentinel-1 + Sentinel-2 conditioning for TerraMind: band ordering, dB
conversion, and reprojecting S2 onto the S1 grid."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .._common.geo import make_safe_profile
from .._common.sar import S1_SCALE_FACTOR, VH_OFFSET_DB, convert_sar_band_to_db
from ._model import S2_MEAN

# dB clip range TerraMind expects — different from the Sen1Floods11 baseline's
# [-50, 1], see ../sar/_preprocess.py.
CLIP_MIN = -35.0
CLIP_MAX = 5.0

# If S2 is 3-band, assume this band order.
S2_RGB_ORDER = "RGB"

# If S2 has no CRS metadata, assume it shares S1's (common for some export
# pipelines that drop georeferencing on a subset of bands).
USE_S1_CRS_WHEN_S2_CRS_MISSING = True


def prepare_s1(input_path, output_path):
    """Load a 1- or 2-band Sentinel-1 GeoTIFF, order/derive VV+VH, convert to dB
    clipped to TerraMind's expected range, and write a clean 2-band raster."""
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

    output = np.stack([vv_db, vh_db], axis=0).astype(np.float32)
    out_profile = make_safe_profile(profile, width, height, 2, "float32", CLIP_MIN)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(output)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.update_tags(SOURCE_FILE=str(input_path), SCALE_MODE=scale_mode)

    return output_path, scale_mode


def align_s2_to_s1(s2_path, s1_reference_path, output_path):
    """Reproject/resample Sentinel-2 onto the S1 raster's exact grid (CRS,
    transform, size) and expand to the 13 bands TerraMind expects. 3-band (RGB)
    and 4-band (RGB+NIR) inputs are expanded by placing the available bands in
    their correct slots and filling every other band with its Sen1Floods11
    training-set mean — this degrades to prototype quality but still runs.
    """
    s2_path = Path(s2_path)
    s1_reference_path = Path(s1_reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s1_reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        ref_profile = ref.profile.copy()

    if dst_crs is None:
        raise ValueError("S1 has no CRS")

    with rasterio.open(s2_path) as src:
        band_count = src.count
        if band_count not in (3, 4, 13):
            raise ValueError(f"{s2_path.name} has {band_count} bands, only 3, 4, 13 supported")

        src_crs = src.crs
        if src_crs is None:
            if USE_S1_CRS_WHEN_S2_CRS_MISSING:
                src_crs = dst_crs
            else:
                raise ValueError("S2 has no CRS")

        aligned = np.empty((13, dst_height, dst_width), dtype=np.float32)
        for i in range(13):
            aligned[i].fill(float(S2_MEAN[i]))

        if band_count == 13:
            mode = "13_band_original"
            mapping = {i: i - 1 for i in range(1, 14)}
        elif band_count == 3:
            mode = "RGB_3band_expanded_to_13"
            mapping = {1: 3, 2: 2, 3: 1} if S2_RGB_ORDER.upper() == "RGB" else {1: 1, 2: 2, 3: 3}
        else:
            mode = "RGBNIR_4band_expanded_to_13"
            mapping = {1: 3, 2: 2, 3: 1, 4: 7} if S2_RGB_ORDER.upper() == "RGB" else {1: 1, 2: 2, 3: 3, 4: 7}

        for src_band, dst_idx in mapping.items():
            reproject(
                source=rasterio.band(src, src_band),
                destination=aligned[dst_idx],
                src_transform=src.transform,
                src_crs=src_crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=float(S2_MEAN[dst_idx]),
                resampling=Resampling.bilinear,
            )

    profile = make_safe_profile(ref_profile, dst_width, dst_height, 13, "float32", None)
    names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(aligned)
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(SOURCE_FILE=str(s2_path), SOURCE_MODE=mode, OUTPUT_BAND_COUNT="13")

    return output_path, mode
