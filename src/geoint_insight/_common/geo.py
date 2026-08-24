"""Small raster/geometry helpers shared across the package."""

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


def auto_utm_epsg(src_crs, bounds):
    """Pick a reasonable UTM EPSG code for area calculations, from a raster's own
    bounds — so the package works out of the box anywhere in the world instead of
    assuming one fixed region. Uses the scene's centroid longitude/latitude with
    the standard UTM zone formula.
    """
    left, bottom, right, top = transform_bounds(src_crs, "EPSG:4326", *bounds)
    lon = (left + right) / 2.0
    lat = (bottom + top) / 2.0
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def make_safe_profile(source_profile, width, height, count, dtype, nodata):
    """rasterio profile for a derived raster, stripped of tiling/compression
    settings that don't necessarily still apply once shape/dtype change."""
    profile = source_profile.copy()
    for key in ["blockxsize", "blockysize", "tiled", "interleave", "photometric"]:
        profile.pop(key, None)
    profile.update(
        driver="GTiff", width=width, height=height, count=count, dtype=dtype,
        nodata=nodata, compress="deflate", BIGTIFF="IF_SAFER",
    )
    if width >= 16 and height >= 16:
        profile.update(
            tiled=True,
            blockxsize=min(512, max(16, (width // 16) * 16)),
            blockysize=min(512, max(16, (height // 16) * 16)),
        )
    else:
        profile.update(tiled=False)
    return profile


def find_valid_window(path):
    """Tight rasterio Window bounding every finite pixel across all bands, or
    None if the raster is already fully valid. Some products declare a fixed
    tile size but only cover it partially (e.g. clipped to a swath edge); cropping
    first avoids wasting compute on all-nodata regions and keeps reported flood
    area/percentage honest (computed against the real footprint, not the nominal
    file size).
    """
    with rasterio.open(path) as src:
        data = src.read()

    valid = np.isfinite(data).any(axis=0)
    if valid.all():
        return None
    if not valid.any():
        raise ValueError(f"{path}: no valid (finite) pixels found in any band")

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    from rasterio.windows import Window

    return Window(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)


def crop_raster_to_window(input_path, output_path, window):
    input_path, output_path = str(input_path), str(output_path)
    with rasterio.open(input_path) as src:
        data = src.read(window=window)
        transform = src.window_transform(window)
        nodata = src.nodata
        count = src.count
        dtype = src.dtypes[0]
        descriptions = src.descriptions
        profile = make_safe_profile(src.profile, window.width, window.height, count, dtype, nodata)
    profile["transform"] = transform

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
    return output_path


def create_edge_fade(height, width, discount_px):
    """Fade factor (0..1) that ramps down near the raster's border. Predictions
    right at the edge of a scene are inherently less reliable — the model has no
    real neighboring context beyond the crop — so this dampens flood probability
    within `discount_px` of any border instead of trusting it at full confidence.

    Note: this always fades all four sides. The full package's grid-mosaic mode
    (detecting which sides border a real neighboring tile and skipping the fade
    there) is not included in this lightweight version — if you're tiling a large
    area into a grid, set discount_px=0 or trim the fixed border yourself before
    mosaicking, otherwise every tile edge will show reduced confidence.
    """
    if discount_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    y_idx = np.arange(height, dtype=np.float32)
    x_idx = np.arange(width, dtype=np.float32)
    dist_y = np.minimum(y_idx, height - 1 - y_idx)
    dist_x = np.minimum(x_idx, width - 1 - x_idx)
    dist = np.minimum(dist_y[:, None], dist_x[None, :])
    return np.clip(dist / float(discount_px), 0.0, 1.0)
