"""Rice-specific post-processing: sieve-based mask cleanup, polygon export
(area in m² and rai), and a downsampled RGB+mask preview.

Kept separate from ._common.postprocess (used by sar/multisensor) because the
cleanup algorithm (GDAL sieve, no morphological closing/hole-fill) and export
units (rai, the Thai land-area unit) are genuinely different from the flood
models' conventions.
"""

import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.features import sieve as rasterio_sieve
from shapely.geometry import shape

MIN_OBJECT_AREA_M2 = 3000.0
RAI_PER_M2 = 1.0 / 1600.0


def clean_rice_mask(probability, threshold, pixel_area_m2, min_object_area_m2=MIN_OBJECT_AREA_M2, do_sieve=True):
    """Threshold -> (optional) sieve out objects smaller than min_object_area_m2.
    Returns (raw_mask, clean_mask), both uint8 with 255 marking no-data."""
    valid = np.isfinite(probability)
    rice_binary = probability >= threshold  # NaN compares False -> reads as "not rice" pre-mask

    if do_sieve:
        minimum_pixels = max(1, int(math.ceil(min_object_area_m2 / pixel_area_m2)))
        clean_rice = rasterio_sieve(rice_binary.astype(np.uint8), size=minimum_pixels, connectivity=8)
    else:
        clean_rice = rice_binary.astype(np.uint8)

    raw_mask = np.full(probability.shape, 255, dtype=np.uint8)
    raw_mask[valid] = rice_binary.astype(np.uint8)[valid]

    clean_mask = np.full(probability.shape, 255, dtype=np.uint8)
    clean_mask[valid] = clean_rice[valid]

    return raw_mask, clean_mask


def export_rice_polygons(clean_mask, reference_path, output_path, scene_id, area_crs=None, simplify_tolerance_m=5.0):
    with rasterio.open(str(reference_path)) as ref:
        transform = ref.transform
        source_crs = ref.crs

    if source_crs is None:
        return None, 0.0, 0

    geometries = []
    for geom, value in shapes(clean_mask, mask=clean_mask == 1, transform=transform, connectivity=8):
        if int(value) != 1:
            continue
        geometry = shape(geom)
        if simplify_tolerance_m > 0:
            geometry = geometry.simplify(simplify_tolerance_m, preserve_topology=True)
        if not geometry.is_empty:
            geometries.append(geometry)

    if not geometries:
        return None, 0.0, 0

    gdf = gpd.GeoDataFrame({"scene": [str(scene_id)] * len(geometries), "class": ["rice"] * len(geometries)}, geometry=geometries, crs=source_crs)

    area_gdf = gdf.to_crs(area_crs) if area_crs else gdf
    gdf["area_m2"] = area_gdf.geometry.area
    gdf["area_rai"] = gdf["area_m2"] * RAI_PER_M2

    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, layer="rice", driver="GPKG")

    return output_path, float(gdf["area_rai"].sum()), len(gdf)


def _stretch_rgb(rgb):
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        low, high = np.nanpercentile(rgb[..., channel], [2, 98])
        output[..., channel] = np.clip((rgb[..., channel] - low) / max(high - low, 1e-6), 0, 1)
    return output


def create_rice_preview(scene_id, rgb_source_path, rgb_band_indexes, probability, clean_mask, rice_area_rai, output_path, max_size=1400):
    height, width = probability.shape
    scale = min(max_size / width, max_size / height, 1.0)
    preview_w = max(1, int(round(width * scale)))
    preview_h = max(1, int(round(height * scale)))

    with rasterio.open(str(rgb_source_path)) as src:
        rgb = src.read(indexes=rgb_band_indexes, out_shape=(3, preview_h, preview_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    rgb = _stretch_rgb(np.transpose(rgb, (1, 2, 0)))

    mask_preview = clean_mask.astype(np.float32)
    mask_preview[clean_mask == 255] = np.nan

    if scale < 1.0:
        step_h = max(1, height // preview_h)
        step_w = max(1, width // preview_w)
        prob_preview = probability[::step_h, ::step_w][:preview_h, :preview_w]
        mask_preview = mask_preview[::step_h, ::step_w][:preview_h, :preview_w]
    else:
        prob_preview = probability

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(rgb)
    axes[0].imshow(np.ma.masked_where(mask_preview != 1, mask_preview), alpha=0.55, vmin=0, vmax=1)
    axes[0].set_title(f"Scene {scene_id}\nRGB + rice mask")
    axes[0].axis("off")

    axes[1].imshow(prob_preview, vmin=0, vmax=1)
    axes[1].set_title("Rice probability")
    axes[1].axis("off")

    axes[2].imshow(mask_preview, vmin=0, vmax=1)
    axes[2].set_title(f"Clean rice mask\nArea = {rice_area_rai:.1f} rai")
    axes[2].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
