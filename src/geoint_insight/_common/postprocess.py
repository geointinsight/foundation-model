"""Thresholding, mask cleaning, and export (raster + polygons + preview)."""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from .geo import make_safe_profile

MIN_COMPONENT_PIXELS = 30
MIN_POLYGON_AREA_M2 = 1000.0
MAX_HOLE_PIXELS = 500
FLOOD_THRESHOLD = 0.6
AUTO_THRESHOLD_MULTIPLIER = 0.9


def compute_otsu_threshold(probability, bins=256):
    """Auto-pick a split point between the two probability clusters in the
    histogram instead of relying on one fixed cutoff. Validated on Sen1Floods11
    ground-truth chips to beat every fixed threshold tried, and adapts per scene
    — helpful for low-water scenes a single global threshold would under-detect.
    """
    values = probability.ravel().astype(np.float64)
    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    if weight1[-1] == 0:
        return 0.5

    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if not np.any(inter_class_variance > 0):
        return 0.5
    return float(bin_centers[np.argmax(inter_class_variance)])


def fill_small_holes(mask, max_hole_pixels):
    """Fill enclosed non-flood gaps up to max_hole_pixels in size. Unlike
    ndimage.binary_fill_holes (which fills every enclosed region regardless of
    size), this caps what counts as a fillable "hole" so a large real terrain
    feature (e.g. high ground surrounded by flooding) isn't swallowed whole."""
    if max_hole_pixels <= 0:
        return mask
    inverse = ~mask
    labeled, n = ndimage.label(inverse)
    if n == 0:
        return mask
    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)
    sizes = np.bincount(labeled.ravel())
    fillable = np.ones(len(sizes), dtype=bool)
    fillable[0] = False
    fillable[sizes > max_hole_pixels] = False
    for label_id in border_labels:
        fillable[label_id] = False
    return mask | fillable[labeled]


def clean_flood_mask(probability, threshold, min_component_pixels=MIN_COMPONENT_PIXELS, max_hole_pixels=MAX_HOLE_PIXELS):
    nodata_mask = ~np.isfinite(probability)
    raw = probability >= threshold  # NaN comparisons are False -> nodata reads as "not flood" here

    labeled, _ = ndimage.label(raw)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_component_pixels
    if keep.size > 0:
        keep[0] = False
    clean = keep[labeled]
    clean = ndimage.binary_closing(clean, structure=np.ones((3, 3), dtype=bool))
    clean = fill_small_holes(clean, max_hole_pixels)

    raw_out = raw.astype(np.uint8)
    clean_out = clean.astype(np.uint8)
    # 255 marks true no-data so it reads as nodata in GIS tools, not "confidently
    # not flooded".
    raw_out[nodata_mask] = 255
    clean_out[nodata_mask] = 255
    return raw_out, clean_out


def save_single_band_raster(output_path, array, reference_path, dtype, nodata, description):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if np.issubdtype(array.dtype, np.floating) and nodata is not None:
        array = np.where(np.isfinite(array), array, nodata)

    with rasterio.open(str(reference_path)) as ref:
        profile = make_safe_profile(ref.profile, ref.width, ref.height, 1, dtype, nodata)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.set_band_description(1, description)
    return output_path


def export_flood_polygons(clean_mask, reference_path, output_path, scene_id, area_crs, min_polygon_area_m2=MIN_POLYGON_AREA_M2):
    with rasterio.open(str(reference_path)) as ref:
        transform = ref.transform
        source_crs = ref.crs

    if source_crs is None:
        return None, 0.0, 0

    records = []
    for geom, value in shapes(clean_mask, mask=clean_mask == 1, transform=transform):
        if int(value) == 1:
            records.append({"scene": str(scene_id), "class": "probable_flood", "geometry": shape(geom)})

    if not records:
        return None, 0.0, 0

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    area_gdf = gdf.to_crs(area_crs)
    area_gdf["area_m2"] = area_gdf.geometry.area

    keep = area_gdf["area_m2"] >= min_polygon_area_m2
    gdf = gdf.loc[keep.values].copy()
    area_gdf = area_gdf.loc[keep.values].copy()
    if len(gdf) == 0:
        return None, 0.0, 0

    gdf["area_m2"] = area_gdf["area_m2"].values
    gdf["area_km2"] = gdf["area_m2"] / 1_000_000

    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, layer="probable_flood", driver="GPKG")

    return output_path, float(gdf["area_km2"].sum()), len(gdf)


def create_scene_preview(scene_id, s1_path, probability, clean_mask, flood_area_km2, output_path):
    with rasterio.open(str(s1_path)) as src:
        vv = src.read(1).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title(f"Scene {scene_id}\nSentinel-1 VV dB")
    axes[0].axis("off")

    im = axes[1].imshow(probability, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Flood probability")
    axes[1].axis("off")

    vv_norm = np.clip((vv + 25.0) / 25.0, 0.0, 1.0)
    rgb = np.repeat(vv_norm[..., None], 3, axis=-1)
    axes[2].imshow(rgb)
    axes[2].imshow(np.ma.masked_where(clean_mask == 0, clean_mask), cmap="autumn", alpha=0.60, vmin=0, vmax=1)
    axes[2].set_title(f"Probable flood overlay\nArea = {flood_area_km2:.2f} km²")
    axes[2].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
