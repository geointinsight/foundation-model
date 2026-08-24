"""Public API: predict flood extent from paired Sentinel-1 + Sentinel-2 GeoTIFFs."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch

from .._common.geo import auto_utm_epsg, create_edge_fade, crop_raster_to_window, find_valid_window
from .._common.postprocess import (
    AUTO_THRESHOLD_MULTIPLIER,
    FLOOD_THRESHOLD,
    MAX_HOLE_PIXELS,
    MIN_COMPONENT_PIXELS,
    MIN_POLYGON_AREA_M2,
    clean_flood_mask,
    compute_otsu_threshold,
    create_scene_preview,
    export_flood_polygons,
    save_single_band_raster,
)
from ._inference import PATCH_SIZE, STRIDE, predict_flood_probability
from ._model import discover_checkpoint, load_model as _load_multisensor_model, resolve_device
from ._preprocess import align_s2_to_s1, prepare_s1


@dataclass
class SceneResult:
    scene_id: str
    s1_input: str
    s2_input: str
    threshold: float
    threshold_mode: str
    flood_area_km2: float
    flood_pixels: int
    polygon_count: int
    probability_path: str
    raw_mask_path: str
    clean_mask_path: str
    polygon_path: Optional[str]
    preview_path: Optional[str]


def load_model(checkpoint_path=None, device=None, search_dirs=None):
    """Resolve a device and load TerraMind once, so it can be reused across many
    predict_scene() calls instead of reloading per scene. Requires the
    'multisensor' extra (pip install -e '.[multisensor]')."""
    device = resolve_device(device)
    checkpoint_path = discover_checkpoint(search_dirs=search_dirs, explicit_path=checkpoint_path)
    model = _load_multisensor_model(checkpoint_path, device)
    return model, device, checkpoint_path


def predict_scene(
    s1_path,
    s2_path,
    output_dir,
    scene_id=None,
    model=None,
    device=None,
    checkpoint_path=None,
    threshold=None,
    auto_threshold=True,
    auto_threshold_multiplier=AUTO_THRESHOLD_MULTIPLIER,
    edge_discount_px=24,
    min_component_pixels=MIN_COMPONENT_PIXELS,
    max_hole_pixels=MAX_HOLE_PIXELS,
    min_polygon_area_m2=MIN_POLYGON_AREA_M2,
    area_crs=None,
    crop_nodata=True,
    save_preview=True,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    progress=True,
) -> SceneResult:
    """Predict probable flood extent from a Sentinel-1 + Sentinel-2 scene pair
    using TerraMind. Both modalities are required — for SAR-only scenes (no
    usable S2), use geoint_insight.sar instead.

    Pass a preloaded (model, device) — see load_model() — to avoid reloading
    the checkpoint on every call when predicting many scenes in a loop.
    """
    s1_path = Path(s1_path)
    s2_path = Path(s2_path)
    output_dir = Path(output_dir)
    scene_id = scene_id or s1_path.stem
    processed_dir = output_dir / "processed"
    result_dir = output_dir / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, device, checkpoint_path = load_model(checkpoint_path, device)
    else:
        device = device or next(model.parameters()).device

    s1_input = s1_path
    if crop_nodata:
        with rasterio.open(str(s1_path)) as src:
            crs, bounds = src.crs, src.bounds
        crop_window = find_valid_window(s1_path)
        if crop_window is not None:
            cropped_path = processed_dir / f"{scene_id}_S1_cropped.tif"
            crop_raster_to_window(s1_path, cropped_path, crop_window)
            s1_input = cropped_path
    else:
        with rasterio.open(str(s1_path)) as src:
            crs, bounds = src.crs, src.bounds

    if area_crs is None:
        area_crs = auto_utm_epsg(crs, bounds)

    s1_ready = processed_dir / f"{scene_id}_S1_db.tif"
    prepare_s1(s1_input, s1_ready)

    s2_ready = processed_dir / f"{scene_id}_S2_aligned.tif"
    align_s2_to_s1(s2_path, s1_ready, s2_ready)

    probability = predict_flood_probability(model, device, s1_ready, s2_ready, patch_size, stride, progress=progress)

    if edge_discount_px > 0:
        edge_fade = create_edge_fade(*probability.shape, edge_discount_px)
        mask_probability = (probability * edge_fade).astype(np.float32)
    else:
        mask_probability = probability

    if auto_threshold:
        otsu = compute_otsu_threshold(mask_probability)
        effective_threshold = min(otsu * auto_threshold_multiplier, 1.0)
        threshold_mode = "otsu"
    else:
        effective_threshold = threshold if threshold is not None else FLOOD_THRESHOLD
        threshold_mode = "fixed"

    raw_mask, clean_mask = clean_flood_mask(mask_probability, effective_threshold, min_component_pixels, max_hole_pixels)

    probability_path = result_dir / "01_flood_probability.tif"
    raw_mask_path = result_dir / "02_flood_mask_raw.tif"
    clean_mask_path = result_dir / "03_flood_mask_clean.tif"
    polygon_path = result_dir / "04_probable_flood_polygons.gpkg"
    preview_path = result_dir / "05_flood_prediction_preview.png"

    save_single_band_raster(probability_path, probability, s1_ready, "float32", -9999, "Flood probability")
    save_single_band_raster(raw_mask_path, raw_mask, s1_ready, "uint8", 255, "Raw probable flood mask")
    save_single_band_raster(clean_mask_path, clean_mask, s1_ready, "uint8", 255, "Clean probable flood mask")

    saved_polygon_path, flood_area_km2, polygon_count = export_flood_polygons(
        clean_mask, s1_ready, polygon_path, scene_id, area_crs, min_polygon_area_m2
    )

    saved_preview_path = None
    if save_preview:
        saved_preview_path = create_scene_preview(scene_id, s1_ready, probability, clean_mask, flood_area_km2, preview_path)

    flood_pixels = int((clean_mask == 1).sum())

    del probability, mask_probability, raw_mask, clean_mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return SceneResult(
        scene_id=str(scene_id),
        s1_input=str(s1_path),
        s2_input=str(s2_path),
        threshold=float(effective_threshold),
        threshold_mode=threshold_mode,
        flood_area_km2=flood_area_km2,
        flood_pixels=flood_pixels,
        polygon_count=polygon_count,
        probability_path=str(probability_path),
        raw_mask_path=str(raw_mask_path),
        clean_mask_path=str(clean_mask_path),
        polygon_path=str(saved_polygon_path) if saved_polygon_path else None,
        preview_path=str(saved_preview_path) if saved_preview_path else None,
    )


def predict_folder(scenes_dir, output_dir, s1_pattern="S1*.tif", s2_pattern="S2*.tif", **kwargs) -> list:
    """Convenience batch runner: pairs files matching s1_pattern/s2_pattern by
    matching filename suffix after the pattern's prefix (e.g. S1_sceneA.tif <->
    S2_sceneA.tif), reusing one loaded model across all of them. Returns a list
    of SceneResult.
    """
    scenes_dir = Path(scenes_dir)
    s1_paths = sorted(scenes_dir.glob(s1_pattern))
    if not s1_paths:
        raise FileNotFoundError(f"No files matching {s1_pattern!r} under {scenes_dir}")

    s1_prefix = s1_pattern.split("*")[0]
    s2_prefix = s2_pattern.split("*")[0]

    model, device, checkpoint_path = load_model(
        kwargs.pop("checkpoint_path", None), kwargs.pop("device", None)
    )

    results = []
    for s1_path in s1_paths:
        suffix = s1_path.name[len(s1_prefix):]
        s2_path = s1_path.with_name(f"{s2_prefix}{suffix}")
        if not s2_path.exists():
            raise FileNotFoundError(f"No matching S2 file for {s1_path.name}: expected {s2_path.name}")
        results.append(predict_scene(s1_path, s2_path, output_dir, model=model, device=device, **kwargs))
    return results
