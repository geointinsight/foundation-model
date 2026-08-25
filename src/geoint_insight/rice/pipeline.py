"""Public API: predict rice extent from a multitemporal stack of Sentinel-2
GeoTIFFs sharing one grid (same scene, several acquisition dates).

Requires the optional 'rice' extra: pip install -e ".[rice]"
"""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
import torch

from .._common.postprocess import save_single_band_raster
from ._inference import MIN_PATCH_VALID_RATIO, PATCH_SIZE, STRIDE, predict_rice_probability
from ._model import discover_checkpoint, load_model as _load_rice_model, resolve_device
from ._postprocess import MIN_OBJECT_AREA_M2, clean_rice_mask, create_rice_preview, export_rice_polygons
from ._preprocess import find_timestamp_stacks, resolve_band_indexes


@dataclass
class SceneResult:
    scene_id: str
    stack_inputs: list
    timestamps: list
    threshold: float
    rice_area_m2: float
    rice_area_km2: float
    rice_area_rai: float
    rice_pixels: int
    probability_path: str
    raw_mask_path: str
    clean_mask_path: str
    polygon_path: Optional[str]
    preview_path: Optional[str]


def load_model(checkpoint_path=None, device=None, search_dirs=None, allow_hf_fallback=True):
    """Resolve a device and load the fine-tuned rice checkpoint (plus its
    training_config.json / normalization_stats.json) once, so it can be reused
    across many predict_scene() calls. Returns (model, device, config) — config
    is a RiceCheckpointConfig (band_names, band_means, band_stds,
    recommended_threshold, training_timestamps), needed by predict_scene()."""
    device = resolve_device(device)
    config = discover_checkpoint(search_dirs=search_dirs, explicit_path=checkpoint_path)
    model = _load_rice_model(config.checkpoint_path, device, allow_hf_fallback=allow_hf_fallback)
    return model, device, config


def _open_and_validate_grid(stack_paths, valid_mask_paths):
    sources = [rasterio.open(str(p)) for p in stack_paths]
    mask_sources = [rasterio.open(str(p)) if p is not None else None for p in valid_mask_paths]

    reference = sources[0]
    for source, path in zip(sources, stack_paths):
        if (source.width, source.height, source.transform, source.crs) != (
            reference.width, reference.height, reference.transform, reference.crs
        ):
            raise RuntimeError(f"Grid mismatch: {path}")
    for source, path in zip(mask_sources, valid_mask_paths):
        if source is None:
            continue
        if (source.width, source.height, source.transform, source.crs) != (
            reference.width, reference.height, reference.transform, reference.crs
        ):
            raise RuntimeError(f"Valid-mask grid mismatch: {path}")

    return sources, mask_sources, reference


def predict_scene(
    stack_paths,
    output_dir,
    valid_mask_paths=None,
    scene_id=None,
    model=None,
    device=None,
    config=None,
    checkpoint_path=None,
    threshold=None,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    min_patch_valid_ratio=MIN_PATCH_VALID_RATIO,
    do_sieve=True,
    min_object_area_m2=MIN_OBJECT_AREA_M2,
    export_vector=False,
    simplify_tolerance_m=5.0,
    area_crs=None,
    save_preview=True,
    progress=True,
) -> SceneResult:
    """Predict rice extent from a multitemporal stack of Sentinel-2 GeoTIFFs —
    one file per acquisition date, all sharing the same grid. Order matters:
    stack_paths[i] must correspond to the same real-world date the model
    expects at temporal position i (see config.training_timestamps).

    Pass a preloaded (model, device, config) — see load_model() — to avoid
    reloading the checkpoint on every call when predicting many scenes.
    """
    stack_paths = [Path(p) for p in stack_paths]
    valid_mask_paths = list(valid_mask_paths) if valid_mask_paths is not None else [None] * len(stack_paths)
    if len(valid_mask_paths) != len(stack_paths):
        raise ValueError("valid_mask_paths must have the same length as stack_paths")

    output_dir = Path(output_dir)
    scene_id = scene_id or stack_paths[0].stem
    result_dir = output_dir / "outputs"
    result_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model, device, config = load_model(checkpoint_path, device)
    elif config is None:
        raise ValueError("config is required when passing a preloaded model (see load_model())")
    else:
        device = device or next(model.parameters()).device

    sources, mask_sources, reference = _open_and_validate_grid(stack_paths, valid_mask_paths)
    try:
        band_indexes = resolve_band_indexes(reference, config.band_names)
        for source, path in zip(sources, stack_paths):
            if source.count < max(band_indexes):
                raise RuntimeError(f"{path}: has only {source.count} bands, need at least {max(band_indexes)}")

        pixel_area_m2 = abs(reference.res[0] * reference.res[1])

        probability = predict_rice_probability(
            model, device, sources, mask_sources, band_indexes, config.band_means, config.band_stds,
            patch_size=patch_size, stride=stride, min_patch_valid_ratio=min_patch_valid_ratio, progress=progress,
        )

        effective_threshold = threshold if threshold is not None else config.recommended_threshold
        raw_mask, clean_mask = clean_rice_mask(probability, effective_threshold, pixel_area_m2, min_object_area_m2, do_sieve)

        probability_path = result_dir / "01_rice_probability.tif"
        raw_mask_path = result_dir / "02_rice_mask_raw.tif"
        clean_mask_path = result_dir / "03_rice_mask_clean.tif"
        polygon_path = result_dir / "04_rice_polygons.gpkg"
        preview_path = result_dir / "05_rice_prediction_preview.png"

        reference_raster_path = stack_paths[0]
        save_single_band_raster(probability_path, probability, reference_raster_path, "float32", -9999, "Rice probability")
        save_single_band_raster(raw_mask_path, raw_mask, reference_raster_path, "uint8", 255, "Raw rice mask")
        save_single_band_raster(clean_mask_path, clean_mask, reference_raster_path, "uint8", 255, "Clean rice mask")

        saved_polygon_path = None
        if export_vector:
            saved_polygon_path, _polygon_area_rai, _polygon_count = export_rice_polygons(
                clean_mask, reference_raster_path, polygon_path, scene_id, area_crs, simplify_tolerance_m
            )

        rice_pixels = int((clean_mask == 1).sum())
        rice_area_m2 = rice_pixels * pixel_area_m2
        rice_area_rai = rice_area_m2 / 1600.0

        saved_preview_path = None
        if save_preview:
            try:
                rgb_indexes = resolve_band_indexes(reference, ["B04", "B03", "B02"])
                middle_path = stack_paths[len(stack_paths) // 2]
                saved_preview_path = create_rice_preview(
                    scene_id, middle_path, rgb_indexes, probability, clean_mask, rice_area_rai, preview_path
                )
            except ValueError:
                pass  # RGB bands not resolvable on this source — skip preview, keep the raster outputs

    finally:
        for source in sources:
            source.close()
        for source in mask_sources:
            if source is not None:
                source.close()

    del probability, raw_mask, clean_mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return SceneResult(
        scene_id=str(scene_id),
        stack_inputs=[str(p) for p in stack_paths],
        timestamps=list(config.training_timestamps) if config.training_timestamps else [],
        threshold=float(effective_threshold),
        rice_area_m2=float(rice_area_m2),
        rice_area_km2=float(rice_area_m2 / 1_000_000),
        rice_area_rai=float(rice_area_rai),
        rice_pixels=rice_pixels,
        probability_path=str(probability_path),
        raw_mask_path=str(raw_mask_path),
        clean_mask_path=str(clean_mask_path),
        polygon_path=str(saved_polygon_path) if saved_polygon_path else None,
        preview_path=str(saved_preview_path) if saved_preview_path else None,
    )


def predict_stacks_folder(
    stack_dir,
    output_dir,
    stack_glob="*STACK*.tif",
    valid_mask_dir=None,
    valid_mask_glob="*.tif",
    timestamps=None,
    model=None,
    device=None,
    config=None,
    checkpoint_path=None,
    **kwargs,
) -> SceneResult:
    """Discover per-date stack GeoTIFFs under stack_dir (see
    ._preprocess.find_timestamp_stacks) and predict one multitemporal scene
    from them. If timestamps is not given, uses the checkpoint's own
    training_config.json timestamps when available, else every discovered date.
    """
    if model is None:
        model, device, config = load_model(checkpoint_path, device)
    elif config is None:
        raise ValueError("config is required when passing a preloaded model (see load_model())")

    selected_timestamps = timestamps if timestamps is not None else (config.training_timestamps or None)
    _dates, stack_paths, mask_paths = find_timestamp_stacks(stack_dir, stack_glob, valid_mask_dir, valid_mask_glob, selected_timestamps)

    return predict_scene(
        stack_paths, output_dir, valid_mask_paths=mask_paths,
        model=model, device=device, config=config, **kwargs,
    )
