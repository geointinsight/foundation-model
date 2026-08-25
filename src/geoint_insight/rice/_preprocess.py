"""Multitemporal Sentinel-2 stack discovery, band mapping, and per-patch
temporal-median imputation for the rice model."""

import re
import warnings
from pathlib import Path

import numpy as np
from rasterio.windows import Window

DATE_PATTERN = re.compile(r"(20\d{6})")

# Fallback band-index mapping used when a stack has no usable band
# descriptions but has enough bands to plausibly be the expected showcase
# layout (10+ bands, B02/B03/B04/B8A/B11/B12 at fixed positions).
DEFAULT_SOURCE_BAND_INDEX = {"B02": 1, "B03": 2, "B04": 3, "B8A": 8, "B11": 9, "B12": 10}

# Each group is a set of equivalent band names. A checkpoint's own band names
# (from model_args["backbone_bands"], e.g. "BLUE"/"NIR_NARROW" — TerraMind's
# internal vocabulary) and a source raster's band descriptions (e.g.
# Sentinel-2 "B02"/"B8A") are matched symmetrically against these groups, so
# it doesn't matter which vocabulary either side happens to use.
ALIAS_GROUPS = [
    {"B02", "BLUE"},
    {"B03", "GREEN"},
    {"B04", "RED"},
    {"B8A", "NIR_NARROW"},
    {"B11", "SWIR_1", "SWIR1"},
    {"B12", "SWIR_2", "SWIR2"},
]


def extract_date(path: Path):
    match = DATE_PATTERN.search(Path(path).name)
    return match.group(1) if match else None


def find_timestamp_stacks(stack_dir, stack_glob="*STACK*.tif", valid_mask_dir=None, valid_mask_glob="*.tif", timestamps=None):
    """Discover per-date stack GeoTIFFs (and optional matching valid masks) in
    stack_dir, keyed by a YYYYMMDD date parsed from each filename.

    If timestamps is given, only those exact dates are used (error if any are
    missing) and the result is ordered to match. Otherwise every discovered
    date is used, sorted ascending.

    Returns (timestamps, stack_paths, mask_paths) — mask_paths entries are
    None where no matching valid mask was found for that date.
    """
    stack_dir = Path(stack_dir)
    stack_by_date = {}
    for path in sorted(stack_dir.glob(stack_glob)):
        date = extract_date(path)
        if date:
            stack_by_date[date] = path
    if not stack_by_date:
        raise FileNotFoundError(f"No stacks matching {stack_glob!r} found under {stack_dir}")

    mask_by_date = {}
    if valid_mask_dir is not None:
        valid_mask_dir = Path(valid_mask_dir)
        if valid_mask_dir.exists():
            for path in sorted(valid_mask_dir.glob(valid_mask_glob)):
                date = extract_date(path)
                if date:
                    mask_by_date[date] = path

    if timestamps is not None:
        selected = list(timestamps)
        missing = [d for d in selected if d not in stack_by_date]
        if missing:
            raise FileNotFoundError(f"Missing stacks for dates: {missing}")
    else:
        selected = sorted(stack_by_date)

    stack_paths = [stack_by_date[d] for d in selected]
    mask_paths = [mask_by_date.get(d) for d in selected]
    return selected, stack_paths, mask_paths


def _normalize_description(value):
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9_]", "", str(value).upper())


def _alias_group(name):
    """The set of normalized names equivalent to `name` (its ALIAS_GROUPS
    entry), or just {name} itself if it's not part of any known group."""
    normalized = _normalize_description(name)
    for group in ALIAS_GROUPS:
        normalized_group = {_normalize_description(a) for a in group}
        if normalized in normalized_group:
            return normalized_group
    return {normalized}


def _default_index_for(band_name):
    group = _alias_group(band_name)
    for canonical, index in DEFAULT_SOURCE_BAND_INDEX.items():
        if _normalize_description(canonical) in group:
            return index
    return None


def resolve_band_indexes(source, band_names):
    """Resolve model band names to 1-based band indexes in `source`: prefer
    matching GeoTIFF band descriptions (with known aliases, checked
    symmetrically — works whether band_names or the source's own
    descriptions use the "B02" or "BLUE"-style vocabulary), else fall back to
    the fixed showcase layout if the source has enough bands for it."""
    descriptions = [_normalize_description(v) for v in source.descriptions]

    resolved = []
    for band_name in band_names:
        aliases = _alias_group(band_name)
        found_index = None
        for index, description in enumerate(descriptions, start=1):
            if description in aliases:
                found_index = index
                break
        if found_index is None:
            resolved = []
            break
        resolved.append(found_index)

    if resolved:
        return resolved

    fallback_indexes = [_default_index_for(b) for b in band_names]
    if all(i is not None for i in fallback_indexes) and source.count >= max(fallback_indexes):
        return fallback_indexes

    raise ValueError(
        f"Cannot map bands {band_names} onto source with {source.count} bands "
        f"and descriptions {source.descriptions}"
    )


def temporal_median_impute(images, valid_masks):
    """images: [T,C,H,W] float32, valid_masks: [T,H,W] (nonzero = valid).
    Per-pixel invalid timestamps are replaced with the per-pixel median across
    the valid timestamps; pixels invalid at every timestamp become 0."""
    images_float = images.astype(np.float32, copy=False)
    valid_bool = valid_masks.astype(bool, copy=False)

    masked = np.where(valid_bool[:, None, :, :], images_float, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)

    output = images_float.copy()
    for t in range(images.shape[0]):
        invalid = ~valid_bool[t]
        output[t] = np.where(invalid[None, :, :], median, output[t])
    return output


def read_patch(sources, mask_sources, band_indexes, x, y, patch_size):
    """Read one [T,C,patch,patch] window across all timestamp sources, plus a
    [T,patch,patch] valid-pixel mask (from mask_sources where given, else
    inferred as nonzero-and-not-65535 across bands)."""
    window = Window(x, y, patch_size, patch_size)
    images, masks = [], []

    for image_source, mask_source in zip(sources, mask_sources):
        image = image_source.read(indexes=band_indexes, window=window).astype(np.float32, copy=False)
        if image.shape != (len(band_indexes), patch_size, patch_size):
            raise RuntimeError(f"Unexpected patch shape: {image.shape}")

        if mask_source is not None:
            valid = mask_source.read(1, window=window) > 0
        else:
            valid = np.all(image > 0, axis=0) & np.all(image != 65535, axis=0)

        images.append(image)
        masks.append(valid.astype(np.uint8, copy=False))

    return np.stack(images, axis=0), np.stack(masks, axis=0)


def preprocess_patch(raw_images, valid_masks, band_means, band_stds):
    """[T,C,H,W] -> temporal-impute -> [C,T,H,W] -> z-normalize per band."""
    images = temporal_median_impute(raw_images, valid_masks)
    images = np.transpose(images, (1, 0, 2, 3))
    images = (images - band_means[:, None, None, None]) / band_stds[:, None, None, None]
    return np.ascontiguousarray(images, dtype=np.float32)
