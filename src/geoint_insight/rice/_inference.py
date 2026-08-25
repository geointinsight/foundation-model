"""Tiled multitemporal inference for the rice model — Hann-blended patches."""

import numpy as np
import torch
from tqdm.auto import tqdm

from .._common.tiling import autocast_context, create_hann_weight, generate_positions
from ._preprocess import preprocess_patch, read_patch

PATCH_SIZE = 256
STRIDE = 128
MIN_PATCH_VALID_RATIO = 0.05


def _extract_logits(result):
    if torch.is_tensor(result):
        return result
    if hasattr(result, "output"):
        return result.output
    if hasattr(result, "logits"):
        return result.logits
    if isinstance(result, dict):
        for key in ["output", "logits", "prediction", "pred"]:
            if key in result:
                return result[key]
    if isinstance(result, (tuple, list)):
        return result[0]
    raise TypeError(f"Cannot extract logits: {type(result)}")


def predict_rice_probability(
    model,
    device,
    sources,
    mask_sources,
    band_indexes,
    band_means,
    band_stds,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    min_patch_valid_ratio=MIN_PATCH_VALID_RATIO,
    progress=True,
):
    """Tiled multitemporal rice-probability inference over a full scene.
    sources/mask_sources are open rasterio datasets, one pair per timestamp,
    already validated to share the same grid. Returns a full-size (H,W)
    float32 probability array with NaN where no patch covered a pixel with
    enough valid data.
    """
    reference = sources[0]
    h, w = reference.height, reference.width

    tile_weight = create_hann_weight(patch_size)
    probability_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    xs = generate_positions(w, patch_size, stride)
    ys = generate_positions(h, patch_size, stride)
    bar = tqdm(total=len(xs) * len(ys), desc="Rice inference (multitemporal)", disable=not progress)

    for y in ys:
        for x in xs:
            raw_images, valid_masks = read_patch(sources, mask_sources, band_indexes, x, y, patch_size)

            usable_ratio = float(np.any(valid_masks == 1, axis=0).mean())
            if usable_ratio < min_patch_valid_ratio:
                bar.update(1)
                continue

            patch = preprocess_patch(raw_images, valid_masks, band_means, band_stds)
            batch = torch.from_numpy(patch[None, ...]).to(device, non_blocking=True)

            with torch.inference_mode():
                with autocast_context(device):
                    result = model(batch)
                logits = _extract_logits(result)
                prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()

            probability_sum[y:y + patch_size, x:x + patch_size] += prob * tile_weight
            weight_sum[y:y + patch_size, x:x + patch_size] += tile_weight

            bar.update(1)
    bar.close()

    probability = np.full((h, w), np.nan, dtype=np.float32)
    valid = weight_sum > 0
    probability[valid] = np.clip(probability_sum[valid] / weight_sum[valid], 0.0, 1.0)
    return probability
