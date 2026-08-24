"""Tiled inference over a prepared 2-band dB Sentinel-1 raster."""

from contextlib import nullcontext

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm.auto import tqdm

from ._model import NORM_MEAN, NORM_STD
from ._preprocess import CLIP_MAX, CLIP_MIN

PATCH_SIZE = 512
STRIDE = 256

_NORM_MEAN = np.array(NORM_MEAN, dtype=np.float32)
_NORM_STD = np.array(NORM_STD, dtype=np.float32)


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def generate_positions(full_size, patch_size, stride):
    if full_size <= patch_size:
        return [0]
    positions = list(range(0, full_size - patch_size + 1, stride))
    last = full_size - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_array(array, target_size):
    """Edge-extend (not zero-pad) so patches smaller than PATCH_SIZE don't create
    a hard, unrealistic discontinuity at the true scene boundary."""
    c, h, w = array.shape
    pad_h, pad_w = target_size - h, target_size - w
    if pad_h == 0 and pad_w == 0:
        return array.astype(np.float32)
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge").astype(np.float32)


def create_tile_weight(size):
    """Pyramidal weight for blending overlapping tiles — highest confidence at
    the tile center, tapering toward its own edges."""
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    w = np.maximum(1.0 - np.abs(x), 0.05)
    return np.outer(w, w).astype(np.float32)


def predict_flood_probability(model, device, s1_path, patch_size=PATCH_SIZE, stride=STRIDE, progress=True):
    """Run the model over the whole raster via overlapping tiles, blending with
    a pyramidal weight to avoid seams at tile boundaries. Pixels with no valid
    input coverage anywhere are marked NaN (true no-data), not 0.0, so they don't
    read as "confidently no flood" downstream."""
    tile_weight = create_tile_weight(patch_size)
    clip_range = CLIP_MAX - CLIP_MIN

    with rasterio.open(str(s1_path)) as src:
        if src.count != 2:
            raise ValueError("Prepared S1 raster must have exactly 2 bands (VV, VH)")

        h, w = src.height, src.width
        probability_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        valid_any = np.zeros((h, w), dtype=bool)

        xs = generate_positions(w, patch_size, stride)
        ys = generate_positions(h, patch_size, stride)
        total = len(xs) * len(ys)
        bar = tqdm(total=total, desc="Sen1Floods11 inference", disable=not progress)

        for y in ys:
            for x in xs:
                patch_w = min(patch_size, w - x)
                patch_h = min(patch_size, h - y)
                window = Window(x, y, patch_w, patch_h)

                s1 = src.read(window=window).astype(np.float32)
                s1 = pad_array(s1, patch_size)
                valid = np.any(s1 > (CLIP_MIN + 0.1), axis=0)

                rescaled = (s1 - CLIP_MIN) / clip_range
                normed = (rescaled - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
                model_input = torch.from_numpy(normed[None]).to(device, non_blocking=True)

                with torch.inference_mode():
                    with autocast_context(device):
                        result = model(model_input)
                    logits = result["out"]
                    prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()

                prob[~valid] = 0.0
                weight = tile_weight[:patch_h, :patch_w]
                probability_sum[y:y + patch_h, x:x + patch_w] += prob[:patch_h, :patch_w] * weight
                weight_sum[y:y + patch_h, x:x + patch_w] += weight
                valid_any[y:y + patch_h, x:x + patch_w] |= valid[:patch_h, :patch_w]

                bar.update(1)
        bar.close()

    probability = np.clip(probability_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32)
    probability[~valid_any] = np.nan
    return probability
