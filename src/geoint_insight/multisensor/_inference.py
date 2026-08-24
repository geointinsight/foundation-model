"""Tiled dual-modality (S1+S2) inference for TerraMind."""

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm.auto import tqdm

from .._common.tiling import autocast_context, create_tile_weight, generate_positions, pad_array
from ._model import S1_MEAN, S1_STD, S2_MEAN, S2_STD

PATCH_SIZE = 512
STRIDE = 256


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


def predict_flood_probability(model, device, s1_path, s2_path, patch_size=PATCH_SIZE, stride=STRIDE, progress=True):
    """Tiled S1+S2 inference. TerraMind's backbone merges modality tokens with
    `mean` — passing only "S1GRD" (no S2L1C) would also work for S1-only
    inference, but that path is known to collapse (out-of-distribution decoder
    output); this package always requires both modalities for TerraMind — use
    the sar model instead for SAR-only scenes.
    """
    tile_weight = create_tile_weight(patch_size)

    with rasterio.open(str(s1_path)) as s1_src, rasterio.open(str(s2_path)) as s2_src:
        if s1_src.count != 2:
            raise ValueError("Prepared S1 must have 2 bands")
        if s2_src.count != 13:
            raise ValueError("Prepared S2 must have 13 bands")

        h, w = s1_src.height, s1_src.width
        probability_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        valid_any = np.zeros((h, w), dtype=bool)

        xs = generate_positions(w, patch_size, stride)
        ys = generate_positions(h, patch_size, stride)
        bar = tqdm(total=len(xs) * len(ys), desc="TerraMind inference (S1+S2)", disable=not progress)

        for y in ys:
            for x in xs:
                patch_w = min(patch_size, w - x)
                patch_h = min(patch_size, h - y)
                window = Window(x, y, patch_w, patch_h)

                s1 = s1_src.read(window=window).astype(np.float32)
                s1 = pad_array(s1, patch_size)
                s1_norm = (s1 - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
                valid = np.any(s1 > -34.9, axis=0)

                s2 = s2_src.read(window=window).astype(np.float32)
                s2 = pad_array(s2, patch_size)
                s2_norm = (s2 - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
                valid = valid | np.any(s2 != 0, axis=0)

                model_input = {
                    "S1GRD": torch.from_numpy(s1_norm[None]).to(device, non_blocking=True),
                    "S2L1C": torch.from_numpy(s2_norm[None]).to(device, non_blocking=True),
                }

                with torch.inference_mode():
                    with autocast_context(device):
                        result = model(model_input)
                    logits = _extract_logits(result)
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
