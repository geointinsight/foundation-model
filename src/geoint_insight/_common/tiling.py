"""Generic tiled-inference utilities — model-agnostic, shared by every model."""

from contextlib import nullcontext

import numpy as np
import torch


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
    """Edge-extend (not zero-pad) so patches smaller than the model's patch size
    don't create a hard, unrealistic discontinuity at the true scene boundary."""
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
