"""geoint_insight.sentinel1 — Sentinel-1 SAR flood-extent prediction
(Sen1Floods11 S1-only baseline, FCN-ResNet50). Works from Sentinel-1 VV/VH
alone, no Sentinel-2 required.

    from geoint_insight.sentinel1 import predict_scene, sample_path
    result = predict_scene(sample_path(), "outputs/")
"""

from ._data import sample_path
from .pipeline import SceneResult, load_model, predict_folder, predict_scene

__all__ = ["predict_scene", "predict_folder", "load_model", "SceneResult", "sample_path"]
