"""geoint_insight.multisensor — Sentinel-1 + Sentinel-2 dual-modality flood-extent
prediction (TerraMind foundation model). Both S1 and S2 are required — for
SAR-only scenes, use geoint_insight.sar instead.

Requires the optional 'multisensor' extra: pip install -e ".[multisensor]"

    from geoint_insight.multisensor import predict_scene, sample_s1_path, sample_s2_path
    result = predict_scene(sample_s1_path(), sample_s2_path(), "outputs/")
"""

from ._data import sample_s1_path, sample_s2_path
from .pipeline import SceneResult, load_model, predict_folder, predict_scene

__all__ = [
    "predict_scene", "predict_folder", "load_model", "SceneResult",
    "sample_s1_path", "sample_s2_path",
]
