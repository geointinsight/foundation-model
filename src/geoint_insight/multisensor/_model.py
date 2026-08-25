"""TerraMind (S1+S2 dual-modality) flood model — loading + checkpoint discovery.

Requires the optional `multisensor` extra: pip install -e ".[multisensor]"
"""

from pathlib import Path

import numpy as np
import torch

# Sen1Floods11 training-set band statistics TerraMind was fine-tuned against.
S2_MEAN = np.array(
    [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849,
     3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929],
    dtype=np.float32,
)
S2_STD = np.array(
    [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131,
     1679.732, 1727.260, 1024.687, 442.165, 1331.411, 1160.419],
    dtype=np.float32,
)
S1_MEAN = np.array([-12.599, -20.293], dtype=np.float32)
S1_STD = np.array([5.195, 5.890], dtype=np.float32)

MODEL_ARGS = {
    "backbone": "terramind_v1_base",
    "backbone_pretrained": False,
    "backbone_modalities": ["S2L1C", "S1GRD"],
    "backbone_merge_method": "mean",
    "necks": [
        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
        {"name": "ReshapeTokensToImage", "remove_cls_token": False},
        {"name": "LearnedInterpolateToPyramidal"},
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "head_dropout": 0.1,
    "num_classes": 2,
}

# Canonical checkpoint filename this model looks for after `geoint-insight
# setup` — see ../_setup.py.
CHECKPOINT_NAME = "multisensor_geoint_insight.ckpt"


def resolve_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_checkpoint(search_dirs=None, explicit_path=None):
    """Find a .ckpt checkpoint file. Looks in explicit_path first, then any of
    search_dirs (default: ./checkpoints and ./checkpoint relative to cwd)."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    search_dirs = search_dirs or [Path("checkpoints"), Path("checkpoint")]
    candidates = []
    for d in search_dirs:
        d = Path(d)
        if d.exists():
            candidates.extend(d.rglob("*.ckpt"))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt checkpoint found (expected {CHECKPOINT_NAME}). Run:\n"
            "  geoint-insight setup\n"
            "to download it, or pass checkpoint_path explicitly."
        )
    named = [p for p in candidates if p.name == CHECKPOINT_NAME]
    return named[0] if named else candidates[0]


def load_model(checkpoint_path, device):
    """Requires terratorch (pip install -e '.[multisensor]')."""
    try:
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError as exc:
        raise ImportError(
            "multisensor requires the optional 'multisensor' extra: "
            "pip install -e '.[multisensor]'"
        ) from exc

    model = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_factory="EncoderDecoderFactory",
        model_args=MODEL_ARGS,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model
