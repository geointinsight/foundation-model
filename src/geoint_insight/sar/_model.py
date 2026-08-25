"""Sen1Floods11 S1-only baseline model (FCN-ResNet50) — loading + checkpoint discovery."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models

GROUPNORM_GROUPS = 16
NORM_MEAN = [0.6851, 0.5235]
NORM_STD = [0.0820, 0.1102]

# Canonical checkpoint filename this model looks for after `geoint-insight
# setup` — see ../_setup.py.
CHECKPOINT_NAME = "sar_geoint_insight.cp"


def resolve_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_checkpoint(search_dirs=None, explicit_path=None):
    """Find a .cp checkpoint file. Looks in explicit_path first, then any of
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
            candidates.extend(d.rglob("*.cp"))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .cp checkpoint found (expected {CHECKPOINT_NAME}). Run:\n"
            "  geoint-insight setup\n"
            "to download it, or pass checkpoint_path explicitly."
        )
    named = [p for p in candidates if p.name == CHECKPOINT_NAME]
    return named[0] if named else candidates[0]


def build_baseline_model():
    """torchvision fcn_resnet50 adapted for 2-channel SAR input, BatchNorm
    replaced with GroupNorm (matches the original Sen1Floods11 training setup)."""

    def convert_bn_to_gn(module, num_groups):
        if isinstance(module, nn.BatchNorm2d):
            return nn.GroupNorm(num_groups, module.num_features, eps=module.eps, affine=module.affine)
        for name, child in module.named_children():
            module.add_module(name, convert_bn_to_gn(child, num_groups))
        return module

    net = tv_models.segmentation.fcn_resnet50(weights=None, num_classes=2, weights_backbone=None)
    net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return convert_bn_to_gn(net, GROUPNORM_GROUPS)


def load_baseline_model(checkpoint_path, device):
    net = build_baseline_model()
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(state_dict, strict=True)
    net = net.to(device)
    net.eval()
    return net
