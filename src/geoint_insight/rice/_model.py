"""TerraMind multitemporal rice model — checkpoint discovery, config, loading.

Requires the optional `rice` extra: pip install -e ".[rice]"

Unlike sar/multisensor (one self-contained checkpoint file), a rice checkpoint
is fine-tuned per dataset and ships as a directory of three files:

    <model_dir>/best.ckpt (or last.ckpt / *rice*iou*.ckpt)
    <model_dir>/training_config.json      -> timestamps, recommended threshold
    <model_dir>/normalization_stats.json  -> band_names, means, stds

The checkpoint is self-describing (architecture read from its own
hyper_parameters), so there's no hardcoded MODEL_ARGS here unlike multisensor.
"""

import copy
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

# Pretrained TerraMind fallback, only used if offline checkpoint reconstruction
# fails (see load_model below).
HF_REPO_ID = "ibm-esa-geospatial/TerraMind-1.0-base"
HF_FILENAME = "TerraMind_v1_base.pt"


@dataclass
class RiceCheckpointConfig:
    checkpoint_path: Path
    band_names: list
    band_means: "object"
    band_stds: "object"
    recommended_threshold: float
    training_timestamps: list


def resolve_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_epoch(path: Path) -> int:
    for pattern in [r"epoch[=_-]?(\d+)", r"rice-iou-(\d+)", r"(\d+)(?=\.ckpt$)"]:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return -1


def _select_checkpoint_in_dir(model_dir: Path) -> Path:
    """Priority: best.ckpt > highest-epoch *rice*iou*.ckpt > last.ckpt > any .ckpt."""
    best = model_dir / "best.ckpt"
    if best.exists():
        return best
    historical_best = sorted(model_dir.glob("*rice*iou*.ckpt"), key=_extract_epoch)
    if historical_best:
        return historical_best[-1]
    last = model_dir / "last.ckpt"
    if last.exists():
        return last
    any_ckpt = sorted(model_dir.glob("*.ckpt"))
    if any_ckpt:
        return any_ckpt[0]
    raise FileNotFoundError(f"No .ckpt checkpoint found in {model_dir}")


def discover_checkpoint(search_dirs=None, explicit_path=None) -> RiceCheckpointConfig:
    """Find a rice checkpoint directory (containing a .ckpt plus
    training_config.json + normalization_stats.json) and load its config.

    explicit_path may be a .ckpt file directly, or a directory to search
    within. Otherwise looks under search_dirs (default: ./checkpoints/rice,
    ./checkpoints, ./checkpoint), picking the most-recently-modified directory
    that has both required JSON config files.
    """
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        if not explicit_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {explicit_path}")
        model_dir = explicit_path.parent if explicit_path.is_file() else explicit_path
        checkpoint_path = explicit_path if explicit_path.is_file() else _select_checkpoint_in_dir(model_dir)
    else:
        search_dirs = search_dirs or [Path("checkpoints/rice"), Path("checkpoints"), Path("checkpoint")]
        candidate_dirs = []
        for d in search_dirs:
            d = Path(d)
            if not d.exists():
                continue
            for config_path in d.rglob("training_config.json"):
                if (config_path.parent / "normalization_stats.json").exists():
                    candidate_dirs.append(config_path.parent)

        candidate_dirs = sorted(set(candidate_dirs), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidate_dirs:
            raise FileNotFoundError(
                "No rice checkpoint found (expected a directory with a .ckpt, "
                "training_config.json, and normalization_stats.json under "
                f"{[str(d) for d in search_dirs]}). Pass checkpoint_path explicitly."
            )
        model_dir = candidate_dirs[0]
        checkpoint_path = _select_checkpoint_in_dir(model_dir)

    training_config_path = model_dir / "training_config.json"
    normalization_path = model_dir / "normalization_stats.json"
    for required_path in [training_config_path, normalization_path]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing {required_path.name} next to checkpoint {checkpoint_path} "
                f"(expected in {model_dir})"
            )

    import numpy as np

    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    normalization_stats = json.loads(normalization_path.read_text(encoding="utf-8"))

    band_names = list(normalization_stats["band_names"])
    band_means = np.asarray(normalization_stats["means"], dtype=np.float32)
    band_stds = np.asarray(normalization_stats["stds"], dtype=np.float32)
    if band_means.shape != (len(band_names),) or band_stds.shape != (len(band_names),):
        raise ValueError(f"{normalization_path}: means/stds length must match band_names")

    return RiceCheckpointConfig(
        checkpoint_path=checkpoint_path,
        band_names=band_names,
        band_means=band_means,
        band_stds=band_stds,
        recommended_threshold=float(training_config.get("recommended_probability_threshold", 0.5)),
        training_timestamps=[str(v) for v in training_config.get("timestamps", [])],
    )


def _disable_pretrained_flags(value):
    if isinstance(value, dict):
        for key in list(value.keys()):
            if key in {"backbone_pretrained", "pretrained_backbone"}:
                value[key] = False
            else:
                _disable_pretrained_flags(value[key])
    elif isinstance(value, list):
        for item in value:
            _disable_pretrained_flags(item)


def load_model(checkpoint_path, device, allow_hf_fallback=True):
    """Load a fine-tuned rice checkpoint. Tries offline reconstruction first
    (architecture read from the checkpoint's own hyper_parameters, no network
    needed); falls back to downloading pretrained TerraMind from Hugging Face
    only if that fails (e.g. an older checkpoint without embedded model_args).
    """
    try:
        from terratorch.tasks import SemanticSegmentationTask
    except ImportError as exc:
        raise ImportError("rice requires the optional 'rice' extra: pip install -e '.[rice]'") from exc

    offline_error = None
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        if "model_args" not in hparams:
            raise RuntimeError("Checkpoint has no hyper_parameters['model_args']")

        model_args = copy.deepcopy(hparams["model_args"])
        _disable_pretrained_flags(model_args)

        load_kwargs = {"model_args": model_args}
        if "model_factory" in hparams:
            load_kwargs["model_factory"] = hparams["model_factory"]

        del checkpoint
        gc.collect()

        model = SemanticSegmentationTask.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu", strict=True, **load_kwargs
        )
        model = model.to(device)
        model.eval()
        return model
    except Exception as error:
        offline_error = error

    if not allow_hf_fallback:
        raise RuntimeError("Offline model load failed and HF fallback is disabled") from offline_error

    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = SemanticSegmentationTask.load_from_checkpoint(str(checkpoint_path), map_location="cpu", strict=True)
    model = model.to(device)
    model.eval()
    return model
