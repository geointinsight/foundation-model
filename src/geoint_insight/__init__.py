"""geoint_insight — free, lightweight flood-extent prediction toolkit, provided
by GEOINT Insight (https://geoint-insight.com/foundation/).

Organized as one subpackage per model:

  - sar          — Sentinel-1 SAR only (Sen1Floods11 baseline)
  - multisensor  — Sentinel-1 + Sentinel-2 dual-modality (TerraMind; requires
                   the optional 'multisensor' extra, pip install -e ".[multisensor]")

Import a model's subpackage directly:

    from geoint_insight.sar import predict_scene, sample_path
    result = predict_scene(sample_path(), "outputs/")

    from geoint_insight.multisensor import predict_scene
    result = predict_scene("S1.tif", "S2.tif", "outputs/")

...or use the unified top-level predict(), which always requires naming the
model explicitly (since which model applies is never implicit — different
models expect different inputs and produce different SceneResult shapes):

    from geoint_insight import predict
    result = predict(model="sar", s1_path="path/to/S1.tif", output_dir="outputs/")

Or from the command line:

    geoint-insight sar --sample --output-dir outputs/
    geoint-insight multisensor --s1 S1.tif --s2 S2.tif --output-dir outputs/

More models are added over time as additional subpackages — see
geoint_insight/cli.py for how a new model registers its own CLI subcommand,
and AVAILABLE_MODELS below for what predict(model=...) currently accepts.
"""

import importlib

__version__ = "0.1.0"

AVAILABLE_MODELS = ["sar", "multisensor"]

__all__ = ["__version__", "AVAILABLE_MODELS", "predict"]


def predict(model, *args, **kwargs):
    """Unified entry point across all models — model is required (not
    defaulted) because it determines what kind of input is expected and what
    kind of result comes back; there's no sensible implicit choice. Equivalent
    to calling predict_scene() on that model's own subpackage directly, e.g.
    predict(model="sar", ...) == geoint_insight.sar.predict_scene(...).
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model {model!r}. Available models: {AVAILABLE_MODELS}")
    module = importlib.import_module(f"geoint_insight.{model}")
    return module.predict_scene(*args, **kwargs)
