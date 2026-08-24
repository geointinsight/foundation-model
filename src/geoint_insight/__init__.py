"""geoint_insight — free, lightweight flood-extent prediction toolkit, provided
by GEOINT Insight (https://geoint-insight.com/foundation/).

Organized as one subpackage per model. Import a model's subpackage directly:

    from geoint_insight.sentinel1 import predict_scene, sample_path
    result = predict_scene(sample_path(), "outputs/")

...or use the unified top-level predict(), which always requires naming the
model explicitly (since which model applies is never implicit — different
models expect different inputs and produce different SceneResult shapes):

    from geoint_insight import predict
    result = predict(model="sentinel1", s1_path="path/to/S1.tif", output_dir="outputs/")

Or from the command line:

    geoint-insight sentinel1 --sample --output-dir outputs/

More models (e.g. Sentinel-2, multi-modal) are added over time as additional
subpackages alongside geoint_insight.sentinel1 — see geoint_insight/cli.py for
how a new model registers its own CLI subcommand, and AVAILABLE_MODELS below
for what predict(model=...) currently accepts.
"""

import importlib

__version__ = "0.1.0"

AVAILABLE_MODELS = ["sentinel1"]

__all__ = ["__version__", "AVAILABLE_MODELS", "predict"]


def predict(model, *args, **kwargs):
    """Unified entry point across all models — model is required (not
    defaulted) because it determines what kind of input is expected and what
    kind of result comes back; there's no sensible implicit choice. Equivalent
    to calling predict_scene() on that model's own subpackage directly, e.g.
    predict(model="sentinel1", ...) == geoint_insight.sentinel1.predict_scene(...).
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model {model!r}. Available models: {AVAILABLE_MODELS}")
    module = importlib.import_module(f"geoint_insight.{model}")
    return module.predict_scene(*args, **kwargs)
