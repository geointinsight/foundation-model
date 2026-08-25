"""geoint_insight.rice — multitemporal Sentinel-2 rice-extent prediction
(TerraMind foundation model, fine-tuned per dataset).

Requires the optional 'rice' extra: pip install -e ".[rice]"

Unlike sar/multisensor, a rice checkpoint is dataset-specific and ships as a
directory (checkpoint + training_config.json + normalization_stats.json), not
a single portable file — see ._model.discover_checkpoint. Fetch it with
`geoint-insight setup --rice` (downloads into ./checkpoints/rice/).

A small bundled sample (one real Sentinel-2 crop, Prathumthanee, Thailand,
single timestamp) lets you confirm the pipeline runs with zero setup beyond
the checkpoint itself — but since it's only one date, not the multitemporal
stack the model needs, expect it to detect ~0 rice even over visibly green
paddy fields. It proves mechanics, not detection quality — see ._data for why.

    from geoint_insight.rice import predict_scene, load_model, sample_stack_paths
    model, device, config = load_model()  # auto-discovers ./checkpoints/rice/
    result = predict_scene(sample_stack_paths(), "outputs/", model=model, device=device, config=config)

    from geoint_insight.rice import predict_stacks_folder
    result = predict_stacks_folder("stacks_10m/", "outputs/", checkpoint_path="path/to/model_dir")
"""

from ._data import sample_stack_paths
from .pipeline import SceneResult, load_model, predict_scene, predict_stacks_folder

__all__ = ["predict_scene", "predict_stacks_folder", "load_model", "SceneResult", "sample_stack_paths"]
