"""geoint_insight.rice — multitemporal Sentinel-2 rice-extent prediction
(TerraMind foundation model, fine-tuned per dataset).

Requires the optional 'rice' extra: pip install -e ".[rice]"

Unlike sar/multisensor, a rice checkpoint is dataset-specific and ships as a
directory (checkpoint + training_config.json + normalization_stats.json), not
a single portable file — see ._model.discover_checkpoint. There's no bundled
sample yet; you need your own checkpoint directory and matching multitemporal
Sentinel-2 stack.

    from geoint_insight.rice import predict_scene, load_model
    model, device, config = load_model(checkpoint_path="path/to/model_dir")
    result = predict_scene(["2026-03-13_STACK.tif", "2026-03-26_STACK.tif"], "outputs/",
                            model=model, device=device, config=config)

    from geoint_insight.rice import predict_stacks_folder
    result = predict_stacks_folder("stacks_10m/", "outputs/", checkpoint_path="path/to/model_dir")
"""

from .pipeline import SceneResult, load_model, predict_scene, predict_stacks_folder

__all__ = ["predict_scene", "predict_stacks_folder", "load_model", "SceneResult"]
