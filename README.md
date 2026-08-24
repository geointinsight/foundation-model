# geoint-insight

Free, lightweight flood-extent prediction toolkit, provided by
[GEOINT Insight](https://geoint-insight.com/foundation/). Organized as one
subpackage per model — the first one is **`sentinel1`**, predicting flood
extent from Sentinel-1 SAR alone using a Sen1Floods11-style S1-only baseline
model (FCN-ResNet50). No Sentinel-2 imagery required. More models are added
over time as additional subpackages alongside `geoint_insight.sentinel1`.

Each model subpackage is a lightweight subset of a larger internal pipeline.
For `sentinel1`, not included here:

- **Test-time augmentation** (averaging predictions over flipped/rotated views)
- **Adaptive spatial thresholding** (smoothly-varying threshold across one large scene)
- **Grid-mosaic support** — sharing one threshold and cleaning connected flood
  components across a whole grid of adjacent tiles. Without this, tiling a
  large area into a grid and mosaicking the results afterward may show slightly
  inconsistent flood extent right at tile boundaries.

Every scene is predicted independently, with a straightforward
prepare → infer → threshold → clean → export pipeline.

## Install

This repo uses [Git LFS](https://git-lfs.github.com) to store the model
checkpoint, so install that first (once per machine):

```bash
brew install git-lfs      # macOS; see git-lfs.github.com for other platforms
git lfs install
```

Then clone and install:

```bash
git clone https://github.com/geointinsight/foundation-model.git
cd foundation-model
pip install -e .
```

This installs the `geoint_insight` Python package and a `geoint-insight` CLI
command. The checkpoint (`checkpoints/sen1floods11_s1_baseline_fcn_resnet50.cp`)
and a small real Sentinel-1 sample scene are already included, so you can try
it immediately without sourcing anything yourself — see Quickstart below.

If you only have the checkpoint pointer (e.g. you cloned without Git LFS
installed first), run `git lfs pull` inside the repo to fetch the real file.

## Quickstart

```bash
geoint-insight sentinel1 --sample --output-dir outputs/
```

Runs the bundled sample scene end to end and writes results under
`outputs/outputs/` (probability raster, mask, polygon, preview PNG — see
Output below).

## CLI

Every model is a required subcommand — there's no default model, since
different models expect different inputs.

```bash
geoint-insight sentinel1 --s1 path/to/S1.tif --output-dir outputs/
geoint-insight sentinel1 --s1 scene1.tif --s1 scene2.tif --device cpu
geoint-insight sentinel1 --s1 scene.tif --threshold 0.6 --no-auto-threshold
```

Run `geoint-insight --help` or `geoint-insight sentinel1 --help` for the full
flag list.

## Python API

Either import a model's subpackage directly, or use the unified `predict()`,
which always requires naming the model explicitly:

```python
from geoint_insight.sentinel1 import predict_scene, predict_folder, sample_path

result = predict_scene(sample_path(), "outputs/")   # bundled sample, zero setup
print(result.flood_area_km2, result.threshold, result.polygon_path)

result = predict_scene("path/to/S1.tif", "outputs/")

# Batch over a folder, reusing one loaded model:
results = predict_folder("scenes/", "outputs/", pattern="*.tif")
```

```python
from geoint_insight import predict

result = predict(model="sentinel1", s1_path="path/to/S1.tif", output_dir="outputs/")
```

`s1_path` must be a 1- or 2-band Sentinel-1 GeoTIFF (VV, or VV+VH). A single
band is treated as VV and VH is approximated from Sen1Floods11 training
statistics — expect lower accuracy than genuine VV+VH input.

## Output

Per scene, under `<output_dir>/outputs/`:

- `01_flood_probability.tif` — per-pixel flood probability (float32, NaN = no data)
- `02_flood_mask_raw.tif` / `03_flood_mask_clean.tif` — thresholded masks before/after cleanup
- `04_probable_flood_polygons.gpkg` — vectorized flood extent
- `05_flood_prediction_preview.png` — 3-panel preview (SAR / probability / overlay)

Results are a **probable flood extent prototype**, not a validated flood
product — SAR alone cannot always distinguish real floodwater from other flat,
wet, or smooth surfaces (wet soil, freshly-plowed fields, etc).

## Package layout (for adding a new model)

```
src/geoint_insight/
  __init__.py       top-level predict(model=..., ...) unified entry point
  _common/          shared helpers (geo/CRS, thresholding, mask cleanup, export) — model-agnostic
  sentinel1/         this model: preprocessing, model loading, inference, pipeline, CLI
    cli.py            add_subparser(subparsers) registers `geoint-insight sentinel1 ...`
  cli.py            top-level CLI dispatcher — new models register by adding
                    one line to _SUBCOMMAND_MODULES
```

A new model subpackage should expose the same shape as `sentinel1`:
`predict_scene()`, `predict_folder()`, `load_model()`, a `SceneResult`
dataclass, and a `cli.py` with `add_subparser(subparsers)` — then register its
package name in both `geoint_insight/cli.py`'s `_SUBCOMMAND_MODULES` and
`geoint_insight/__init__.py`'s `AVAILABLE_MODELS` (the latter is what
`predict(model=...)` checks against).

## About

`geoint-insight` is provided free and open-source by
[GEOINT Insight](https://geoint-insight.com/foundation/). Licensed under MIT —
see [LICENSE](LICENSE).
