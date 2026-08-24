# geoint-insight

Free, lightweight geospatial AI toolkit, provided by
[GEOINT Insight](https://geoint-insight.com/foundation/). Organized by
capability area, each with one subpackage per model sharing a consistent
`predict_scene()` / `predict_folder()` API.

- [Flood Detection](#flood-detection) — available now
- Rice Classification — coming soon
- Building Detection — coming soon
- Road Detection — coming soon

## Install

This repo uses [Git LFS](https://git-lfs.github.com) to store model
checkpoints, so install that first (once per machine):

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
command. See each capability area below for what's bundled and what needs a
separate download.

---

## Flood Detection

Predicts probable flood extent from satellite imagery. Two models:

- **`sar`** — Sentinel-1 SAR only (Sen1Floods11 baseline, FCN-ResNet50).
  No Sentinel-2 imagery required.
- **`multisensor`** — Sentinel-1 + Sentinel-2 dual-modality (TerraMind
  foundation model). Both bands are required; needs the optional
  `multisensor` extra.

Each model is a lightweight subset of a larger internal pipeline. Not
included here:

- **Test-time augmentation** (averaging predictions over flipped/rotated views)
- **Adaptive spatial thresholding** (smoothly-varying threshold across one large scene)
- **Grid-mosaic support** — sharing one threshold and cleaning connected flood
  components across a whole grid of adjacent tiles. Without this, tiling a
  large area into a grid and mosaicking the results afterward may show slightly
  inconsistent flood extent right at tile boundaries.

Every scene is predicted independently, with a straightforward
prepare → infer → threshold → clean → export pipeline.

### Setup

The `sar` checkpoint (`checkpoints/sen1floods11_s1_baseline_fcn_resnet50.cp`)
and a small real Sentinel-1 sample scene are already included via Git LFS, so
you can try `sar` immediately without sourcing anything yourself — see
Quickstart below.

If you only have the checkpoint pointer (e.g. you cloned without Git LFS
installed first), run `git lfs pull` inside the repo to fetch the real file.

To use `multisensor`, install the extra and get its checkpoint separately
(it's ~1.2GB — too large for this repo's Git LFS quota, so it's distributed
via [Releases](https://github.com/geointinsight/foundation-model/releases)
instead). A bundled real S1+S2 sample pair is included, so `--sample` works
as soon as the checkpoint is in place:

```bash
pip install -e ".[multisensor]"
# download the TerraMind checkpoint from the Releases page above,
# then place it under ./checkpoints/
```

### Quickstart

```bash
geoint-insight sar --sample --output-dir outputs/
geoint-insight multisensor --sample --output-dir outputs/   # needs the multisensor extra + its checkpoint, see above
```

Runs a bundled real sample scene end to end and writes results under
`outputs/outputs/` (probability raster, mask, polygon, preview PNG — see
Output below).

### CLI

Every model is a required subcommand — there's no default model, since
different models expect different inputs.

```bash
geoint-insight sar --s1 path/to/S1.tif --output-dir outputs/
geoint-insight sar --s1 scene1.tif --s1 scene2.tif --device cpu
geoint-insight sar --s1 scene.tif --threshold 0.6 --no-auto-threshold

geoint-insight multisensor --s1 path/to/S1.tif --s2 path/to/S2.tif --output-dir outputs/
geoint-insight multisensor --sample --output-dir outputs/
```

Run `geoint-insight --help`, `geoint-insight sar --help`, or
`geoint-insight multisensor --help` for the full flag list.

### Python API

Either import a model's subpackage directly, or use the unified `predict()`,
which always requires naming the model explicitly:

```python
from geoint_insight.sar import predict_scene, predict_folder, sample_path

result = predict_scene(sample_path(), "outputs/")   # bundled sample, zero setup
print(result.flood_area_km2, result.threshold, result.polygon_path)

result = predict_scene("path/to/S1.tif", "outputs/")

# Batch over a folder, reusing one loaded model:
results = predict_folder("scenes/", "outputs/", pattern="*.tif")
```

```python
from geoint_insight.multisensor import predict_scene, sample_s1_path, sample_s2_path

result = predict_scene(sample_s1_path(), sample_s2_path(), "outputs/")   # bundled sample, zero setup
result = predict_scene("path/to/S1.tif", "path/to/S2.tif", "outputs/")
```

```python
from geoint_insight import predict

result = predict(model="sar", s1_path="path/to/S1.tif", output_dir="outputs/")
result = predict(model="multisensor", s1_path="path/to/S1.tif", s2_path="path/to/S2.tif", output_dir="outputs/")
```

`s1_path` must be a 1- or 2-band Sentinel-1 GeoTIFF (VV, or VV+VH). A single
band is treated as VV and VH is approximated from Sen1Floods11 training
statistics — expect lower accuracy than genuine VV+VH input. `s2_path` (for
`multisensor`) must be a 3-, 4-, or 13-band Sentinel-2 GeoTIFF — anything
short of the full 13 L1C bands is filled in with training-set band means and
should be treated as prototype quality.

### Output

Per scene, under `<output_dir>/outputs/`:

- `01_flood_probability.tif` — per-pixel flood probability (float32, NaN = no data)
- `02_flood_mask_raw.tif` / `03_flood_mask_clean.tif` — thresholded masks before/after cleanup
- `04_probable_flood_polygons.gpkg` — vectorized flood extent
- `05_flood_prediction_preview.png` — 3-panel preview (SAR / probability / overlay)

Results are a **probable flood extent prototype**, not a validated flood
product — SAR alone cannot always distinguish real floodwater from other flat,
wet, or smooth surfaces (wet soil, freshly-plowed fields, etc).

---

## Rice Classification

Coming soon.

## Building Detection

Coming soon.

## Road Detection

Coming soon.

---

## Package layout (for adding a new model)

```
src/geoint_insight/
  __init__.py       top-level predict(model=..., ...) unified entry point
  _common/          shared helpers (geo/CRS, tiling, SAR dB conversion,
                    thresholding, mask cleanup, export) — model-agnostic
  sar/               S1-only flood model: preprocessing, model, inference, pipeline, CLI
  multisensor/       S1+S2 flood model: same shape as sar
    cli.py            add_subparser(subparsers) registers `geoint-insight multisensor ...`
  cli.py            top-level CLI dispatcher — new models register by adding
                    one line to _SUBCOMMAND_MODULES
```

A new model subpackage (flood detection or a future capability area) should
expose the same shape as `sar`/`multisensor`: `predict_scene()`,
`predict_folder()`, `load_model()`, a `SceneResult` dataclass, and a `cli.py`
with `add_subparser(subparsers)` — then register its package name in both
`geoint_insight/cli.py`'s `_SUBCOMMAND_MODULES` and
`geoint_insight/__init__.py`'s `AVAILABLE_MODELS` (the latter is what
`predict(model=...)` checks against). Put anything genuinely model-agnostic
(tiling, generic raster helpers, thresholding/cleaning/export) in `_common/`
rather than duplicating it per model.

## About

`geoint-insight` is provided free and open-source by
[GEOINT Insight](https://geoint-insight.com/foundation/). Licensed under MIT —
see [LICENSE](LICENSE).
