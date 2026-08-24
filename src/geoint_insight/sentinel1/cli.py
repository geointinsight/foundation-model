"""sentinel1 subcommand: geoint-insight sentinel1 --s1 path/to/S1.tif --output-dir outputs/

Also runnable standalone (kept working for anyone scripting directly against
this subpackage): python -m geoint_insight.sentinel1.cli --s1 ...
"""

import json
import sys
from pathlib import Path

import pandas as pd

from .._common.postprocess import AUTO_THRESHOLD_MULTIPLIER, FLOOD_THRESHOLD, MAX_HOLE_PIXELS, MIN_COMPONENT_PIXELS, MIN_POLYGON_AREA_M2
from ._data import sample_path
from ._inference import PATCH_SIZE, STRIDE
from .pipeline import load_model, predict_scene

HELP = "Sentinel-1 SAR flood-extent prediction (Sen1Floods11 baseline)"


def add_arguments(parser):
    """Add this model's flags to an argparse parser (used both for the
    `geoint-insight sentinel1 ...` subcommand and for standalone `python -m
    geoint_insight.sentinel1.cli` use)."""
    parser.add_argument("--s1", type=Path, action="append", help="Path to a Sentinel-1 GeoTIFF (VV/VH). Repeatable for multiple scenes.")
    parser.add_argument("--sample", action="store_true", help="Use the small bundled sample scene instead of --s1 (quick way to try the tool)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write results")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to the Sen1Floods11 .cp checkpoint (default: auto-discover under ./checkpoints/)")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None, help="Force a device")
    parser.add_argument("--threshold", type=float, default=FLOOD_THRESHOLD, help="Fixed threshold (used with --no-auto-threshold)")
    parser.add_argument("--no-auto-threshold", action="store_true", help="Use --threshold instead of per-scene Otsu auto-thresholding")
    parser.add_argument("--auto-threshold-multiplier", type=float, default=AUTO_THRESHOLD_MULTIPLIER, help="Scale factor applied to the Otsu threshold")
    parser.add_argument("--speckle-filter-size", type=int, default=3, help="Median filter size for speckle reduction (0 disables)")
    parser.add_argument("--edge-discount-px", type=int, default=24, help="Fade probability within N px of the scene border (0 disables)")
    parser.add_argument("--min-component-pixels", type=int, default=MIN_COMPONENT_PIXELS)
    parser.add_argument("--max-hole-pixels", type=int, default=MAX_HOLE_PIXELS)
    parser.add_argument("--min-polygon-area-m2", type=float, default=MIN_POLYGON_AREA_M2)
    parser.add_argument("--area-crs", default=None, help="EPSG code for area calculations (default: auto-detect UTM zone)")
    parser.add_argument("--no-crop-nodata", action="store_true", help="Skip auto-cropping to the valid-data extent")
    parser.add_argument("--no-preview", action="store_true", help="Skip PNG preview generation")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.set_defaults(func=run)
    return parser


def add_subparser(subparsers):
    """Register this model as a `geoint-insight sentinel1 ...` subcommand."""
    parser = subparsers.add_parser("sentinel1", help=HELP, formatter_class=__import__("argparse").ArgumentDefaultsHelpFormatter)
    return add_arguments(parser)


def run(args):
    s1_paths = list(args.s1) if args.s1 else []
    if args.sample:
        s1_paths.append(sample_path())
    if not s1_paths:
        print("No input given — pass --s1 path/to/S1.tif, or --sample to try the bundled example.", file=sys.stderr)
        return 1

    model, device, checkpoint_path = load_model(args.checkpoint, args.device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    results = []
    for s1_path in s1_paths:
        print(f"\n=== {s1_path} ===")
        result = predict_scene(
            s1_path,
            args.output_dir,
            model=model,
            device=device,
            threshold=args.threshold,
            auto_threshold=not args.no_auto_threshold,
            auto_threshold_multiplier=args.auto_threshold_multiplier,
            speckle_filter_size=args.speckle_filter_size,
            edge_discount_px=args.edge_discount_px,
            min_component_pixels=args.min_component_pixels,
            max_hole_pixels=args.max_hole_pixels,
            min_polygon_area_m2=args.min_polygon_area_m2,
            area_crs=args.area_crs,
            crop_nodata=not args.no_crop_nodata,
            save_preview=not args.no_preview,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        print(f"Threshold: {result.threshold:.3f} ({result.threshold_mode})")
        print(f"Flood area: {result.flood_area_km2:.3f} km²  ({result.polygon_count} polygons)")
        results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [r.__dict__ for r in results]
    summary_csv = args.output_dir / "summary.csv"
    summary_json = args.output_dir / "summary.json"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSummary: {summary_csv}")
    return 0


def main(argv=None):
    """Standalone entry point (python -m geoint_insight.sentinel1.cli)."""
    import argparse

    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
