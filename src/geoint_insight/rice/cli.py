"""rice subcommand: geoint-insight rice --stacks 2026-03-13.tif 2026-03-26.tif --output-dir outputs/
or geoint-insight rice --stack-dir stacks_10m/ --output-dir outputs/

Also runnable standalone: python -m geoint_insight.rice.cli --stack-dir ...

Unlike sar/multisensor, there's no bundled sample yet — the rice model needs a
checkpoint directory (checkpoint + training_config.json + normalization_stats.json)
and a matching multitemporal Sentinel-2 stack that must be supplied explicitly.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from ._inference import MIN_PATCH_VALID_RATIO, PATCH_SIZE, STRIDE
from ._postprocess import MIN_OBJECT_AREA_M2
from .pipeline import load_model, predict_scene, predict_stacks_folder

HELP = "Multitemporal Sentinel-2 rice-extent prediction (TerraMind)"


def add_arguments(parser):
    parser.add_argument("--stacks", type=Path, nargs="+", default=None, help="Explicit ordered list of per-date stack GeoTIFFs (same grid, one per timestamp)")
    parser.add_argument("--stack-dir", type=Path, default=None, help="Directory of per-date stack GeoTIFFs to auto-discover instead of --stacks")
    parser.add_argument("--stack-glob", default="*STACK*.tif", help="Glob used with --stack-dir")
    parser.add_argument("--valid-mask-dir", type=Path, default=None, help="Optional directory of per-date valid-pixel masks (matched by date)")
    parser.add_argument("--valid-mask-glob", default="*.tif", help="Glob used with --valid-mask-dir")
    parser.add_argument("--timestamps", nargs="+", default=None, help="Restrict --stack-dir discovery to these YYYYMMDD dates (default: checkpoint's training_config.json timestamps, else every date found)")
    parser.add_argument("--scene-id", default=None, help="Scene identifier for output filenames (default: first stack's filename stem)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write results")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a rice .ckpt, or its containing directory (default: auto-discover under ./checkpoints/)")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None, help="Force a device")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed probability threshold (default: checkpoint's recommended_probability_threshold)")
    parser.add_argument("--min-patch-valid-ratio", type=float, default=MIN_PATCH_VALID_RATIO)
    parser.add_argument("--no-sieve", action="store_true", help="Skip sieving out small rice objects")
    parser.add_argument("--min-object-area-m2", type=float, default=MIN_OBJECT_AREA_M2)
    parser.add_argument("--export-vector", action="store_true", help="Also write a GeoPackage of rice polygons")
    parser.add_argument("--simplify-tolerance-m", type=float, default=5.0)
    parser.add_argument("--area-crs", default=None, help="EPSG code for polygon area calculations (default: keep the raster's own CRS)")
    parser.add_argument("--no-preview", action="store_true", help="Skip PNG preview generation")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.set_defaults(func=run)
    return parser


def add_subparser(subparsers):
    """Register this model as a `geoint-insight rice ...` subcommand."""
    parser = subparsers.add_parser("rice", help=HELP, formatter_class=__import__("argparse").ArgumentDefaultsHelpFormatter)
    return add_arguments(parser)


def run(args):
    if not args.stacks and not args.stack_dir:
        print("No input given — pass --stacks (ordered list of files) or --stack-dir.", file=sys.stderr)
        return 1

    model, device, config = load_model(args.checkpoint, args.device)
    print(f"Loaded checkpoint: {config.checkpoint_path}")
    print(f"Device: {device}")
    print(f"Bands: {config.band_names}")

    predict_kwargs = dict(
        model=model,
        device=device,
        config=config,
        scene_id=args.scene_id,
        threshold=args.threshold,
        min_patch_valid_ratio=args.min_patch_valid_ratio,
        do_sieve=not args.no_sieve,
        min_object_area_m2=args.min_object_area_m2,
        export_vector=args.export_vector,
        simplify_tolerance_m=args.simplify_tolerance_m,
        area_crs=args.area_crs,
        save_preview=not args.no_preview,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    if args.stacks:
        print(f"\n=== Stacks: {[str(p) for p in args.stacks]} ===")
        result = predict_scene(args.stacks, args.output_dir, **predict_kwargs)
    else:
        print(f"\n=== Stack dir: {args.stack_dir} ===")
        result = predict_stacks_folder(
            args.stack_dir, args.output_dir,
            stack_glob=args.stack_glob,
            valid_mask_dir=args.valid_mask_dir,
            valid_mask_glob=args.valid_mask_glob,
            timestamps=args.timestamps,
            **predict_kwargs,
        )

    print(f"Threshold: {result.threshold:.3f}")
    print(f"Rice area: {result.rice_area_rai:.1f} rai  ({result.rice_area_km2:.3f} km²)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "summary.csv"
    summary_json = args.output_dir / "summary.json"
    pd.DataFrame([result.__dict__]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps([result.__dict__], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSummary: {summary_csv}")
    return 0


def main(argv=None):
    """Standalone entry point (python -m geoint_insight.rice.cli)."""
    import argparse

    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
