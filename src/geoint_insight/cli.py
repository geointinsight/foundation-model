"""Top-level CLI: dispatches to a per-model subcommand, plus `setup` to fetch
checkpoints.

    geoint-insight setup
    geoint-insight sar --s1 path/to/S1.tif --output-dir outputs/

New models register themselves here by adding one line to _SUBCOMMAND_MODULES —
each model subpackage owns its own argument parsing (see
sar/cli.py:add_subparser) so adding a model never requires touching this
file's logic, only its registration list.
"""

import argparse
import sys

_SUBCOMMAND_MODULES = ["sar", "multisensor"]  # add new model package names here as they're added


def _load_subcommand(name):
    module = __import__(f"geoint_insight.{name}.cli", fromlist=["add_subparser"])
    return module


def _run_setup(args):
    from ._setup import download_checkpoints

    download_checkpoints(args.dest, args.force)
    return 0


def _add_setup_subparser(subparsers):
    parser = subparsers.add_parser(
        "setup", help="Download model checkpoints into ./checkpoints/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dest", default="checkpoints", help="Where to place checkpoint files")
    parser.add_argument("--force", action="store_true", help="Re-download even if checkpoints already exist")
    parser.set_defaults(func=_run_setup)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="geoint-insight",
        description="GEOINT Insight — free, lightweight flood-extent prediction toolkit (multi-model).",
    )
    subparsers = parser.add_subparsers(dest="model", required=True)
    _add_setup_subparser(subparsers)
    for name in _SUBCOMMAND_MODULES:
        _load_subcommand(name).add_subparser(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
