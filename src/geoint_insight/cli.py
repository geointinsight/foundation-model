"""Top-level CLI: dispatches to a per-model subcommand.

    geoint-insight sentinel1 --s1 path/to/S1.tif --output-dir outputs/

New models register themselves here by adding one line to _SUBCOMMAND_MODULES —
each model subpackage owns its own argument parsing (see
sentinel1/cli.py:add_subparser) so adding a model never requires touching this
file's logic, only its registration list.
"""

import argparse
import sys

_SUBCOMMAND_MODULES = ["sentinel1"]  # add new model package names here as they're added


def _load_subcommand(name):
    module = __import__(f"geoint_insight.{name}.cli", fromlist=["add_subparser"])
    return module


def build_parser():
    parser = argparse.ArgumentParser(
        prog="geoint-insight",
        description="GEOINT Insight — free, lightweight flood-extent prediction toolkit (multi-model).",
    )
    subparsers = parser.add_subparsers(dest="model", required=True)
    for name in _SUBCOMMAND_MODULES:
        _load_subcommand(name).add_subparser(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
