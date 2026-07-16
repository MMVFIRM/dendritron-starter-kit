"""Command-line interface for demos and smoke verification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from . import __version__
from .benchmarks import run_smoke_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dendritron", description="Dendritron starter kit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run fast dependency-light benchmarks")
    smoke.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    subparsers.add_parser("info", help="show the architecture layers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print("Dendritron primitive → owned tissue → mixed geometry → functional memory registry")
        print("Archived research benchmarks live under benchmarks/archive.")
        return 0
    results = run_smoke_suite()
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status:4}  {name:12}  {result}")
    return 0 if all(result["passed"] for result in results.values()) else 1
