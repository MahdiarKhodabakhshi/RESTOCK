import argparse
from itertools import pairwise
from pathlib import Path

import httpx
from graphql import build_schema

from restock.schema.diff import diff_schemas, query_density
from restock.schema.fetch import DEFAULT_ROOT, fetch_schema, load_schema


def _fetch(args: argparse.Namespace) -> int:
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for version in args.versions:
            path = fetch_schema(version, args.root, client=client, force=args.force)
            print(f"{version}  {path}")
    return 0


def _diff(args: argparse.Namespace) -> int:
    schemas = [build_schema(load_schema(version, args.root)) for version in args.versions]
    print(
        f"{'from':<8} {'to':<8} {'fields':>7} {'broken':>7} {'share':>7} {'deprec':>7} {'added':>6}"
    )
    for (old_version, old), (new_version, new) in pairwise(
        zip(args.versions, schemas, strict=True)
    ):
        changes = diff_schemas(old, new)
        density = query_density(old, changes)
        print(
            f"{old_version:<8} {new_version:<8} {density.fields:>7} {density.broken:>7} "
            f"{density.broken_share:>7.2%} {density.deprecated:>7} {density.added:>6}"
        )
        if args.verbose:
            for change in sorted(changes, key=lambda c: c.coordinate):
                if change.breaking and change.query_side:
                    print(f"    {change.kind:<28} {change.coordinate}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="restock")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="directory holding downloaded schemas"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download schemas for Saleor release tags")
    fetch.add_argument("versions", nargs="+", metavar="VERSION")
    fetch.add_argument("--force", action="store_true", help="download even if already on disk")
    fetch.set_defaults(run=_fetch)

    diff = commands.add_parser(
        "diff", help="report how much of the query side changes between consecutive versions"
    )
    diff.add_argument("versions", nargs="+", metavar="VERSION")
    diff.add_argument("-v", "--verbose", action="store_true", help="list each breaking change")
    diff.set_defaults(run=_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
