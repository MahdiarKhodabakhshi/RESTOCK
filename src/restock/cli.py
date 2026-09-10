import argparse
from itertools import pairwise
from pathlib import Path

import httpx
from graphql import build_schema

from restock.saleor.build import build_store
from restock.saleor.runtime import Runtime
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


def _saleor_build(args: argparse.Namespace) -> int:
    build_store(Runtime(args.data), args.versions, seed=args.seed)
    return 0


def _saleor_up(args: argparse.Namespace) -> int:
    runtime = Runtime(args.data)
    runtime.start_postgres()
    for version in args.versions:
        print(f"{version}  {runtime.start_api(version, args.workers)}")
    return 0


def _saleor_down(args: argparse.Namespace) -> int:
    runtime = Runtime(args.data)
    if args.versions:
        for version in args.versions:
            runtime.stop_api(version)
        return 0
    for name in runtime.running():
        if name.startswith("api_"):
            runtime.stop_api(name.removeprefix("api_"))
    runtime.stop_postgres()
    return 0


def _saleor_status(args: argparse.Namespace) -> int:
    for name in Runtime(args.data).running():
        print(name)
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

    saleor = commands.add_parser("saleor", help="run Saleor releases locally")
    saleor.add_argument("--data", type=Path, default=Path("data"), help="runtime directory")
    actions = saleor.add_subparsers(dest="action", required=True)

    build = actions.add_parser("build", help="create the store and upgrade it through releases")
    build.add_argument("versions", nargs="+", metavar="VERSION", help="releases, oldest first")
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(run=_saleor_build)

    up = actions.add_parser("up", help="serve releases that have been built")
    up.add_argument("versions", nargs="+", metavar="VERSION")
    up.add_argument("--workers", type=int, default=4)
    up.set_defaults(run=_saleor_up)

    down = actions.add_parser("down", help="stop the given releases, or everything")
    down.add_argument("versions", nargs="*", metavar="VERSION")
    down.set_defaults(run=_saleor_down)

    status = actions.add_parser("status", help="list running processes")
    status.set_defaults(run=_saleor_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
