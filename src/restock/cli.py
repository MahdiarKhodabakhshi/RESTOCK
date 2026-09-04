import argparse
from pathlib import Path

import httpx

from restock.schema.fetch import DEFAULT_ROOT, fetch_schema


def _fetch(args: argparse.Namespace) -> int:
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for version in args.versions:
            path = fetch_schema(version, args.root, client=client, force=args.force)
            print(f"{version}  {path}")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
