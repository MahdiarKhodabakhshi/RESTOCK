"""The ``restock eval`` command."""

import argparse
from pathlib import Path

from graphql import build_schema

from restock.agent.evaluate import evaluate, format_report
from restock.agent.model import Student
from restock.agent.policies import AlwaysRead, MemorisedOnly, Selective
from restock.agent.sft import write_always_read_sft, write_sft
from restock.saleor.client import SaleorClient
from restock.saleor.runtime import Runtime, api_url
from restock.schema.diff import diff_schemas
from restock.schema.fetch import fetch_schema, load_schema
from restock.tasks.cli import served_version
from restock.tasks.task import read_tasks

POLICIES = {
    "always_read": AlwaysRead,
    "memorised": MemorisedOnly,
    "selective": Selective,
}


def add_commands(commands: argparse._SubParsersAction) -> None:
    eval_cmd = commands.add_parser("eval", help="score a baseline on a task set")
    eval_cmd.add_argument("tasks", type=Path, help="JSONL task file")
    eval_cmd.add_argument(
        "--policy", choices=sorted(POLICIES), required=True, help="baseline to run"
    )
    eval_cmd.add_argument("--release", required=True, help="running Saleor release, e.g. 3.20")
    eval_cmd.add_argument("--trained-on", default=None, help="release the weights memorised")
    eval_cmd.add_argument(
        "--student",
        nargs="+",
        metavar="URL",
        required=True,
        help="OpenAI-compatible base URL(s) for the student",
    )
    eval_cmd.add_argument("--model", default="student", help="served model name")
    eval_cmd.add_argument("--data", type=Path, default=Path("data"), help="runtime directory")
    eval_cmd.add_argument("--limit", type=int, default=None, help="evaluate at most N tasks")
    eval_cmd.set_defaults(run=_eval)

    sft = commands.add_parser("sft", help="write supervised chat examples for a baseline")
    sft.add_argument("tasks", type=Path, help="JSONL task file")
    sft.add_argument("--out", type=Path, required=True, help="output chat JSONL")
    sft.add_argument("--version", default="3.20", help="API version the examples claim to target")
    sft.add_argument(
        "--with-schema",
        action="store_true",
        help="always-read examples: include a keyword schema slice in each prompt",
    )
    sft.add_argument(
        "--schema-version",
        default=None,
        help="schema tag to load for --with-schema (default: --version as X.Y.0)",
    )
    sft.set_defaults(run=_sft)


def _eval(args: argparse.Namespace) -> int:
    client = SaleorClient(api_url(args.release), Runtime(args.data).token(), timeout=120)
    version = served_version(client)
    fetch_schema(version, args.root)
    schema = build_schema(load_schema(version, args.root))

    changes = ()
    if args.trained_on:
        fetch_schema(args.trained_on, args.root)
        old = build_schema(load_schema(args.trained_on, args.root))
        changes = tuple(diff_schemas(old, schema))

    tasks = read_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    policy = POLICIES[args.policy]()
    student = Student(args.student, args.model)
    try:
        report = evaluate(
            policy,
            student,
            schema,
            client,
            tasks,
            release=args.release,
            trained_on=args.trained_on,
            changes=changes,
            signal_version=version,
            log=lambda message: print(message, flush=True),
        )
    finally:
        student.close()
        client.close()

    print(format_report(report))
    return 0


def _sft(args: argparse.Namespace) -> int:
    tasks = read_tasks(args.tasks)
    if args.with_schema:
        tag = args.schema_version or (
            args.version if args.version.count(".") >= 2 else f"{args.version}.0"
        )
        fetch_schema(tag, args.root)
        sdl = load_schema(tag, args.root)
        schema = build_schema(sdl)
        count = write_always_read_sft(
            args.out, tasks, schema, version=args.version, sdl=sdl, mutate_prob=0.5
        )
    else:
        count = write_sft(args.out, tasks, version=args.version)
    print(f"{count} examples -> {args.out}")
    return 0
