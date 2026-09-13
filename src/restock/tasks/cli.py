"""The ``restock tasks`` commands."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from graphql import GraphQLSchema, build_schema

from restock.saleor.client import SaleorClient
from restock.saleor.runtime import Runtime, api_url
from restock.schema.diff import diff_schemas
from restock.schema.fetch import fetch_schema, load_schema
from restock.tasks.generate import Generator, Settings
from restock.tasks.port import port
from restock.tasks.split import assign_splits
from restock.tasks.stats import summary
from restock.tasks.task import read_tasks, write_tasks
from restock.tasks.teacher import Teacher
from restock.tasks.verify import verify

TASKS_FILE = "tasks.jsonl"
REJECTED_FILE = "rejected.jsonl"


def served_version(client: SaleorClient) -> str:
    """The exact version a server runs, such as ``3.20.119``."""
    result = client.execute("{ shop { version } }")
    if not result.ok:
        raise RuntimeError("; ".join(result.messages))
    return result.data["shop"]["version"]


def connect(args: argparse.Namespace, release: str) -> tuple[SaleorClient, GraphQLSchema]:
    """Client for a running release and the schema of the exact version it serves."""
    client = SaleorClient(api_url(release), Runtime(args.data).token(), timeout=120)
    version = served_version(client)
    fetch_schema(version, args.root)
    return client, build_schema(load_schema(version, args.root))


def _teacher(args: argparse.Namespace) -> Teacher:
    return Teacher(args.teacher, args.model)


def _log(message: str) -> None:
    # Flushed, so that progress shows when the output goes to a file.
    print(message, flush=True)


def _generate(args: argparse.Namespace) -> int:
    client, schema = connect(args, args.base)
    _, target_schema = connect(args, args.target)
    settings = Settings(
        requests=args.requests, pairs=args.pairs, workers=args.workers, seed=args.seed
    )
    generator = Generator(
        schema,
        diff_schemas(schema, target_schema),
        client,
        _teacher(args),
        version=args.base,
        settings=settings,
        log=_log,
    )
    path = args.out / TASKS_FILE
    if path.exists():
        generator.add(read_tasks(path))
        print(f"continuing from {len(generator.tasks)} tasks in {path}")

    tasks = generator.run(on_round=lambda tasks: write_tasks(path, tasks))
    with (args.out / REJECTED_FILE).open("a", encoding="utf-8") as file:
        for rejected in generator.rejected:
            file.write(json.dumps(asdict(rejected), ensure_ascii=False) + "\n")

    report = generator.report
    print(
        f"{report.prompts} prompts ({report.failed_prompts} failed), {report.pairs} pairs: "
        f"{report.accepted} accepted, {report.duplicates} duplicates, "
        f"rejected {dict(report.rejections.most_common())}"
    )
    print(f"{len(tasks)} tasks in {path}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    client, schema = connect(args, args.base)
    path = args.out / TASKS_FILE
    tasks = read_tasks(path)
    outcomes = verify(
        tasks,
        schema,
        client,
        _teacher(args),
        version=args.base,
        workers=args.workers,
        log=_log,
        checkpoint=lambda: write_tasks(path, tasks),
    )
    write_tasks(path, tasks)
    print(f"verified {sum(outcomes.values())} tasks: {dict(outcomes.most_common())}")
    return 0


def _port(args: argparse.Namespace) -> int:
    client, schema = connect(args, args.to)
    path = args.out / TASKS_FILE
    tasks = read_tasks(path)
    # A reference on the later release is only needed where a model is evaluated.
    chosen = tasks if args.all else [task for task in tasks if task.split == "test"]
    outcomes = port(
        chosen,
        schema,
        client,
        _teacher(args),
        old=args.base,
        new=args.to,
        workers=args.workers,
        log=_log,
        checkpoint=lambda: write_tasks(path, tasks),
    )
    write_tasks(path, tasks)
    print(f"settled {sum(outcomes.values())} tasks on {args.to}: {dict(outcomes.most_common())}")
    return 0


def _split(args: argparse.Namespace) -> int:
    path = args.out / TASKS_FILE
    tasks = read_tasks(path)
    assign_splits(
        tasks,
        test_share=args.test_share,
        changed_test_share=args.changed_test_share,
        seed=args.seed,
    )
    write_tasks(path, tasks)
    for split in ("train", "test"):
        write_tasks(args.out / f"{split}.jsonl", [task for task in tasks if task.split == split])
    return _stats(args)


def _stats(args: argparse.Namespace) -> int:
    _, schema = connect(args, args.base)
    print("\n".join(summary(read_tasks(args.out / TASKS_FILE), schema)))
    return 0


def add_commands(commands: argparse._SubParsersAction) -> None:
    tasks = commands.add_parser("tasks", help="build the task set")
    tasks.add_argument("--data", type=Path, default=Path("data"), help="runtime directory")
    tasks.add_argument(
        "--out", type=Path, default=Path("data/tasks"), help="directory of the task set"
    )
    tasks.add_argument("--base", default="3.20", help="release the tasks are written for")
    actions = tasks.add_subparsers(dest="action", required=True)

    def with_teacher(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--teacher",
            nargs="+",
            default=["http://127.0.0.1:8100/v1"],
            metavar="URL",
            help="OpenAI-compatible endpoints serving the teacher model",
        )
        parser.add_argument("--model", default="teacher", help="name the teacher is served under")
        parser.add_argument("--workers", type=int, default=32, help="parallel requests")

    generate = actions.add_parser("generate", help="have the teacher write tasks and check them")
    generate.add_argument("--target", default="3.23", help="later release to cover the changes of")
    generate.add_argument("--requests", type=int, default=1000, help="prompts to send")
    generate.add_argument("--pairs", type=int, default=6, help="examples asked for per prompt")
    generate.add_argument("--seed", type=int, default=0)
    with_teacher(generate)
    generate.set_defaults(run=_generate)

    verification = actions.add_parser(
        "verify", help="keep track of which tasks the teacher can solve from the request alone"
    )
    with_teacher(verification)
    verification.set_defaults(run=_verify)

    porting = actions.add_parser(
        "port", help="find the reference query of the test tasks on a later release"
    )
    porting.add_argument("--to", default="3.23", help="later release, which must be running")
    porting.add_argument("--all", action="store_true", help="settle the training tasks as well")
    with_teacher(porting)
    porting.set_defaults(run=_port)

    split = actions.add_parser("split", help="assign tasks to the training and the test set")
    split.add_argument("--test-share", type=float, default=0.15)
    split.add_argument(
        "--changed-test-share",
        type=float,
        default=0.5,
        help="test share of the tasks a later release breaks",
    )
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(run=_split)

    stats = actions.add_parser("stats", help="summarise the task set")
    stats.set_defaults(run=_stats)
