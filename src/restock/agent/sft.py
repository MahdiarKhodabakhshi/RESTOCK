"""Supervised examples for the memorised-only and always-read baselines.

Memorised rows show no schema. Always-read rows include the same keyword-based
schema slice the policy gathers at eval time, so the model learns to use what
it is shown rather than only what it memorised.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from graphql import GraphQLSchema

from restock.agent.policies import AlwaysRead
from restock.agent.prompt import SYSTEM, always_read_prompt, memorised_prompt
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.tasks.task import Task


def sft_messages(task: Task, *, version: str) -> list[dict[str, str]]:
    """Chat messages that teach ``task.query`` from memory on ``version``."""
    signal = version_signal(version, trained_on=version).text()
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": memorised_prompt(signal=signal, question=task.question),
        },
        {"role": "assistant", "content": task.query},
    ]


def always_read_messages(
    task: Task,
    schema: GraphQLSchema,
    *,
    version: str,
    policy: AlwaysRead | None = None,
) -> list[dict[str, str]]:
    """Chat messages that teach ``task.query`` given a schema slice."""
    tools = SchemaTools(schema, version_signal(version, trained_on=version))
    lookup = (policy or AlwaysRead()).schema_for(tools, task.question)
    prompt = always_read_prompt(
        signal=tools.version_signal(),
        schema="\n\n".join(lookup),
        question=task.question,
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": task.query},
    ]


def iter_sft(tasks: Iterable[Task], *, version: str) -> Iterator[dict]:
    for task in tasks:
        yield {"messages": sft_messages(task, version=version), "id": task.id}


def iter_always_read_sft(
    tasks: Iterable[Task],
    schema: GraphQLSchema,
    *,
    version: str,
) -> Iterator[dict]:
    policy = AlwaysRead()
    for task in tasks:
        yield {
            "messages": always_read_messages(task, schema, version=version, policy=policy),
            "id": task.id,
        }


def write_sft(path: Path, tasks: Iterable[Task], *, version: str) -> int:
    """Write OpenAI-style chat JSONL. Returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in iter_sft(tasks, version=version):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_always_read_sft(
    path: Path,
    tasks: Iterable[Task],
    schema: GraphQLSchema,
    *,
    version: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in iter_always_read_sft(tasks, schema, version=version):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
