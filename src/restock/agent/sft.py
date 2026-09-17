"""Supervised examples for the memorised-only and always-read baselines.

Memorised rows show no schema. Always-read rows include the same keyword-based
schema slice the policy gathers at eval time. A fraction of always-read rows
rename fields in that slice (and in the target query) so the model has to follow
the schema it is shown, not only the release it was trained against.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from random import Random

from graphql import GraphQLSchema, build_ast_schema, parse

from restock.agent.policies import AlwaysRead
from restock.agent.prompt import SYSTEM, always_read_prompt, memorised_prompt
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.schema.mutate import (
    InvalidQuery,
    apply_mutations,
    rename_query_fields,
    sample_mutations,
)
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
    document=None,
    rng: Random | None = None,
    mutate_prob: float = 0.0,
) -> list[dict[str, str]] | None:
    """Chat messages that teach a query given a schema slice.

    With probability ``mutate_prob``, renames are applied to the schema the
    model sees and to the target query. Returns ``None`` when a mutation would
    break the gold query (for example by removing a field it needs).
    """
    query = task.query
    view = schema
    signal_version = version
    if rng is not None and mutate_prob > 0 and rng.random() < mutate_prob and document is not None:
        try:
            mutations = sample_mutations(schema, rng, renames=4, removals=0)
            mutated_doc = apply_mutations(document, mutations)
            view = build_ast_schema(mutated_doc)
            query = rename_query_fields(task.query, schema, mutations)
            signal_version = f"{version}+mut{rng.randrange(1_000_000)}"
        except (InvalidQuery, ValueError, KeyError):
            return None
    tools = SchemaTools(view, version_signal(signal_version, trained_on=version))
    lookup = (policy or AlwaysRead()).schema_for(tools, task.question)
    prompt = always_read_prompt(
        signal=tools.version_signal(),
        schema="\n\n".join(lookup),
        question=task.question,
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": query},
    ]


def iter_sft(tasks: Iterable[Task], *, version: str) -> Iterator[dict]:
    for task in tasks:
        yield {"messages": sft_messages(task, version=version), "id": task.id}


def iter_always_read_sft(
    tasks: Iterable[Task],
    schema: GraphQLSchema,
    *,
    version: str,
    mutate_prob: float = 0.5,
    seed: int = 0,
    sdl: str | None = None,
) -> Iterator[dict]:
    policy = AlwaysRead()
    rng = Random(seed)
    document = parse(sdl) if sdl is not None else None
    for task in tasks:
        messages = always_read_messages(
            task,
            schema,
            version=version,
            policy=policy,
            document=document,
            rng=rng,
            mutate_prob=mutate_prob,
        )
        if messages is None:
            continue
        yield {"messages": messages, "id": task.id}


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
    mutate_prob: float = 0.5,
    seed: int = 0,
    sdl: str | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in iter_always_read_sft(
            tasks, schema, version=version, mutate_prob=mutate_prob, seed=seed, sdl=sdl
        ):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
