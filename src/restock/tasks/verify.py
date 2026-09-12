"""Check that a task can be solved from its request alone.

The teacher writes request and query together, so nothing guarantees that the
request says everything the query does. Here the teacher gets the request and
the same part of the schema, without the query, and writes its own. A task is
verified when that second query returns the answer of the first: the request
then determines the answer for someone who can read the schema.
"""

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from graphql import GraphQLSchema

from restock.reward import results_match
from restock.saleor.client import SaleorClient
from restock.tasks.check import check
from restock.tasks.focus import Focus, render
from restock.tasks.task import Task
from restock.tasks.teacher import Teacher, TeacherError, parse_query, solve_prompt
from restock.tasks.values import Store

EXACT = "exact"
"""The second query returned the same response."""
VALUES = "values"
"""It returned the same values under other keys or in another shape."""
FAILED = "failed"

CHECKPOINT_EVERY = 500


def verify(
    tasks: list[Task],
    schema: GraphQLSchema,
    client: SaleorClient,
    teacher: Teacher,
    *,
    version: str,
    workers: int = 32,
    log: Callable[[str], None] = print,
    checkpoint: Callable[[], None] | None = None,
) -> Counter:
    """Set ``verified`` on every task that has not been through verification yet.

    ``checkpoint`` is called every few hundred tasks, so that the caller can
    save what is done and a later call continues from there.
    """
    facts = Store(schema, client).facts()

    def solve(task: Task) -> str:
        prompt = solve_prompt(
            version=version,
            root=task.focus.root,
            # The featured coordinates would give away part of the query.
            schema=render(schema, Focus(task.focus.root, task.focus.types)),
            facts=facts,
            question=task.question,
        )
        try:
            query = parse_query(teacher.complete(prompt, temperature=0.0))
        except TeacherError as error:
            log(f"teacher failed: {error}")
            return FAILED
        if query is None:
            return FAILED
        checked = check(schema, client, query, require_content=False, repeat=False)
        return agreement(task.answer, checked.data) if checked.ok else FAILED

    pending = [task for task in tasks if task.verified is None]
    outcomes: Counter = Counter()
    with ThreadPoolExecutor(workers) as pool:
        results = zip(pending, pool.map(solve, pending), strict=True)
        for done, (task, outcome) in enumerate(results, start=1):
            task.verified = outcome
            outcomes[outcome] += 1
            if done % CHECKPOINT_EVERY == 0:
                log(f"verified {done} of {len(pending)}: {dict(outcomes)}")
                if checkpoint is not None:
                    checkpoint()
    return outcomes


def agreement(expected: Any, actual: Any) -> str:
    """How far two answers agree: ``exact``, ``values`` or ``failed``."""
    if results_match(expected, actual):
        return EXACT
    if leaf_values(expected) == leaf_values(actual):
        return VALUES
    return FAILED


def leaf_values(data: Any) -> list[str]:
    """The values of an answer without their keys and nesting, nulls left out."""
    found: list[str] = []
    stack = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif value is not None:
            found.append(json.dumps(value))
    return sorted(found)
