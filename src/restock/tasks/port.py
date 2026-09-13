"""Reference answers on a later release.

Every release serves the same store, so the answer to a request does not depend
on the release, only the query that obtains it may. A task is settled on the
later release in one of four ways:

``same``
    The original query still returns the original answer.
``ported``
    The original query no longer does, and a query written for the later
    release returns the original answer.
``weak``
    As ``ported``, but the answer is a flag, a count or another value so small
    that an unrelated field can return it by chance. The query is kept for
    inspection and is not a reference.
``gone``
    No query returned the answer, and the task uses something the later
    release removed: the information is no longer offered.
``unresolved``
    No query returned the answer although nothing the task uses was removed.
    The release behaves differently and it is not known what the right
    answer is, so the task should stay out of an evaluation on that release.
"""

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from graphql import (
    GraphQLError,
    GraphQLSchema,
    is_enum_type,
    is_input_object_type,
    parse,
    validate,
)

from restock.reward import results_match
from restock.saleor.client import SaleorClient, SaleorError
from restock.tasks.check import check, run
from restock.tasks.focus import render_types
from restock.tasks.task import Later, Task
from restock.tasks.teacher import Teacher, TeacherError, parse_query, port_prompt
from restock.tasks.verify import leaf_values

SAME = "same"
PORTED = "ported"
WEAK = "weak"
GONE = "gone"
UNRESOLVED = "unresolved"

ATTEMPT_TEMPERATURES = (0.0, 0.7)
MIN_ANSWER_CHARS = 8
CHECKPOINT_EVERY = 200


def port(
    tasks: list[Task],
    schema: GraphQLSchema,
    client: SaleorClient,
    teacher: Teacher,
    *,
    old: str,
    new: str,
    workers: int = 32,
    log: Callable[[str], None] = print,
    checkpoint: Callable[[], None] | None = None,
) -> Counter:
    """Settle every task on release ``new``, whose schema and server are given.

    Tasks already settled on ``new`` are skipped. ``checkpoint`` is called every
    few hundred tasks, so that the caller can save what is done.
    """

    def settle(task: Task) -> Later:
        problem = original_problem(task, schema, client)
        if problem is None:
            return Later(SAME, task.query)

        prompt = port_prompt(
            old=old,
            new=new,
            question=task.question,
            query=task.query,
            problem=problem,
            schema=render_types(schema, _roots(task), _types(task, schema)),
        )
        for temperature in ATTEMPT_TEMPERATURES:
            try:
                completion = teacher.complete(prompt, temperature=temperature, thinking=True)
            except TeacherError as error:
                log(f"teacher failed: {error}")
                continue
            query = parse_query(completion)
            if query is None:
                continue
            checked = check(schema, client, query, require_content=False, repeat=False)
            if not checked.ok:
                continue
            status = PORTED if informative(task.answer) else WEAK
            if results_match(task.answer, checked.data):
                return Later(status, checked.query)
            if leaf_values(task.answer) == leaf_values(checked.data):
                return Later(status, checked.query, checked.data)
        return Later(GONE if task.broken else UNRESOLVED)

    pending = [task for task in tasks if new not in task.later]
    outcomes: Counter = Counter()
    with ThreadPoolExecutor(workers) as pool:
        results = zip(pending, pool.map(settle, pending), strict=True)
        for done, (task, later) in enumerate(results, start=1):
            task.later[new] = later
            outcomes[later.status] += 1
            if done % CHECKPOINT_EVERY == 0:
                log(f"settled {done} of {len(pending)}: {dict(outcomes)}")
                if checkpoint is not None:
                    checkpoint()
    return outcomes


def informative(answer: Any) -> bool:
    """Whether an answer is large enough that a wrong query is unlikely to return it.

    The model that ports a query is told to keep the old response keys, so a
    field it picked wrongly still answers under the right name. Only the values
    can tell the two apart, and a lone flag or count cannot.
    """
    values = {value for value in leaf_values(answer) if value not in ("true", "false")}
    return sum(len(value) for value in values) >= MIN_ANSWER_CHARS


def original_problem(task: Task, schema: GraphQLSchema, client: SaleorClient) -> str | None:
    """What goes wrong with the original query on the later release, if anything."""
    try:
        errors = validate(schema, parse(task.query))
    except GraphQLError as error:
        errors = [error]
    if errors:
        return "it is rejected. " + " ".join(error.message for error in errors)
    try:
        data, messages = run(client, task.query)
    except (SaleorError, OSError) as error:
        return f"it fails. {error}"
    if messages:
        return "it fails. " + " ".join(messages)
    if not results_match(task.answer, data):
        return "it returns a different answer."
    return None


def _roots(task: Task) -> list[str]:
    return [c.split(".", 1)[1] for c in task.coordinates if c.startswith("Query.") and "(" not in c]


def _types(task: Task, schema: GraphQLSchema) -> list[str]:
    """Types the query uses that the later release still has, outputs first."""
    names = dict.fromkeys(c.split(".", 1)[0] for c in task.coordinates)
    present = [name for name in names if name in schema.type_map and name != "Query"]

    def is_input(name: str) -> bool:
        type_ = schema.type_map[name]
        return is_enum_type(type_) or is_input_object_type(type_)

    return sorted(present, key=is_input)
