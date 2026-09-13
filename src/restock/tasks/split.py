"""Divide the tasks into a training and a test set.

Two tasks that ask the same thing must not end up on different sides. Tasks
are therefore grouped first: by request, and by the shape of the query, which
is the query with its literal values blanked. Looking up product A and looking
up product B is one group. Groups, not tasks, are then assigned to a side.

Tasks that use something a later release breaks are rare and are what the
evaluation on that release is about, so a larger share of them goes to the test
set. The rest of them stays in training: a model can only go stale on a field
it has learned.
"""

from collections import defaultdict
from random import Random

from graphql import FloatValueNode, IntValueNode, StringValueNode, Visitor, parse, print_ast, visit

from restock.tasks.check import question_key
from restock.tasks.task import Task
from restock.tasks.verify import FAILED

TRAIN = "train"
TEST = "test"


def assign_splits(
    tasks: list[Task],
    *,
    test_share: float = 0.15,
    changed_test_share: float = 0.5,
    seed: int = 0,
) -> None:
    """Set ``split`` on every task.

    Tasks that failed verification get no split: their request does not
    determine their answer, which makes them wrong to train on and unfair to
    test on.
    """
    usable = [task for task in tasks if task.verified != FAILED]
    for task in tasks:
        task.split = None

    rng = Random(seed)
    groups = group_tasks(usable)
    for changed, share in ((True, changed_test_share), (False, test_share)):
        stratum = [group for group in groups if any(task.changed for task in group) is changed]
        rng.shuffle(stratum)
        test_groups = round(len(stratum) * share)
        for index, group in enumerate(stratum):
            for task in group:
                task.split = TEST if index < test_groups else TRAIN


def group_tasks(tasks: list[Task]) -> list[list[Task]]:
    """Groups of tasks connected by an equal request or an equal query shape."""
    parent = list(range(len(tasks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_with: dict[tuple[str, str], int] = {}
    for index, task in enumerate(tasks):
        for key in (("question", question_key(task.question)), ("shape", query_shape(task.query))):
            other = first_with.setdefault(key, index)
            parent[find(index)] = find(other)

    groups: dict[int, list[Task]] = defaultdict(list)
    for index, task in enumerate(tasks):
        groups[find(index)].append(task)
    return list(groups.values())


def query_shape(query: str) -> str:
    """The query with its strings and numbers blanked."""

    class Blank(Visitor):
        def enter_string_value(self, node: StringValueNode, *_):
            return StringValueNode(value="")

        def enter_int_value(self, node: IntValueNode, *_):
            return IntValueNode(value="0")

        def enter_float_value(self, node: FloatValueNode, *_):
            return FloatValueNode(value="0.0")

    return print_ast(visit(parse(query, no_location=True), Blank()))
