"""Summary of a task set."""

from collections import Counter

from graphql import GraphQLSchema

from restock.schema.diff import reachable_types
from restock.tasks.split import query_shape
from restock.tasks.task import Task

KINDS = ("changed", "deprecated", "untouched")


def kind(task: Task) -> str:
    """Whether the target release breaks, deprecates or leaves alone what a task uses."""
    if task.broken:
        return "changed"
    return "deprecated" if task.deprecated else "untouched"


def summary(tasks: list[Task], schema: GraphQLSchema) -> list[str]:
    lines = [f"tasks: {len(tasks)}"]
    # Many tasks differ only in the record they look up; shapes count what is left.
    shapes = {query_shape(task.query) for task in tasks}
    lines.append(f"query shapes: {len(shapes)}")
    verified = Counter(task.verified or "not checked" for task in tasks)
    lines.append("verification: " + _counts(verified))

    lines.append("")
    lines.append(f"{'split':<8} {'tasks':>6} " + " ".join(f"{name:>10}" for name in KINDS))
    for split in ("train", "test", None):
        members = [task for task in tasks if task.split == split]
        if not members:
            continue
        kinds = Counter(kind(task) for task in members)
        cells = " ".join(f"{kinds[name]:>10}" for name in KINDS)
        lines.append(f"{split or 'none':<8} {len(members):>6} {cells}")

    releases = sorted({release for task in tasks for release in task.later})
    for release in releases:
        lines.append("")
        lines.append(f"on {release}, test set:")
        for name in KINDS:
            statuses = Counter(
                task.later[release].status
                for task in tasks
                if task.split == "test" and kind(task) == name and release in task.later
            )
            lines.append(f"  {name:<10} {_counts(statuses) or '-'}")
        lines.append(f"on {release}, test set, by changed coordinate:")
        by_coordinate: dict[str, Counter] = {}
        for task in tasks:
            if task.split == "test" and release in task.later:
                for coordinate in task.broken:
                    by_coordinate.setdefault(coordinate, Counter())[task.later[release].status] += 1
        for coordinate, statuses in sorted(by_coordinate.items()):
            lines.append(f"  {coordinate:<48} {_counts(statuses)}")

    used = {c for task in tasks if task.split for c in task.coordinates if _is_field(c)}
    fields = {
        f"{name}.{field}"
        for name in reachable_types(schema, schema.query_type)
        for field in getattr(schema.type_map[name], "fields", ())
        if not name.startswith("__")
    }
    lines.append("")
    lines.append(f"fields used: {len(used & fields)} of {len(fields)} reachable from Query")
    broken = Counter(c for task in tasks if task.split for c in task.broken)
    lines.append(f"changed coordinates with a task: {len(broken)}")
    return lines


def _is_field(coordinate: str) -> bool:
    return "." in coordinate and "(" not in coordinate


def _counts(counter: Counter) -> str:
    return ", ".join(f"{name} {count}" for name, count in counter.most_common())
