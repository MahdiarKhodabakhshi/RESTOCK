"""Pick a part of the schema for the teacher to write queries about.

The whole Saleor schema is about a megabyte, far more than a prompt should
carry and far more than one query uses. A focus is one field of ``Query`` and a
handful of types reachable from it. Rendering it gives a compact SDL with the
supporting inputs, enums and small value types those types refer to.
"""

from collections import deque
from dataclasses import dataclass
from random import Random

from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLNamedType,
    GraphQLSchema,
    get_named_type,
    is_enum_type,
    is_input_object_type,
    is_interface_type,
    is_object_type,
    is_union_type,
)

SMALL_TYPE_FIELDS = 8
LONG_ENUM_VALUES = 40
SHOWN_ENUM_VALUES = 15
DESCRIPTION_LENGTH = 110
INPUT_DEPTH = 3
# The federation entry points are not part of the store's API.
SKIPPED_ROOTS = frozenset({"_entities", "_service"})


@dataclass(frozen=True)
class Focus:
    root: str
    """Field of ``Query`` every query of this focus starts from."""
    types: tuple[str, ...]
    """Output types shown in full, in reading order."""
    feature: tuple[str, ...] = ()
    """Coordinates the teacher is asked to use."""


class SchemaGraph:
    """Which output types can be reached from each field of ``Query``."""

    def __init__(self, schema: GraphQLSchema):
        if schema.query_type is None:
            raise ValueError("schema has no Query type")
        self.schema = schema
        self.query = schema.query_type
        self.roots = [name for name in self.query.fields if name not in SKIPPED_ROOTS]
        self._parents: dict[str, dict[str, str | None]] = {}
        self._input_users: dict[str, list[tuple[str, str]]] | None = None

    def return_type(self, root: str) -> GraphQLNamedType:
        return get_named_type(self.query.fields[root].type)

    def distance(self, root: str, type_name: str) -> int | None:
        """Number of steps from the root field's type to ``type_name``.

        Going from a Relay connection to its nodes is not a step.
        """
        route = self.route(root, type_name)
        if route is None:
            return None
        wrappers = sum(1 for name in route[:-1] if self._tail(name))
        return len(route) - 1 - 2 * wrappers

    def route(self, root: str, type_name: str) -> list[str] | None:
        """Shortest chain of types from the root field's type to ``type_name``."""
        parents = self._reach(root)
        if type_name not in parents:
            return None
        chain = [type_name]
        while (parent := parents[chain[-1]]) is not None:
            chain.append(parent)
        return chain[::-1]

    def roots_reaching(self, type_name: str, slack: int = 1) -> list[str]:
        """Root fields that reach ``type_name`` in the fewest steps, or ``slack`` more."""
        distances = {root: self.distance(root, type_name) for root in self.roots}
        reachable = {root: d for root, d in distances.items() if d is not None}
        if not reachable:
            return []
        nearest = min(reachable.values())
        return [root for root, d in reachable.items() if d <= nearest + slack]

    def sites(self, coordinate: str) -> list[tuple[str, str]]:
        """Fields, as (type, field), in whose selection a query can use ``coordinate``.

        For a field or one of its arguments that is the field itself. An input
        field or an enum value can be used wherever a field takes an argument of
        that input type, directly or nested.
        """
        type_name, _, member = coordinate.partition(".")
        type_ = self.schema.type_map.get(type_name)
        if type_ is None:
            return []
        if is_object_type(type_) or is_interface_type(type_):
            field = member.split("(", 1)[0]
            return [(type_name, field)] if field in type_.fields else []
        return self._users_of_inputs().get(type_name, [])

    def children(self, type_: GraphQLNamedType) -> list[GraphQLNamedType]:
        """Output types one step away: field types and the members of abstract types."""
        found: dict[str, GraphQLNamedType] = {}
        if is_object_type(type_) or is_interface_type(type_):
            for field in type_.fields.values():
                target = get_named_type(field.type)
                if _is_composite(target):
                    found.setdefault(target.name, target)
        if is_interface_type(type_) or is_union_type(type_):
            for member in self.schema.get_possible_types(type_):
                found.setdefault(member.name, member)
        return list(found.values())

    def _tail(self, type_name: str) -> list[str]:
        return _connection_tail(self.schema, type_name)

    def _reach(self, root: str) -> dict[str, str | None]:
        if root not in self._parents:
            parents: dict[str, str | None] = {}
            queue: deque[GraphQLNamedType] = deque()

            def discover(type_: GraphQLNamedType, parent: str | None) -> None:
                if type_.name in parents:
                    return
                parents[type_.name] = parent
                # The nodes of a connection are found together with it, so that a
                # list of records is as near as a single record.
                previous = type_.name
                for name in self._tail(type_.name):
                    if name in parents:
                        return
                    parents[name] = previous
                    previous = name
                queue.append(self.schema.type_map[previous])

            start = self.return_type(root)
            if _is_composite(start):
                discover(start, None)
            while queue:
                current = queue.popleft()
                for child in self.children(current):
                    discover(child, current.name)
            self._parents[root] = parents
        return self._parents[root]

    def _users_of_inputs(self) -> dict[str, list[tuple[str, str]]]:
        if self._input_users is None:
            users: dict[str, list[tuple[str, str]]] = {}
            for name, type_ in sorted(self.schema.type_map.items()):
                if name.startswith("__") or not (is_object_type(type_) or is_interface_type(type_)):
                    continue
                for field_name, field in type_.fields.items():
                    for input_name in _input_closure(field.args.values()):
                        users.setdefault(input_name, []).append((name, field_name))
            self._input_users = users
        return self._input_users


def sample_focus(
    graph: SchemaGraph,
    rng: Random,
    root: str,
    *,
    through: str | None = None,
    feature: tuple[str, ...] = (),
    expansions: int = 3,
) -> Focus:
    """Draw the types to show for queries that start at ``root``.

    The chain from the root field to its node type is always shown, extended to
    ``through`` when that is given. From the end of the chain up to
    ``expansions`` neighbouring types are added, and sometimes one of theirs.
    """
    start = graph.return_type(root)
    if not _is_composite(start):
        return Focus(root, (), feature)

    if through is not None and through != graph.query.name:
        chain = graph.route(root, through)
        if chain is None:
            raise ValueError(f"{through} is not reachable from Query.{root}")
    else:
        chain = [start.name]
    shown = list(chain)
    _extend(shown, _connection_tail(graph.schema, shown[-1]))

    anchor = graph.schema.type_map[shown[-1]]
    count = rng.randint(1, expansions) if expansions else 0
    first_level = _pick_neighbours(graph, rng, anchor, shown, count)
    for neighbour in first_level:
        _extend(shown, [neighbour.name, *_connection_tail(graph.schema, neighbour.name)])
    if first_level and rng.random() < 0.4:
        deeper = graph.schema.type_map[shown[-1]]
        for neighbour in _pick_neighbours(graph, rng, deeper, shown, 1):
            _extend(shown, [neighbour.name, *_connection_tail(graph.schema, neighbour.name)])
    return Focus(root, tuple(shown), feature)


def render(schema: GraphQLSchema, focus: Focus, budget: int = 40_000) -> str:
    """Compact SDL of a focus, at most about ``budget`` characters long.

    The types of the focus come first and are always printed. Inputs, enums and
    small value types they refer to follow while the budget lasts; anything
    left out is still named in the field signatures.
    """
    query = schema.query_type
    blocks = [f"type {query.name} {{\n{_field_line(focus.root, query.fields[focus.root])}\n}}"]
    main = [schema.type_map[name] for name in focus.types]
    blocks.extend(_print_type(schema, type_) for type_ in main)

    # Types named by a featured coordinate go first, so that the budget cannot drop them.
    featured = {coordinate.split(".", 1)[0] for coordinate in focus.feature}
    supporting = _supporting_types(schema, query.fields[focus.root], main)
    supporting.sort(key=lambda type_: type_.name not in featured)

    used = sum(len(block) for block in blocks)
    for type_ in supporting:
        block = _print_type(schema, type_)
        if used + len(block) > budget:
            continue
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def render_types(schema: GraphQLSchema, roots: list[str], types: list[str]) -> str:
    """Compact SDL of the named fields of ``Query`` and the named types, where they exist."""
    query = schema.query_type
    lines = [_field_line(root, query.fields[root]) for root in roots if root in query.fields]
    blocks = [f"type {query.name} {{\n" + "\n".join(lines) + "\n}"] if lines else []
    blocks.extend(
        _print_type(schema, schema.type_map[name]) for name in types if name in schema.type_map
    )
    return "\n\n".join(blocks)


def fields_of(schema: GraphQLSchema, focus: Focus) -> list[str]:
    """Field coordinates of the types a focus shows in full."""
    coordinates = [f"{schema.query_type.name}.{focus.root}"]
    for name in focus.types:
        type_ = schema.type_map[name]
        coordinates.extend(f"{name}.{field}" for field in getattr(type_, "fields", ()))
    return coordinates


def _is_composite(type_: GraphQLNamedType) -> bool:
    return is_object_type(type_) or is_interface_type(type_) or is_union_type(type_)


def _extend(shown: list[str], names: list[str]) -> None:
    shown.extend(name for name in names if name not in shown)


def _connection_tail(schema: GraphQLSchema, type_name: str) -> list[str]:
    """For a Relay connection, its edge and node types."""
    tail = []
    current = schema.type_map[type_name]
    for step in ("edges", "node"):
        field = getattr(current, "fields", {}).get(step)
        if field is None or not _is_composite(get_named_type(field.type)):
            break
        current = get_named_type(field.type)
        tail.append(current.name)
    return tail if len(tail) == 2 else []


def _is_small(type_: GraphQLNamedType) -> bool:
    """A value type such as ``Money``: few fields and nothing below it worth a choice."""
    return is_object_type(type_) and len(type_.fields) <= SMALL_TYPE_FIELDS


def _pick_neighbours(
    graph: SchemaGraph,
    rng: Random,
    type_: GraphQLNamedType,
    shown: list[str],
    count: int,
) -> list[GraphQLNamedType]:
    candidates = [
        child
        for child in graph.children(type_)
        if child.name not in shown
        and not (_is_small(child) and not _connection_tail(graph.schema, child.name))
    ]
    return rng.sample(candidates, min(count, len(candidates)))


def _supporting_types(
    schema: GraphQLSchema, root: GraphQLField, main: list[GraphQLNamedType]
) -> list[GraphQLNamedType]:
    """Types the printed fields refer to, most useful first."""
    shown = {type_.name for type_ in main}
    fields = [root] + [field for type_ in main for field in getattr(type_, "fields", {}).values()]

    root_inputs = [schema.type_map[name] for name in _input_closure(root.args.values())]
    other_inputs = [
        schema.type_map[name]
        for name in _input_closure(arg for field in fields[1:] for arg in field.args.values())
    ]

    small: dict[str, GraphQLNamedType] = {}
    queue = deque(get_named_type(field.type) for field in fields)
    while queue:
        type_ = queue.popleft()
        if type_.name in shown or type_.name in small:
            continue
        if is_enum_type(type_) or _is_small(type_):
            small[type_.name] = type_
            queue.extend(get_named_type(f.type) for f in getattr(type_, "fields", {}).values())

    ordered: dict[str, GraphQLNamedType] = {}
    for type_ in (*root_inputs, *small.values(), *other_inputs):
        if type_.name not in shown:
            ordered.setdefault(type_.name, type_)
    return list(ordered.values())


def _input_closure(arguments) -> list[str]:
    """Names of the input objects and enums a set of arguments accepts, nested ones included."""
    found: dict[str, None] = {}
    queue = deque((get_named_type(argument.type), 1) for argument in arguments)
    while queue:
        type_, depth = queue.popleft()
        if type_.name in found or not (is_enum_type(type_) or is_input_object_type(type_)):
            continue
        found[type_.name] = None
        if is_input_object_type(type_) and depth < INPUT_DEPTH:
            queue.extend((get_named_type(f.type), depth + 1) for f in type_.fields.values())
    return list(found)


def _print_type(schema: GraphQLSchema, type_: GraphQLNamedType) -> str:
    if is_enum_type(type_):
        values = list(type_.values)
        if len(values) > LONG_ENUM_VALUES:
            values = [*values[:SHOWN_ENUM_VALUES], f"... {len(values) - SHOWN_ENUM_VALUES} more"]
        return f"enum {type_.name} {{ {' '.join(values)} }}"
    if is_union_type(type_):
        return f"union {type_.name} = {' | '.join(member.name for member in type_.types)}"
    if is_input_object_type(type_):
        lines = [_input_line(name, field) for name, field in type_.fields.items()]
        return f"input {type_.name} {{\n" + "\n".join(lines) + "\n}"

    keyword = "interface" if is_interface_type(type_) else "type"
    header = f"{keyword} {type_.name}"
    if type_.interfaces:
        header += " implements " + " & ".join(interface.name for interface in type_.interfaces)
    lines = [_field_line(name, field) for name, field in type_.fields.items()]
    block = f"{header} {{\n" + "\n".join(lines) + "\n}"
    if is_interface_type(type_):
        members = ", ".join(member.name for member in schema.get_possible_types(type_))
        block += f"\n# {type_.name} is implemented by: {members}"
    return block


def _field_line(name: str, field: GraphQLField) -> str:
    arguments = ", ".join(_argument(arg_name, arg) for arg_name, arg in field.args.items())
    signature = f"  {name}({arguments})" if arguments else f"  {name}"
    return f"{signature}: {field.type}" + _notes(field)


def _argument(name: str, argument: GraphQLArgument) -> str:
    return f"{name}: {argument.type}"


def _input_line(name: str, field) -> str:
    return f"  {name}: {field.type}" + _notes(field)


def _notes(item) -> str:
    notes = " @deprecated" if item.deprecation_reason is not None else ""
    summary = _summary(item.description)
    return f"{notes}  # {summary}" if summary else notes


def _summary(description: str | None) -> str:
    """First sentence of a description, on one line."""
    if not description:
        return ""
    text = " ".join(description.split())
    end = text.find(". ")
    if end != -1:
        text = text[: end + 1]
    if len(text) > DESCRIPTION_LENGTH:
        text = text[: DESCRIPTION_LENGTH - 3].rstrip() + "..."
    return text
