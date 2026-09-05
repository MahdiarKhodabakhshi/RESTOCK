"""Structured diff between two versions of a GraphQL schema.

graphql-core can list breaking changes, but only as free-text descriptions. The
changes here carry the schema coordinate they apply to and whether that
coordinate is reachable from ``Query``, which is what decides if a stored query
or a memorised field still works on the new version.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLInputField,
    GraphQLNamedType,
    GraphQLSchema,
    GraphQLType,
    get_named_type,
    is_enum_type,
    is_input_object_type,
    is_interface_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
    is_required_argument,
    is_required_input_field,
    is_union_type,
)


class ChangeKind(StrEnum):
    TYPE_REMOVED = "type_removed"
    TYPE_KIND_CHANGED = "type_kind_changed"
    INTERFACE_REMOVED = "interface_removed"
    UNION_MEMBER_REMOVED = "union_member_removed"
    ENUM_VALUE_REMOVED = "enum_value_removed"
    FIELD_REMOVED = "field_removed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    ARG_REMOVED = "arg_removed"
    ARG_TYPE_CHANGED = "arg_type_changed"
    REQUIRED_ARG_ADDED = "required_arg_added"
    REQUIRED_INPUT_FIELD_ADDED = "required_input_field_added"
    TYPE_ADDED = "type_added"
    FIELD_ADDED = "field_added"


NON_BREAKING = frozenset({ChangeKind.TYPE_ADDED, ChangeKind.FIELD_ADDED})


@dataclass(frozen=True)
class Change:
    kind: ChangeKind
    type_name: str
    member: str | None = None
    """Field, input field, enum value, union member or interface name."""
    argument: str | None = None
    query_side: bool = False
    """Whether the type is reachable from the ``Query`` root."""

    @property
    def breaking(self) -> bool:
        return self.kind not in NON_BREAKING

    @property
    def coordinate(self) -> str:
        if self.member is None:
            return self.type_name
        if self.argument is None:
            return f"{self.type_name}.{self.member}"
        return f"{self.type_name}.{self.member}({self.argument}:)"


def diff_schemas(old: GraphQLSchema, new: GraphQLSchema) -> list[Change]:
    """List the changes that turn ``old`` into ``new``."""
    old_query_side = reachable_types(old, old.query_type)
    new_query_side = reachable_types(new, new.query_type)
    old_types = _user_types(old)
    new_types = _user_types(new)

    changes: list[Change] = []
    for name, old_type in old_types.items():
        query_side = name in old_query_side
        new_type = new_types.get(name)
        if new_type is None:
            changes.append(Change(ChangeKind.TYPE_REMOVED, name, query_side=query_side))
        elif type(old_type) is not type(new_type):
            changes.append(Change(ChangeKind.TYPE_KIND_CHANGED, name, query_side=query_side))
        else:
            for kind, member, argument in _diff_type(old_type, new_type):
                changes.append(Change(kind, name, member, argument, query_side))

    for name in new_types.keys() - old_types.keys():
        changes.append(Change(ChangeKind.TYPE_ADDED, name, query_side=name in new_query_side))
    return changes


def reachable_types(schema: GraphQLSchema, root: GraphQLNamedType | None) -> frozenset[str]:
    """Names of all types a request starting at ``root`` can touch."""
    seen: set[str] = set()
    stack = [root] if root is not None else []
    while stack:
        current = stack.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        if is_object_type(current) or is_interface_type(current):
            for field in current.fields.values():
                stack.append(get_named_type(field.type))
                stack.extend(get_named_type(arg.type) for arg in field.args.values())
            stack.extend(current.interfaces)
        if is_interface_type(current) or is_union_type(current):
            stack.extend(schema.get_possible_types(current))
        if is_input_object_type(current):
            stack.extend(get_named_type(field.type) for field in current.fields.values())
    return frozenset(seen)


_Diff = Iterator[tuple[ChangeKind, str | None, str | None]]


def _user_types(schema: GraphQLSchema) -> dict[str, GraphQLNamedType]:
    return {name: t for name, t in schema.type_map.items() if not name.startswith("__")}


def _diff_type(old: GraphQLNamedType, new: GraphQLNamedType) -> _Diff:
    if is_enum_type(old):
        yield from _diff_enum_values(old.values, new.values)
    elif is_union_type(old):
        new_members = {member.name for member in new.types}
        for member in old.types:
            if member.name not in new_members:
                yield ChangeKind.UNION_MEMBER_REMOVED, member.name, None
    elif is_input_object_type(old):
        yield from _diff_input_fields(old.fields, new.fields)
    elif is_object_type(old) or is_interface_type(old):
        new_interfaces = {interface.name for interface in new.interfaces}
        for interface in old.interfaces:
            if interface.name not in new_interfaces:
                yield ChangeKind.INTERFACE_REMOVED, interface.name, None
        yield from _diff_fields(old.fields, new.fields)


def _diff_enum_values(old: dict, new: dict) -> _Diff:
    for name in old:
        if name not in new:
            yield ChangeKind.ENUM_VALUE_REMOVED, name, None


def _diff_fields(old: dict[str, GraphQLField], new: dict[str, GraphQLField]) -> _Diff:
    for name, old_field in old.items():
        new_field = new.get(name)
        if new_field is None:
            yield ChangeKind.FIELD_REMOVED, name, None
            continue
        if not _output_change_is_safe(old_field.type, new_field.type):
            yield ChangeKind.FIELD_TYPE_CHANGED, name, None
        for kind, argument in _diff_args(old_field.args, new_field.args):
            yield kind, name, argument
    for name in new.keys() - old.keys():
        yield ChangeKind.FIELD_ADDED, name, None


def _diff_args(
    old: dict[str, GraphQLArgument], new: dict[str, GraphQLArgument]
) -> Iterator[tuple[ChangeKind, str]]:
    for name, old_arg in old.items():
        new_arg = new.get(name)
        if new_arg is None:
            yield ChangeKind.ARG_REMOVED, name
        elif not _input_change_is_safe(old_arg.type, new_arg.type):
            yield ChangeKind.ARG_TYPE_CHANGED, name
    for name in new.keys() - old.keys():
        if is_required_argument(new[name]):
            yield ChangeKind.REQUIRED_ARG_ADDED, name


def _diff_input_fields(
    old: dict[str, GraphQLInputField], new: dict[str, GraphQLInputField]
) -> _Diff:
    for name, old_field in old.items():
        new_field = new.get(name)
        if new_field is None:
            yield ChangeKind.FIELD_REMOVED, name, None
        elif not _input_change_is_safe(old_field.type, new_field.type):
            yield ChangeKind.FIELD_TYPE_CHANGED, name, None
    for name in new.keys() - old.keys():
        if is_required_input_field(new[name]):
            yield ChangeKind.REQUIRED_INPUT_FIELD_ADDED, name, None
        else:
            yield ChangeKind.FIELD_ADDED, name, None


def _output_change_is_safe(old: GraphQLType, new: GraphQLType) -> bool:
    """A client reading a field keeps working if the new type is at least as strict."""
    if is_non_null_type(new) and not is_non_null_type(old):
        return _output_change_is_safe(old, new.of_type)
    if is_list_type(old):
        return is_list_type(new) and _output_change_is_safe(old.of_type, new.of_type)
    if is_non_null_type(old):
        return is_non_null_type(new) and _output_change_is_safe(old.of_type, new.of_type)
    return not is_list_type(new) and old.name == new.name


def _input_change_is_safe(old: GraphQLType, new: GraphQLType) -> bool:
    """A client sending a value keeps working if the new type is at least as lenient."""
    if is_non_null_type(old) and not is_non_null_type(new):
        return _input_change_is_safe(old.of_type, new)
    if is_list_type(old):
        return is_list_type(new) and _input_change_is_safe(old.of_type, new.of_type)
    if is_non_null_type(old):
        return is_non_null_type(new) and _input_change_is_safe(old.of_type, new.of_type)
    return not is_list_type(new) and not is_non_null_type(new) and old.name == new.name
