"""Compact SDL for the bits of a schema an agent might look up.

Descriptions are dropped: the agent pays for names and types, which are what
decide whether a memorised query still validates.
"""

from graphql import (
    GraphQLArgument,
    GraphQLEnumType,
    GraphQLField,
    GraphQLInputField,
    GraphQLInputObjectType,
    GraphQLInterfaceType,
    GraphQLNamedType,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLUnionType,
    is_enum_type,
    is_input_object_type,
    is_interface_type,
    is_object_type,
    is_scalar_type,
    is_union_type,
)


def render_type(type_: GraphQLNamedType) -> str:
    """SDL for one named type, without descriptions."""
    if is_scalar_type(type_):
        return _render_scalar(type_)
    if is_enum_type(type_):
        return _render_enum(type_)
    if is_union_type(type_):
        return _render_union(type_)
    if is_input_object_type(type_):
        return _render_input(type_)
    if is_interface_type(type_):
        return _render_interface(type_)
    if is_object_type(type_):
        return _render_object(type_)
    raise TypeError(f"unsupported GraphQL type: {type(type_).__name__}")


def _render_scalar(type_: GraphQLScalarType) -> str:
    return f"scalar {type_.name}"


def _render_enum(type_: GraphQLEnumType) -> str:
    values = "\n".join(f"  {name}" for name in type_.values)
    return f"enum {type_.name} {{\n{values}\n}}"


def _render_union(type_: GraphQLUnionType) -> str:
    members = " | ".join(member.name for member in type_.types)
    return f"union {type_.name} = {members}"


def _render_input(type_: GraphQLInputObjectType) -> str:
    fields = "\n".join(f"  {name}: {_type_name(field)}" for name, field in type_.fields.items())
    return f"input {type_.name} {{\n{fields}\n}}"


def _render_interface(type_: GraphQLInterfaceType) -> str:
    fields = "\n".join(_render_field(name, field) for name, field in type_.fields.items())
    return f"interface {type_.name} {{\n{fields}\n}}"


def _render_object(type_: GraphQLObjectType) -> str:
    implements = ""
    if type_.interfaces:
        implements = " implements " + " & ".join(i.name for i in type_.interfaces)
    fields = "\n".join(_render_field(name, field) for name, field in type_.fields.items())
    return f"type {type_.name}{implements} {{\n{fields}\n}}"


def _render_field(name: str, field: GraphQLField) -> str:
    args = ""
    if field.args:
        parts = [f"{arg_name}: {_type_name(arg)}" for arg_name, arg in field.args.items()]
        args = "(" + ", ".join(parts) + ")"
    return f"  {name}{args}: {_type_name(field)}"


def _type_name(field: GraphQLField | GraphQLArgument | GraphQLInputField | object) -> str:
    return str(field.type)
