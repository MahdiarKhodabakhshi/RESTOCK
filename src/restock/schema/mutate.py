"""Simulated API changes.

Training needs many more schema versions than Saleor has released. A mutation
renames or removes a field in the schema the agent sees.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from random import Random

from graphql import DocumentNode, GraphQLSchema, NameNode, ObjectTypeDefinitionNode, is_object_type

from restock.schema.diff import reachable_types

RENAME_SUFFIXES = ("Value", "Info", "Data", "Details", "Entry")


@dataclass(frozen=True)
class Rename:
    type_name: str
    field: str
    new_name: str


@dataclass(frozen=True)
class Remove:
    type_name: str
    field: str


Mutation = Rename | Remove


def mutable_fields(schema: GraphQLSchema) -> list[tuple[str, str]]:
    """Fields on the query side that can change without touching another type.

    Fields declared by an interface are left out, because changing them on one
    type would require the same change on every other implementation.
    """
    fields = []
    for name in sorted(reachable_types(schema, schema.query_type)):
        type_ = schema.type_map[name]
        if not is_object_type(type_) or name.startswith("__"):
            continue
        inherited = {field for interface in type_.interfaces for field in interface.fields}
        fields.extend((name, field) for field in type_.fields if field not in inherited)
    return fields


def sample_mutations(
    schema: GraphQLSchema, rng: Random, *, renames: int = 0, removals: int = 0
) -> list[Mutation]:
    """Draw distinct fields to rename and remove, keeping every type non-empty."""
    remaining = {name: len(getattr(t, "fields", ())) for name, t in schema.type_map.items()}
    candidates = mutable_fields(schema)
    rng.shuffle(candidates)

    mutations: list[Mutation] = []
    for type_name, field in candidates:
        if renames:
            taken = schema.type_map[type_name].fields
            new_name = next(
                field + suffix
                for suffix in rng.sample(RENAME_SUFFIXES, len(RENAME_SUFFIXES))
                if field + suffix not in taken
            )
            mutations.append(Rename(type_name, field, new_name))
            renames -= 1
        elif removals and remaining[type_name] > 1:
            mutations.append(Remove(type_name, field))
            remaining[type_name] -= 1
            removals -= 1
        if not renames and not removals:
            return mutations
    raise ValueError("schema has too few mutable fields for the requested mutations")


def apply_mutations(document: DocumentNode, mutations: Iterable[Mutation]) -> DocumentNode:
    """Return a copy of a schema document with the mutations applied."""
    by_type: dict[str, dict[str, Mutation]] = {}
    for mutation in mutations:
        by_type.setdefault(mutation.type_name, {})[mutation.field] = mutation

    definitions = []
    for definition in document.definitions:
        pending = (
            dict(by_type.pop(definition.name.value, {}))
            if isinstance(definition, ObjectTypeDefinitionNode)
            else {}
        )
        if pending:
            fields = []
            for field in definition.fields:
                mutation = pending.pop(field.name.value, None)
                if mutation is None:
                    fields.append(field)
                elif isinstance(mutation, Rename):
                    fields.append(replace(field, name=NameNode(value=mutation.new_name)))
            if pending:
                missing = ", ".join(f"{definition.name.value}.{name}" for name in pending)
                raise KeyError(f"no such field: {missing}")
            definition = replace(definition, fields=tuple(fields))
        definitions.append(definition)

    if by_type:
        raise KeyError(f"no such object type: {', '.join(by_type)}")
    return replace(document, definitions=tuple(definitions))
