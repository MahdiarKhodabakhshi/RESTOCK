"""Simulated API changes.

Training needs many more schema versions than Saleor has released. A mutation
renames or removes a field in the schema the agent sees.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from graphql import DocumentNode, NameNode, ObjectTypeDefinitionNode


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
