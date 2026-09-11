"""Schema coordinates a query depends on.

A coordinate names one thing a query needs from the schema: a field
(``Product.name``), an argument (``Query.products(first:)``), an input field
(``ProductFilterInput.search``), an enum value (``OrderStatus.UNFULFILLED``)
or the type of a fragment (``Product``). They use the same notation as
``restock.schema.diff.Change.coordinate``, so the changes between two versions
can be looked up for a query.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from graphql import (
    ArgumentNode,
    DocumentNode,
    EnumValueNode,
    FieldNode,
    GraphQLSchema,
    ObjectFieldNode,
    TypeInfo,
    TypeInfoVisitor,
    Visitor,
    get_named_type,
    is_enum_type,
    visit,
)
from graphql.language import NamedTypeNode

from restock.schema.diff import Change, ChangeKind

LOST_TYPE = (ChangeKind.TYPE_REMOVED, ChangeKind.TYPE_KIND_CHANGED)


def used_coordinates(schema: GraphQLSchema, document: DocumentNode) -> frozenset[str]:
    """Coordinates used by a document that is valid on ``schema``."""
    type_info = TypeInfo(schema)
    found: set[str] = set()

    class Collect(Visitor):
        def enter_field(self, node: FieldNode, *_):
            parent = type_info.get_parent_type()
            if parent is not None and not node.name.value.startswith("__"):
                found.add(f"{parent.name}.{node.name.value}")

        def enter_argument(self, node: ArgumentNode, _key, _parent, _path, ancestors):
            parent = type_info.get_parent_type()
            owner = ancestors[-1]
            # Directives take arguments too; those are not part of the schema's fields.
            if parent is not None and isinstance(owner, FieldNode):
                found.add(f"{parent.name}.{owner.name.value}({node.name.value}:)")

        def enter_object_field(self, node: ObjectFieldNode, *_):
            parent = type_info.get_parent_input_type()
            if parent is not None:
                found.add(f"{get_named_type(parent).name}.{node.name.value}")

        def enter_enum_value(self, node: EnumValueNode, *_):
            type_ = get_named_type(type_info.get_input_type())
            if is_enum_type(type_):
                found.add(f"{type_.name}.{node.value}")

        def enter_named_type(self, node: NamedTypeNode, *_):
            found.add(node.name.value)

    visit(document, TypeInfoVisitor(type_info, Collect()))
    return frozenset(found)


@dataclass(frozen=True)
class Touched:
    """Coordinates of one query that change in a later version."""

    broken: tuple[str, ...] = ()
    deprecated: tuple[str, ...] = ()


def touched(coordinates: Iterable[str], changes: Iterable[Change]) -> Touched:
    """Pick out the coordinates that ``changes`` break or deprecate."""
    broken: set[str] = set()
    deprecated: set[str] = set()
    lost_types: set[str] = set()
    for change in changes:
        if change.kind in LOST_TYPE:
            lost_types.add(change.type_name)
        elif change.breaking:
            broken.add(change.coordinate)
        elif change.kind is ChangeKind.DEPRECATED:
            deprecated.add(change.coordinate)

    coordinates = set(coordinates)
    hit = {c for c in coordinates if c in broken or _type_of(c) in lost_types}
    return Touched(tuple(sorted(hit)), tuple(sorted((coordinates & deprecated) - hit)))


def _type_of(coordinate: str) -> str:
    return coordinate.split(".", 1)[0]
