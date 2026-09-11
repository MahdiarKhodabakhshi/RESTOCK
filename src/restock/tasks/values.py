"""Real values from the store for the teacher to use in its queries.

A query that filters on an invented slug or ID returns nothing. The teacher is
therefore shown a few records of the type it writes about, read from the store
with a query generated from the schema, and a short list of store-wide facts.
"""

import json
from random import Random
from typing import Any

from graphql import (
    GraphQLField,
    GraphQLNamedType,
    GraphQLSchema,
    get_named_type,
    is_enum_type,
    is_interface_type,
    is_leaf_type,
    is_list_type,
    is_non_null_type,
    is_object_type,
    is_required_argument,
)

from restock.saleor.client import Result, SaleorClient

POOL_SIZE = 100
MAX_REPAIRS = 12
LONG_STRING = 120
LONG_LIST = 12
# Fields that identify a related record, which is what arguments and filters ask for.
IDENTITY_FIELDS = ("id", "name", "slug", "email", "code", "number", "sku", "amount", "currency")
# Opaque or secret values that are of no use in a question.
SKIPPED_FIELDS = frozenset(
    {"privateMetadata", "metadata", "accessToken", "token", "cursor", "password", "lastLogin"}
)

FACTS_QUERY = """
{
  channels { slug name currencyCode }
  categories(first: 30) { edges { node { id name slug } } }
  collections(first: 10) { edges { node { id name slug } } }
  productTypes(first: 20) { edges { node { id name slug } } }
  warehouses(first: 10) { edges { node { id name slug } } }
  attributes(first: 20) { edges { node { id name slug } } }
  first: orders(first: 1, sortBy: { field: CREATED_AT, direction: ASC }) {
    edges { node { created } }
  }
  last: orders(first: 1, sortBy: { field: CREATED_AT, direction: DESC }) {
    totalCount
    edges { node { created } }
  }
}
"""

Selection = dict[str, "Selection | None"]


class Store:
    """Reads sample records for the fields of ``Query``, once per field."""

    def __init__(self, schema: GraphQLSchema, client: SaleorClient):
        self.schema = schema
        self.client = client
        self._pools: dict[str, list[dict[str, Any]]] = {}
        self._facts: str | None = None

    def records(self, root: str, rng: Random, count: int = 4) -> list[dict[str, Any]]:
        """A few records of the kind ``Query.<root>`` returns, or none if it has no data."""
        pool = self._pool(root)
        return rng.sample(pool, min(count, len(pool)))

    def facts(self) -> str:
        """Channels, categories and other values that queries on any type refer to."""
        if self._facts is None:
            result = self.client.execute(FACTS_QUERY)
            self._facts = format_facts(result.data) if result.ok else ""
        return self._facts

    def _pool(self, root: str) -> list[dict[str, Any]]:
        if root not in self._pools:
            source = self._source(root)
            self._pools[root] = self._read(source) if source else []
            if source and source != root:
                self._pools[source] = self._pools[root]
        return self._pools[root]

    def _source(self, root: str) -> str | None:
        """The field to read records from: ``root`` itself, or a list of the same type."""
        fields = self.schema.query_type.fields
        if not _needs_arguments(fields[root]) and _record_type(fields[root]) is not None:
            return root
        wanted = get_named_type(fields[root].type)
        for name, field in fields.items():
            record = _record_type(field)
            if record is wanted and not _needs_arguments(field) and _is_list(field):
                return name
        return None

    def _read(self, root: str) -> list[dict[str, Any]]:
        field = self.schema.query_type.fields[root]
        record_type = _record_type(field)
        selection = record_selection(record_type)
        connection = "first" in field.args
        if connection:
            selection = {"edges": {"node": selection}}

        for _ in range(MAX_REPAIRS):
            arguments = f"(first: {POOL_SIZE})" if connection else ""
            result = self.client.execute(f"{{ {root}{arguments} {print_selection(selection)} }}")
            if not result.errors:
                return _records(result, root, connection)
            if not any(_drop(selection, error.get("path", [])[1:]) for error in result.errors):
                break
        return []


def record_selection(type_: GraphQLNamedType) -> Selection:
    """Fields to read from a record: its own values and what identifies related records."""
    selection: Selection = {}
    for name, field in type_.fields.items():
        if name in SKIPPED_FIELDS or _needs_arguments(field):
            continue
        target = get_named_type(field.type)
        if is_leaf_type(target):
            selection[name] = None
        elif is_object_type(target) and not _is_list(field):
            nested = {
                inner: None
                for inner, inner_field in target.fields.items()
                if inner in IDENTITY_FIELDS
                and is_leaf_type(get_named_type(inner_field.type))
                and not _needs_arguments(inner_field)
            }
            if nested:
                selection[name] = nested
    return selection


def print_selection(selection: Selection) -> str:
    parts = [
        name if nested is None else f"{name} {print_selection(nested)}"
        for name, nested in selection.items()
    ]
    return "{ " + " ".join(parts) + " }"


def compact(value: Any) -> Any:
    """Drop empty values and shorten long strings and lists, so that a record fits in a prompt."""
    if isinstance(value, dict):
        items = {key: compact(item) for key, item in value.items()}
        return {key: item for key, item in items.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [compact(item) for item in value[:LONG_LIST]]
    if isinstance(value, str) and len(value) > LONG_STRING:
        return value[:LONG_STRING] + "..."
    return value


def format_records(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(compact(record), ensure_ascii=False) for record in records)


def format_facts(data: dict[str, Any]) -> str:
    def nodes(key: str) -> list[dict[str, Any]]:
        return [edge["node"] for edge in data[key]["edges"]]

    lines = [
        "Channels: "
        + ", ".join(f'"{c["slug"]}" ({c["name"]}, {c["currencyCode"]})' for c in data["channels"])
    ]
    for title, key in (
        ("Categories", "categories"),
        ("Collections", "collections"),
        ("Product types", "productTypes"),
        ("Warehouses", "warehouses"),
        ("Attributes", "attributes"),
    ):
        items = ", ".join(f'{n["name"]} (slug "{n["slug"]}", id "{n["id"]}")' for n in nodes(key))
        lines.append(f"{title}: {items}")
    first, last = nodes("first"), nodes("last")
    if first and last:
        lines.append(
            f"Orders: {data['last']['totalCount']}, placed between "
            f"{first[0]['created'][:10]} and {last[0]['created'][:10]}"
        )
    return "\n".join(lines)


def _record_type(field: GraphQLField) -> GraphQLNamedType | None:
    """The object type of the records a field returns, looking inside a connection."""
    type_ = get_named_type(field.type)
    if not (is_object_type(type_) or is_interface_type(type_)):
        return None
    edges = type_.fields.get("edges")
    if edges is not None:
        node = get_named_type(edges.type).fields.get("node")
        if node is not None:
            return get_named_type(node.type)
    return type_


def _is_list(field: GraphQLField) -> bool:
    type_ = field.type.of_type if is_non_null_type(field.type) else field.type
    return is_list_type(type_) or "edges" in getattr(get_named_type(type_), "fields", {})


def _needs_arguments(field: GraphQLField) -> bool:
    """Whether a field returns nothing useful unless the caller chooses an argument.

    Besides required arguments this covers lookups such as ``product(id:, slug:)``,
    where every argument is optional but one of them has to be given.
    """
    if any(is_required_argument(argument) for argument in field.args.values()):
        return True
    selects_one = not _is_list(field) and not is_enum_type(get_named_type(field.type))
    return selects_one and bool(field.args.keys() & {"id", "slug", "token", "sku", "email"})


def _records(result: Result, root: str, connection: bool) -> list[dict[str, Any]]:
    value = (result.data or {}).get(root)
    if value is None:
        return []
    if connection:
        return [edge["node"] for edge in value["edges"]]
    return value if isinstance(value, list) else [value]


def _drop(selection: Selection, path: list[Any]) -> bool:
    """Remove the field an error path points at. Returns whether anything was removed."""
    names = [step for step in path if isinstance(step, str)]
    current = selection
    for name in names[:-1]:
        nested = current.get(name)
        if nested is None:
            return False
        current = nested
    if not names or names[-1] not in current:
        return False
    del current[names[-1]]
    return True
