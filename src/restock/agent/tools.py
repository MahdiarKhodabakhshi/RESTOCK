"""Schema-reading tools with cumulative token accounting.

An episode starts with a ``SchemaTools`` bound to the schema the agent may see
(real or mutated). ``read_type`` and ``search_fields`` return text and add the
token cost to ``read_tokens``. The version signal lives on the same object so
baselines share one handle, but reading it does not charge the budget.
"""

from dataclasses import dataclass, field

from graphql import GraphQLNamedType, GraphQLSchema, is_specified_scalar_type

from restock.agent.render import render_type
from restock.agent.signal import VersionSignal
from restock.agent.tokens import TokenCounter, count_tokens
from restock.schema.diff import reachable_types


@dataclass(frozen=True)
class ToolResult:
    """Text a tool returned and how many tokens it cost."""

    text: str
    tokens: int


@dataclass
class SchemaTools:
    """Look up types and search fields on one schema, counting every read."""

    schema: GraphQLSchema
    signal: VersionSignal
    count_tokens: TokenCounter = field(default=count_tokens, repr=False)
    _read_tokens: int = field(default=0, init=False, repr=False)
    _query_side: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._query_side = reachable_types(self.schema, self.schema.query_type)

    @property
    def read_tokens(self) -> int:
        return self._read_tokens

    def version_signal(self) -> str:
        """Prompt text for the API version. Not counted as a schema read."""
        return self.signal.text()

    def read_type(self, name: str) -> ToolResult:
        """Return the compact SDL for a named type and charge its tokens."""
        type_ = self._named(name)
        if type_ is None:
            return self._charge(f"Type not found: {name}")
        return self._charge(render_type(type_))

    def search_fields(self, query: str, *, limit: int = 40) -> ToolResult:
        """Find query-side fields whose type or field name contains ``query``.

        Matches are listed as ``Type.field: ReturnType`` lines. An empty query
        returns nothing so callers cannot dump the whole schema by accident.
        """
        needle = query.strip()
        if not needle:
            return self._charge("No matches.")
        needle_lower = needle.lower()
        hits: list[str] = []
        for type_name in sorted(self._query_side):
            type_ = self.schema.type_map[type_name]
            fields = getattr(type_, "fields", None)
            if not fields:
                continue
            for field_name, gql_field in fields.items():
                coordinate = f"{type_name}.{field_name}"
                if needle_lower in type_name.lower() or needle_lower in field_name.lower():
                    hits.append(f"{coordinate}: {gql_field.type}")
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        if not hits:
            return self._charge(f"No fields matching {needle!r}.")
        return self._charge("\n".join(hits))

    def _named(self, name: str) -> GraphQLNamedType | None:
        type_ = self.schema.type_map.get(name)
        if type_ is None or name.startswith("__") or is_specified_scalar_type(type_):
            return None
        return type_

    def _charge(self, text: str) -> ToolResult:
        tokens = self.count_tokens(text)
        self._read_tokens += tokens
        return ToolResult(text=text, tokens=tokens)
