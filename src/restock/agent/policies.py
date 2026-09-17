"""Baseline policies that write a GraphQL query for one request."""

import json
import re
from typing import Any

from graphql import get_named_type, is_enum_type, is_input_object_type

from restock.agent.episode import completion_query
from restock.agent.model import Student
from restock.agent.prompt import (
    SYSTEM,
    always_read_prompt,
    memorised_prompt,
    selective_prompt,
)
from restock.agent.tools import SchemaTools

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_type",
            "description": "Return the compact SDL for one named GraphQL type.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fields",
            "description": "Search query-side fields by a substring of the type or field name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "from",
        "this",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "does",
        "did",
        "not",
        "but",
        "what",
        "which",
        "when",
        "where",
        "how",
        "who",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "about",
        "into",
        "over",
        "such",
        "than",
        "then",
        "them",
        "they",
        "their",
        "its",
        "list",
        "show",
        "give",
        "find",
        "get",
        "all",
        "any",
        "first",
        "last",
        "each",
        "only",
        "also",
        "just",
        "like",
        "using",
        "used",
        "use",
        "request",
        "please",
        "tell",
        "me",
    }
)


class MemorisedOnly:
    """Answer from weights alone; no schema is shown and no tools are offered."""

    name = "memorised"

    def answer(self, student: Student, tools: SchemaTools, question: str) -> str | None:
        prompt = memorised_prompt(signal=tools.version_signal(), question=question)
        return completion_query(student.complete(prompt))


class AlwaysRead:
    """Look up schema for the request, charge those reads, then write a query.

    Keywords from the question are searched on the query side and the matching
    types are read. That is the always-read expert: schema comes from tools on
    every episode, never from memory alone.
    """

    name = "always_read"

    def __init__(self, *, max_types: int = 25, max_keywords: int = 12):
        self.max_types = max_types
        self.max_keywords = max_keywords

    def answer(self, student: Student, tools: SchemaTools, question: str) -> str | None:
        schema = "\n\n".join(self.schema_for(tools, question))
        prompt = always_read_prompt(signal=tools.version_signal(), schema=schema, question=question)
        return completion_query(student.complete(prompt))

    def schema_for(self, tools: SchemaTools, question: str) -> list[str]:
        """Compact SDL slices gathered for ``question`` (charged on ``tools``).

        Field argument input types and enums are pulled in with each hit, so a
        lookup for ``appExtensions`` also shows ``AppExtensionFilterInput``.
        """
        seen: set[str] = set()
        texts: list[str] = []

        def remember(type_name: str) -> None:
            if type_name in seen or len(seen) >= self.max_types:
                return
            seen.add(type_name)
            texts.append(tools.read_type(type_name).text)

        for word in _keywords(question)[: self.max_keywords]:
            hits = tools.search_fields(word, limit=15).text
            for line in hits.splitlines():
                type_name, field_name = _hit_parts(line)
                if type_name is None:
                    continue
                remember(type_name)
                for related in _argument_types(tools, type_name, field_name):
                    remember(related)
                if len(seen) >= self.max_types:
                    return texts
        if not texts:
            remember("Query")
        return texts


class Selective:
    """Let the model call schema tools, then write a query.

    Used for efficiency-RL and RESTOCK. Stops after ``max_turns`` tool rounds
    or when the model returns content without further tool calls.
    """

    name = "selective"

    def __init__(self, *, max_turns: int = 8):
        self.max_turns = max_turns

    def answer(self, student: Student, tools: SchemaTools, question: str) -> str | None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": selective_prompt(signal=tools.version_signal(), question=question),
            },
        ]
        for _ in range(self.max_turns):
            message = student.complete("", tools=TOOL_SPECS, messages=messages)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return completion_query(message)
            messages.append(_assistant_with_tools(message))
            for call in tool_calls:
                messages.append(_tool_message(tools, call))
        return None


def _keywords(question: str) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD.finditer(question):
        word = match.group(0)
        key = word.lower()
        if key in _STOP or key in seen:
            continue
        seen.add(key)
        words.append(word)
    return words


def _hit_parts(line: str) -> tuple[str | None, str | None]:
    if ":" not in line or line.startswith("No ") or line.startswith("Type not"):
        return None, None
    coordinate = line.split(":", 1)[0].strip().split("(", 1)[0]
    if "." not in coordinate:
        return None, None
    type_name, field_name = coordinate.split(".", 1)
    return type_name, field_name


def _argument_types(
    tools: SchemaTools, type_name: str, field_name: str | None = None
) -> list[str]:
    """Input-object and enum names used as arguments on a type (or one field)."""
    type_ = tools.schema.type_map.get(type_name)
    fields = getattr(type_, "fields", None)
    if not fields:
        return []
    if field_name is not None:
        field = fields.get(field_name)
        selected = [field] if field is not None else []
    else:
        selected = list(fields.values())
    found: list[str] = []
    seen: set[str] = set()
    for field in selected:
        args = getattr(field, "args", None)
        if not args:
            continue
        for arg in args.values():
            named = get_named_type(arg.type)
            if named.name in seen:
                continue
            if is_input_object_type(named) or is_enum_type(named):
                seen.add(named.name)
                found.append(named.name)
    return found


def _assistant_with_tools(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": message["tool_calls"],
    }


def _tool_message(tools: SchemaTools, call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    name = function.get("name", "")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except ValueError:
        arguments = {}
    if name == "read_type":
        result = tools.read_type(str(arguments.get("name", "")))
    elif name == "search_fields":
        limit = arguments.get("limit", 40)
        try:
            limit_n = int(limit)
        except (TypeError, ValueError):
            limit_n = 40
        result = tools.search_fields(str(arguments.get("query", "")), limit=limit_n)
    else:
        result = tools.observe(f"Unknown tool: {name}")
    return {
        "role": "tool",
        "tool_call_id": call.get("id", name),
        "content": result.text,
    }
