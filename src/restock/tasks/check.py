"""Decide whether a query written by the teacher can serve as a reference.

A reference query has to be valid, run without errors and return something: an
empty answer is matched by too many wrong queries to tell anything. It also has
to give the same answer twice, which rules out tokens and other values that
change with every request.
"""

import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from graphql import (
    DocumentNode,
    GraphQLError,
    GraphQLSchema,
    OperationDefinitionNode,
    OperationType,
    parse,
    print_ast,
    validate,
)

from restock.saleor.client import SaleorClient, SaleorError

MAX_ANSWER_BYTES = 20_000
# Each release is served on its own port, which shows up in media and download URLs.
LOCAL_URL = re.compile(r"http://(?:127\.0\.0\.1|localhost):\d+/")
STABLE_URL = "http://saleor.local/"


class Rejection(StrEnum):
    SYNTAX = "syntax"
    NOT_ONE_QUERY = "not_one_query"
    VARIABLES = "variables"
    INVALID = "invalid"
    ERROR = "error"
    EMPTY = "empty"
    TOO_LARGE = "too_large"
    UNSTABLE = "unstable"


@dataclass(frozen=True)
class Checked:
    query: str
    """The query in normal form, or as given when it could not be parsed."""
    document: DocumentNode | None = None
    data: Any = None
    rejection: Rejection | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.rejection is None


def check(
    schema: GraphQLSchema,
    client: SaleorClient,
    query: str,
    *,
    require_content: bool = True,
    repeat: bool = True,
) -> Checked:
    """Validate ``query`` against ``schema`` and run it on the server behind ``client``."""
    try:
        document = normal_form(query)
    except GraphQLError as error:
        return Checked(query, rejection=Rejection.SYNTAX, detail=error.message)
    checked = Checked(print_ast(document), document)

    shape = _shape_problem(document)
    if shape is not None:
        return replace(checked, rejection=shape)
    errors = validate(schema, document)
    if errors:
        detail = "; ".join(error.message for error in errors)
        return replace(checked, rejection=Rejection.INVALID, detail=detail)

    try:
        data, messages = run(client, checked.query)
    except (SaleorError, OSError) as error:
        return replace(checked, rejection=Rejection.ERROR, detail=str(error))
    if messages:
        return replace(checked, rejection=Rejection.ERROR, detail="; ".join(messages))
    if require_content and not has_content(data):
        return replace(checked, rejection=Rejection.EMPTY)
    if len(json.dumps(data)) > MAX_ANSWER_BYTES:
        return replace(checked, rejection=Rejection.TOO_LARGE)
    if repeat and run(client, checked.query)[0] != data:
        return replace(checked, rejection=Rejection.UNSTABLE)
    return replace(checked, data=data)


def run(client: SaleorClient, query: str) -> tuple[Any, list[str]]:
    """Execute a query and return its data, comparable across releases, and any errors."""
    result = client.execute(query)
    return stable(result.data), result.messages


def normal_form(query: str) -> DocumentNode:
    """Parse a query and drop what does not change its meaning: layout and operation name."""
    document = parse(query, no_location=True)
    definitions = tuple(
        replace(definition, name=None)
        if isinstance(definition, OperationDefinitionNode)
        else definition
        for definition in document.definitions
    )
    return replace(document, definitions=definitions)


def normalise(query: str) -> str:
    return print_ast(normal_form(query))


def stable(value: Any) -> Any:
    """Replace the address of the local server in URLs by a fixed one."""
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable(item) for item in value]
    if isinstance(value, str):
        return LOCAL_URL.sub(STABLE_URL, value)
    return value


def has_content(data: Any) -> bool:
    """Whether every top-level field of an answer holds at least one value.

    Nulls, empty lists and empty strings are not values, and neither are a count
    of zero or the paging details of a connection.
    """
    if not isinstance(data, dict) or not data:
        return False
    return all(_prune(value) is not None for value in data.values())


def question_key(question: str) -> str:
    """Form of a question in which trivial rewordings compare equal."""
    return " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())


def _shape_problem(document: DocumentNode) -> Rejection | None:
    if len(document.definitions) != 1:
        return Rejection.NOT_ONE_QUERY
    operation = document.definitions[0]
    if (
        not isinstance(operation, OperationDefinitionNode)
        or operation.operation is not OperationType.QUERY
    ):
        return Rejection.NOT_ONE_QUERY
    if operation.variable_definitions:
        return Rejection.VARIABLES
    return None


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if key == "pageInfo" or (key == "totalCount" and item == 0):
                continue
            item = _prune(item)
            if item is not None:
                kept[key] = item
        return kept or None
    if isinstance(value, list):
        kept = [item for item in map(_prune, value) if item is not None]
        return kept or None
    if value is None or value == "":
        return None
    return value
