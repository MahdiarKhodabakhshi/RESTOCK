"""Prompts for the GraphQL-writing baselines."""

from restock.tasks.teacher import parse_query

SYSTEM = (
    "You write one GraphQL query that answers a request about a Saleor store. "
    "The query is executed against the Saleor API; you do not invent data."
)

QUERY_INSTRUCTIONS = """\
The query
- is a single read-only operation that starts with { and ends with }, without a name, \
variables, fragment definitions or directives;
- uses only fields, arguments and enum values that the schema you were given allows;
- passes `first` (20 at most) to every connection it reads;
- selects the fields the request asks for and no others.

Answer with the query and nothing else."""

MEMORISED = """\
{signal}

Request: {question}

Write the GraphQL query that answers the request from what you know of the Saleor API.
{instructions}"""

ALWAYS_READ = """\
{signal}

Saleor schema (query side):

```graphql
{schema}
```

Request: {question}

Write the GraphQL query that answers the request. Use only the schema above.
{instructions}"""

SELECTIVE = """\
{signal}

You may look up parts of the Saleor schema with the tools before you answer.
Read only what you need; every read costs tokens.

Request: {question}

When you know the query, answer with the query and nothing else.
{instructions}"""


def memorised_prompt(*, signal: str, question: str) -> str:
    return MEMORISED.format(signal=signal, question=question, instructions=QUERY_INSTRUCTIONS)


def always_read_prompt(*, signal: str, schema: str, question: str) -> str:
    return ALWAYS_READ.format(
        signal=signal, schema=schema, question=question, instructions=QUERY_INSTRUCTIONS
    )


def selective_prompt(*, signal: str, question: str) -> str:
    return SELECTIVE.format(signal=signal, question=question, instructions=QUERY_INSTRUCTIONS)


__all__ = [
    "SYSTEM",
    "always_read_prompt",
    "memorised_prompt",
    "parse_query",
    "selective_prompt",
]
