"""The local model that writes requests and queries.

Any server with an OpenAI-compatible chat endpoint works, for example vLLM.
Several URLs can be given to spread the load over replicas of one model.
"""

import itertools
import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

STYLES = (
    "short commands, the way a busy shop manager types them",
    "full questions in plain English",
    "polite requests addressed to an assistant",
    "terse notes with only the essentials",
    "requests from a support agent who is looking into one specific case",
)

SYSTEM = (
    "You create training examples for an assistant that answers requests about a Saleor "
    "store by running one GraphQL query against the Saleor API."
)

QUERY_RULES = """\
- is a single read-only operation, written as `{{ ... }}` without a name, variables, fragment \
definitions or directives;
- starts at `Query.{root}` and selects nothing else at the top level;
- uses only fields, arguments and enum values that appear in the schema above, and never \
selects a field of a type that is not shown;
- uses an alias only when the same field is needed twice;
- passes `first` (20 at most) to every connection it reads;
- selects the fields the request asks for and no others."""

WRITE = """\
The part of the Saleor {version} schema you may use:

```graphql
{schema}
```

Facts about the store:
{facts}

Sample records behind `{root}`:
{records}

Write {count} examples. Each one is a request and the GraphQL query that answers it.

The query
- takes its argument values (IDs, slugs, emails, SKUs, dates, enum values) from the facts and \
the sample records, so that it returns data;
{rules}

The request
- is written by someone who works with the store and has never seen the API: it does not use \
field, type or argument names and does not mention queries;
- names every value the query returns and every condition, limit and sort order it applies, \
so that two engineers reading it would write the same query;
- asks for nothing the query does not do: no condition the query does not apply, no "if \
available", no summary or calculation;
- quotes the concrete values the query uses exactly as they are, including the ID, slug or \
other key by which the query looks a record up;
- style: {style}.

Make the examples different from each other: single lookups, filtered lists, sorted lists, \
counts, and nested data two or three levels deep.
{feature}
Answer with a JSON array and nothing else:
[{{"request": "...", "query": "..."}}]"""

SOLVE = """\
The part of the Saleor {version} schema you may use:

```graphql
{schema}
```

Facts about the store:
{facts}

Request: {question}

Write the GraphQL query that answers the request. The query
{rules}

Answer with the query and nothing else."""

PORT = """\
On Saleor {old} this request:

{question}

was answered by this query:

```graphql
{query}
```

On Saleor {new} the query does not give that answer any more: {problem}

These are the types the query uses, as Saleor {new} defines them:

```graphql
{schema}
```

Write a query for Saleor {new} that returns the same information. Keep the response keys of \
the old query: when a field has a new name, alias the new field to the old name. Use only \
what is shown above.

Answer with the query and nothing else. If Saleor {new} no longer offers this information, \
answer with the single word GONE."""

GONE = "GONE"
THINKING = re.compile(r"^.*?</think>", re.DOTALL)

PAIRS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "examples",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"request": {"type": "string"}, "query": {"type": "string"}},
                "required": ["request", "query"],
                "additionalProperties": False,
            },
        },
    },
}


@dataclass(frozen=True)
class Pair:
    question: str
    query: str


class TeacherError(RuntimeError):
    """The model server did not return a completion."""


class Teacher:
    def __init__(
        self,
        urls: Sequence[str],
        model: str = "teacher",
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        timeout: float = 600,
        transport: httpx.BaseTransport | None = None,
    ):
        if not urls:
            raise ValueError("at least one server URL is required")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._urls = itertools.cycle(url.rstrip("/") for url in urls)
        self._lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: bool = False,
    ) -> str:
        """Answer one prompt. Safe to call from several threads.

        Qwen reasons before it answers when ``thinking`` is set. That is worth
        its tokens for the few hard prompts, not for the bulk whose answers are
        checked by execution anyway.
        """
        with self._lock:
            url = next(self._urls)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens * (3 if thinking else 1),
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if response_format is not None:
            body["response_format"] = response_format
        try:
            response = self._http.post(f"{url}/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            raise TeacherError(f"{url}: {error}") from error
        # Without a reasoning parser on the server the reasoning comes inline.
        return THINKING.sub("", content)

    def close(self) -> None:
        self._http.close()


def write_prompt(
    *,
    version: str,
    root: str,
    schema: str,
    facts: str,
    records: str,
    count: int,
    style: str,
    feature: Sequence[str] = (),
) -> str:
    if feature:
        names = ", ".join(f"`{coordinate}`" for coordinate in feature)
        feature_line = f"\nAt least one example must use each of these: {names}.\n"
    else:
        feature_line = ""
    return WRITE.format(
        version=version,
        root=root,
        schema=schema,
        facts=facts or "(none)",
        records=records or "(none available; use values the schema itself offers, such as enums)",
        count=count,
        rules=QUERY_RULES.format(root=root),
        style=style,
        feature=feature_line,
    )


def solve_prompt(*, version: str, root: str, schema: str, facts: str, question: str) -> str:
    return SOLVE.format(
        version=version,
        schema=schema,
        facts=facts or "(none)",
        question=question,
        rules=QUERY_RULES.format(root=root),
    )


def port_prompt(*, old: str, new: str, question: str, query: str, problem: str, schema: str) -> str:
    return PORT.format(
        old=old, new=new, question=question, query=query, problem=problem, schema=schema
    )


def parse_pairs(text: str) -> list[Pair]:
    """Read the examples out of a completion, skipping anything malformed."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except ValueError:
        return []
    pairs = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question, query = item.get("request"), item.get("query")
        if isinstance(question, str) and isinstance(query, str) and question.strip():
            pairs.append(Pair(" ".join(question.split()), query.strip()))
    return pairs


def parse_query(text: str) -> str | None:
    """Read a single query out of a completion. ``None`` means the model declined."""
    fenced = re.search(r"```(?:graphql)?\s*(.*?)```", text, re.DOTALL)
    query = (fenced.group(1) if fenced else text).strip()
    if not query or query.upper().startswith(GONE):
        return None
    return query
