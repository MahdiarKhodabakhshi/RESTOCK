"""Single-turn selective reads for efficiency-RL.

The model emits zero or more ``<read>TypeName</read>`` tags and then a GraphQL
query. Reads are charged through ``SchemaTools``; the query is executed against
Saleor. This keeps one completion per episode so GRPO can score it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from restock.agent.episode import completion_query
from restock.agent.prompt import SYSTEM, selective_prompt
from restock.agent.tools import SchemaTools
from restock.reward import RewardConfig, results_match, reward
from restock.saleor.client import SaleorClient

READ_TAG = re.compile(r"<read>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</read>", re.IGNORECASE)
QUERY_BLOCK = re.compile(r"<query>\s*(.*?)\s*</query>", re.IGNORECASE | re.DOTALL)

EFFICIENCY_INSTRUCTIONS = """\
You may look up schema types before answering. To read a type, write
`<read>TypeName</read>` (as many as you need). Then write the GraphQL query
inside `<query> ... </query>`. Reading costs reward; skip reads when you already
know the API.

Example:
<read>Product</read>
<query>{ products(first: 5) { name } }</query>
"""


@dataclass(frozen=True)
class EfficiencyParse:
    reads: tuple[str, ...]
    query: str | None
    raw: str


def efficiency_prompt(*, signal: str, question: str) -> str:
    base = selective_prompt(signal=signal, question=question)
    return f"{base}\n\n{EFFICIENCY_INSTRUCTIONS}"


def parse_efficiency(text: str) -> EfficiencyParse:
    """Pull read tags and the query out of one completion."""
    reads = tuple(dict.fromkeys(READ_TAG.findall(text)))  # unique, order kept
    block = QUERY_BLOCK.search(text)
    if block:
        query = completion_query({"content": block.group(1)})
    else:
        # Fall back to a bare query with no tags.
        stripped = READ_TAG.sub("", text).strip()
        query = completion_query({"content": stripped})
    return EfficiencyParse(reads=reads, query=query, raw=text)


def apply_reads(tools: SchemaTools, reads: tuple[str, ...]) -> None:
    """Charge ``tools`` for each requested type name."""
    for name in reads:
        tools.read_type(name)


def score_completion(
    text: str,
    tools: SchemaTools,
    client: SaleorClient,
    expected: Any,
    *,
    config: RewardConfig | None = None,
) -> float:
    """Parse one completion, apply reads, execute the query, return the reward."""
    parsed = parse_efficiency(text)
    apply_reads(tools, parsed.reads)
    if not parsed.query:
        return 0.0
    result = client.execute(parsed.query)
    if not result.ok:
        return 0.0
    correct = results_match(expected, result.data)
    return reward(correct, tools.read_tokens, config or RewardConfig())


def chat_prompt(*, signal: str, question: str) -> list[dict[str, str]]:
    """Chat messages used as the GRPO ``prompt`` column."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": efficiency_prompt(signal=signal, question=question)},
    ]
