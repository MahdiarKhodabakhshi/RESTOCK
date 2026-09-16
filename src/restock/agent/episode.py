"""One episode: a policy writes a query, the store runs it, the reward scores it."""

from dataclasses import dataclass
from typing import Any, Protocol

from restock.agent.model import Student
from restock.agent.prompt import parse_query
from restock.agent.tools import SchemaTools
from restock.reward import RewardConfig, results_match, reward
from restock.saleor.client import SaleorClient

DEFAULT_REWARD = RewardConfig()


class Policy(Protocol):
    """How a baseline turns a question into a GraphQL query."""

    name: str

    def answer(self, student: Student, tools: SchemaTools, question: str) -> str | None:
        """Return a query, or ``None`` if the model produced nothing usable."""


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    question: str
    query: str | None
    read_tokens: int
    correct: bool
    reward: float
    answer: Any = None
    error: str | None = None


def run_episode(
    policy: Policy,
    student: Student,
    tools: SchemaTools,
    client: SaleorClient,
    question: str,
    expected: Any,
    *,
    config: RewardConfig = DEFAULT_REWARD,
) -> EpisodeResult:
    """Ask the policy for a query, execute it and score the episode."""
    try:
        query = policy.answer(student, tools, question)
    except Exception as error:  # noqa: BLE001 - episode must record the failure
        return EpisodeResult(
            policy=policy.name,
            question=question,
            query=None,
            read_tokens=tools.read_tokens,
            correct=False,
            reward=0.0,
            error=str(error),
        )

    if not query:
        return EpisodeResult(
            policy=policy.name,
            question=question,
            query=None,
            read_tokens=tools.read_tokens,
            correct=False,
            reward=0.0,
            error="no query",
        )

    result = client.execute(query)
    if not result.ok:
        return EpisodeResult(
            policy=policy.name,
            question=question,
            query=query,
            read_tokens=tools.read_tokens,
            correct=False,
            reward=0.0,
            error="; ".join(result.messages) or "execution failed",
        )

    correct = results_match(expected, result.data)
    return EpisodeResult(
        policy=policy.name,
        question=question,
        query=query,
        read_tokens=tools.read_tokens,
        correct=correct,
        reward=reward(correct, tools.read_tokens, config),
        answer=result.data,
    )


def completion_query(message: dict[str, Any]) -> str | None:
    """Extract a GraphQL query from an assistant message."""
    query = parse_query(message.get("content") or "")
    if query is None:
        return None
    stripped = query.lstrip().lower()
    if not (stripped.startswith("{") or stripped.startswith("query")):
        return None
    return query
