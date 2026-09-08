"""Episode reward: get the answer right, and read as little schema as that takes."""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    read_budget: int = 4000
    """Tokens of schema read at which the cost stops growing."""
    cost_weight: float = 0.3
    """Share of the reward a correct answer loses when it uses the whole budget."""

    def __post_init__(self) -> None:
        if self.read_budget <= 0:
            raise ValueError("read_budget must be positive")
        if not 0 <= self.cost_weight < 1:
            raise ValueError("cost_weight must be in [0, 1)")


def reward(correct: bool, read_tokens: int, config: RewardConfig = RewardConfig()) -> float:
    """Reward for one episode.

    Reading only discounts a correct answer and the discount is below one, so a
    correct answer after reading always beats a wrong answer from memory. A
    wrong answer scores zero however little was read, which removes the
    incentive to stop reading altogether.
    """
    if not correct:
        return 0.0
    used = min(max(read_tokens, 0) / config.read_budget, 1.0)
    return 1.0 - config.cost_weight * used


def results_match(expected: Any, actual: Any, *, ordered: bool = False) -> bool:
    """Compare two GraphQL response payloads.

    Lists are compared as multisets unless ``ordered`` is set, since most tasks
    do not ask for a sort order and the server is then free to pick one.
    """
    # Compared as JSON text so that `true` and `1` stay different values.
    return _sort_key(_canonical(expected, ordered)) == _sort_key(_canonical(actual, ordered))


def _canonical(value: Any, ordered: bool) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(item, ordered) for key, item in value.items()}
    if isinstance(value, list):
        items = [_canonical(item, ordered) for item in value]
        return items if ordered else sorted(items, key=_sort_key)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)
