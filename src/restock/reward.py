"""Episode reward: get the answer right, and read as little schema as that takes."""

from dataclasses import dataclass


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
