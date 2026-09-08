import pytest

from restock.reward import RewardConfig, reward

CONFIG = RewardConfig(read_budget=1000, cost_weight=0.4)


def test_correct_answer_from_memory_gets_full_reward():
    assert reward(True, 0, CONFIG) == 1.0


def test_reading_discounts_a_correct_answer_up_to_the_budget():
    assert reward(True, 500, CONFIG) == pytest.approx(0.8)
    assert reward(True, 1000, CONFIG) == pytest.approx(0.6)
    assert reward(True, 50_000, CONFIG) == pytest.approx(0.6)


@pytest.mark.parametrize("read_tokens", [0, 500, 50_000])
def test_wrong_answer_scores_zero_whatever_was_read(read_tokens):
    assert reward(False, read_tokens, CONFIG) == 0.0


def test_correct_after_reading_beats_wrong_from_memory():
    assert reward(True, 50_000, CONFIG) > reward(False, 0, CONFIG)


@pytest.mark.parametrize(
    "kwargs", [{"read_budget": 0}, {"cost_weight": 1.0}, {"cost_weight": -0.1}]
)
def test_config_rejects_values_that_break_the_ordering(kwargs):
    with pytest.raises(ValueError):
        RewardConfig(**kwargs)
