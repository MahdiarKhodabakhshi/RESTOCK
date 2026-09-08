import pytest

from restock.reward import RewardConfig, results_match, reward

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


def test_results_match_ignores_list_and_key_order():
    expected = {"products": [{"name": "Mug", "price": 12.0}, {"name": "Cap", "price": 8.5}]}
    actual = {"products": [{"price": 8.5, "name": "Cap"}, {"price": 12, "name": "Mug"}]}

    assert results_match(expected, actual)
    assert not results_match(expected, actual, ordered=True)


def test_results_match_keeps_duplicates_and_nulls_distinct():
    assert not results_match({"tags": ["a", "a", "b"]}, {"tags": ["a", "b", "b"]})
    assert not results_match({"product": None}, {"product": {}})
    assert not results_match({"count": 1}, {"count": "1"})
    assert not results_match({"count": 1}, {"count": True})


def test_results_match_requires_the_same_keys():
    assert not results_match({"product": {"name": "Mug"}}, {"product": {"title": "Mug"}})
    assert not results_match({"product": {"name": "Mug"}}, {"product": {"name": "Mug", "id": 1}})
