from restock.agent.tokens import count_tokens


def test_count_tokens_splits_words_and_punctuation():
    assert count_tokens("type Product { name: String! }") == 8


def test_count_tokens_treats_empty_as_zero():
    assert count_tokens("") == 0
