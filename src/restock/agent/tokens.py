"""Token counts for schema text the agent chooses to read.

The reward discounts by how much schema was read, so every tool return needs a
stable count. This default does not depend on a model tokenizer: it splits on
whitespace and punctuation the way a rough English BPE would. Baselines that
must match a specific model can pass their own counter into ``SchemaTools``.
"""

import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Number of tokens in ``text`` under the default counter."""
    return sum(1 for _ in _TOKEN.finditer(text))
