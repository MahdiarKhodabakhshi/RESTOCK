"""Supervised examples for the memorised-only baseline.

Each training row is the version signal plus the request, paired with the
reference query. No schema is shown, so the model has to hold the API in its
weights.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from restock.agent.prompt import SYSTEM, memorised_prompt
from restock.agent.signal import version_signal
from restock.tasks.task import Task


def sft_messages(task: Task, *, version: str) -> list[dict[str, str]]:
    """Chat messages that teach ``task.query`` from memory on ``version``."""
    signal = version_signal(version, trained_on=version).text()
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": memorised_prompt(signal=signal, question=task.question),
        },
        {"role": "assistant", "content": task.query},
    ]


def iter_sft(tasks: Iterable[Task], *, version: str) -> Iterator[dict]:
    for task in tasks:
        yield {"messages": sft_messages(task, version=version), "id": task.id}


def write_sft(path: Path, tasks: Iterable[Task], *, version: str) -> int:
    """Write OpenAI-style chat JSONL. Returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in iter_sft(tasks, version=version):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
