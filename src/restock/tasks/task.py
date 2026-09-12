"""A task: a request, the query that answers it and what is known about both."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from restock.tasks.focus import Focus


@dataclass
class Later:
    """How a task fares on a later release that serves the same store."""

    status: str
    """``same``, ``ported``, ``weak``, ``gone`` or ``unresolved``; see ``restock.tasks.port``."""
    query: str | None = None
    """Query for that release. A reference only when the status is ``same`` or ``ported``."""
    answer: Any = None
    """Its answer, when it is not the answer of the original query."""


@dataclass
class Task:
    id: str
    question: str
    query: str
    answer: Any
    focus: Focus
    coordinates: list[str]
    broken: list[str] = field(default_factory=list)
    """Coordinates the query uses that the target release removes or changes."""
    deprecated: list[str] = field(default_factory=list)
    """Coordinates the query uses that the target release deprecates."""
    verified: str | None = None
    """How an independent solution agreed with the answer; see ``restock.tasks.verify``."""
    later: dict[str, Later] = field(default_factory=dict)
    split: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.broken)


def task_id(query: str) -> str:
    """Identifier derived from the normal form of the reference query."""
    return hashlib.sha1(query.encode()).hexdigest()[:12]


def write_tasks(path: Path, tasks: Iterable[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def read_tasks(path: Path) -> list[Task]:
    return list(_read(path))


def _read(path: Path) -> Iterator[Task]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            focus = record.pop("focus")
            later = record.pop("later", {})
            yield Task(
                **record,
                focus=Focus(focus["root"], tuple(focus["types"]), tuple(focus["feature"])),
                later={version: Later(**entry) for version, entry in later.items()},
            )
