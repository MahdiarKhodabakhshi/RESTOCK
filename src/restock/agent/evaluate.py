"""Run a policy over a task set and summarise accuracy and reads."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from graphql import GraphQLSchema

from restock.agent.episode import DEFAULT_REWARD, EpisodeResult, Policy, run_episode
from restock.agent.model import Student
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.reward import RewardConfig
from restock.saleor.client import SaleorClient
from restock.schema.diff import Change
from restock.tasks.task import Task

Log = Callable[[str], None]


@dataclass
class BucketStats:
    episodes: int = 0
    correct: int = 0
    read_tokens: int = 0
    reward_sum: float = 0.0

    def add(self, result: EpisodeResult) -> None:
        self.episodes += 1
        self.correct += int(result.correct)
        self.read_tokens += result.read_tokens
        self.reward_sum += result.reward

    @property
    def accuracy(self) -> float:
        return self.correct / self.episodes if self.episodes else 0.0

    @property
    def mean_read_tokens(self) -> float:
        return self.read_tokens / self.episodes if self.episodes else 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.episodes if self.episodes else 0.0


@dataclass
class EvalReport:
    policy: str
    release: str
    by_bucket: dict[str, BucketStats] = field(default_factory=dict)
    results: list[EpisodeResult] = field(default_factory=list)

    @property
    def overall(self) -> BucketStats:
        total = BucketStats()
        for bucket in self.by_bucket.values():
            total.episodes += bucket.episodes
            total.correct += bucket.correct
            total.read_tokens += bucket.read_tokens
            total.reward_sum += bucket.reward_sum
        return total


def task_bucket(task: Task) -> str:
    """changed / deprecated / untouched, matching the task-set split tags."""
    if task.broken:
        return "changed"
    if task.deprecated:
        return "deprecated"
    return "untouched"


def expected_answer(task: Task, release: str) -> Any | None:
    """Reference answer on ``release``, or ``None`` when the task is not scored there."""
    if release not in task.later:
        return task.answer
    later = task.later[release]
    if later.status in ("same", "ported"):
        return task.answer if later.answer is None else later.answer
    return None


def evaluate(
    policy: Policy,
    student: Student,
    schema: GraphQLSchema,
    client: SaleorClient,
    tasks: Iterable[Task],
    *,
    release: str,
    trained_on: str | None = None,
    changes: Iterable[Change] = (),
    signal_version: str | None = None,
    config: RewardConfig = DEFAULT_REWARD,
    log: Log | None = None,
) -> EvalReport:
    """Score ``tasks`` on ``release``. Unscored later-statuses are skipped.

    ``release`` is the key used in ``task.later`` (for example ``3.23``).
    ``signal_version`` is what the prompt shows; it defaults to ``release``.
    """
    report = EvalReport(policy=policy.name, release=release)
    signal = version_signal(signal_version or release, trained_on=trained_on, changes=changes)
    for task in tasks:
        expected = expected_answer(task, release)
        if expected is None:
            continue
        tools = SchemaTools(schema, signal)
        result = run_episode(policy, student, tools, client, task.question, expected, config=config)
        bucket = task_bucket(task)
        report.by_bucket.setdefault(bucket, BucketStats()).add(result)
        report.results.append(result)
        if log is not None:
            mark = "ok" if result.correct else "fail"
            log(
                f"{task.id}  {bucket:<10} {mark:<4}  "
                f"read={result.read_tokens}  reward={result.reward:.3f}"
            )
    return report


def format_report(report: EvalReport) -> str:
    """Plain-text summary: overall plus each bucket."""
    lines = [f"policy={report.policy}  release={report.release}"]
    order = ["changed", "deprecated", "untouched"]
    buckets = list(report.by_bucket)
    for name in order + sorted(set(buckets) - set(order)):
        stats = report.by_bucket.get(name)
        if stats is None:
            continue
        lines.append(_line(name, stats))
    lines.append(_line("overall", report.overall))
    return "\n".join(lines)


def _line(name: str, stats: BucketStats) -> str:
    return (
        f"{name:<10}  n={stats.episodes:<5}  "
        f"acc={stats.accuracy:.3f}  "
        f"read={stats.mean_read_tokens:.0f}  "
        f"reward={stats.mean_reward:.3f}"
    )


def iter_scored(tasks: Iterable[Task], release: str) -> Iterator[Task]:
    for task in tasks:
        if expected_answer(task, release) is not None:
            yield task
