"""Generate tasks: the teacher writes requests and queries, the store checks them.

Generation runs in rounds. Each round plans a number of prompts, sends them to
the teacher in parallel and keeps the pairs that pass ``restock.tasks.check``.
What was kept steers the next round: fields no task uses yet are asked for by
name, and coordinates that change in the target release are asked for until
enough tasks use them.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from random import Random

from graphql import GraphQLSchema

from restock.saleor.client import SaleorClient
from restock.schema.diff import Change, ChangeKind
from restock.tasks.check import Checked, check, question_key
from restock.tasks.coordinates import touched, used_coordinates
from restock.tasks.focus import Focus, SchemaGraph, fields_of, render, sample_focus
from restock.tasks.task import Task, task_id
from restock.tasks.teacher import (
    PAIRS_FORMAT,
    STYLES,
    Pair,
    Teacher,
    TeacherError,
    parse_pairs,
    write_prompt,
)
from restock.tasks.values import Store, format_records

# Fields of connections and edges; every list query uses them and none needs asking for.
PLUMBING = frozenset({"edges", "node", "cursor", "pageInfo", "totalCount"})


@dataclass(frozen=True)
class Target:
    """A coordinate that changes in the target release and that a query can use."""

    coordinate: str
    broken: bool


@dataclass(frozen=True)
class Settings:
    requests: int = 1000
    """Prompts to send in total."""
    pairs: int = 6
    """Examples asked for per prompt."""
    round_size: int = 96
    workers: int = 32
    target_share: float = 0.4
    """Largest share of a round spent on changed coordinates."""
    broken_quota: int = 30
    """Tasks wanted for each coordinate the target release breaks."""
    deprecated_quota: int = 6
    broken_attempts: int = 10
    """Prompts after which a coordinate without enough tasks is given up."""
    deprecated_attempts: int = 3
    feature_fields: int = 3
    """Fields without a task that a general prompt asks for."""
    feature_attempts: int = 2
    seed: int = 0


@dataclass
class Report:
    prompts: int = 0
    failed_prompts: int = 0
    pairs: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejections: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class Rejected:
    question: str
    query: str
    rejection: str
    detail: str


def find_targets(graph: SchemaGraph, changes: Iterable[Change]) -> list[Target]:
    """Changed coordinates that some query starting at ``Query`` can use."""
    found: dict[str, Target] = {}
    for change in changes:
        if not change.query_side:
            continue
        if change.kind is ChangeKind.TYPE_REMOVED:
            fields = getattr(graph.schema.type_map[change.type_name], "fields", {})
            coordinates = [f"{change.type_name}.{name}" for name in fields]
        elif change.breaking or change.kind is ChangeKind.DEPRECATED:
            coordinates = [change.coordinate]
        else:
            continue
        for coordinate in coordinates:
            if coordinate.split(".", 1)[-1] not in PLUMBING and _reachable_sites(graph, coordinate):
                found.setdefault(coordinate, Target(coordinate, change.breaking))
    return sorted(found.values(), key=lambda target: target.coordinate)


class Generator:
    def __init__(
        self,
        schema: GraphQLSchema,
        changes: Iterable[Change],
        client: SaleorClient,
        teacher: Teacher,
        *,
        version: str,
        settings: Settings = Settings(),
        log: Callable[[str], None] = print,
    ):
        self.schema = schema
        self.changes = list(changes)
        self.client = client
        self.teacher = teacher
        self.version = version
        self.settings = settings
        self.log = log

        self.graph = SchemaGraph(schema)
        self.store = Store(schema, client)
        self.targets = find_targets(self.graph, self.changes)
        self.rng = Random(settings.seed)

        self.tasks: dict[str, Task] = {}
        self.rejected: list[Rejected] = []
        self.report = Report()
        self._questions: set[str] = set()
        self._uses: Counter = Counter()
        self._asked: Counter = Counter()
        self._root_prompts: Counter = Counter()
        self._root_accepted: Counter = Counter()
        self._weights: dict[str, float] = {}

    def add(self, tasks: Iterable[Task]) -> None:
        """Start from tasks of an earlier run, so that a new run adds to them."""
        for task in tasks:
            self._keep(task)

    def run(self, on_round: Callable[[list[Task]], None] | None = None) -> list[Task]:
        """Send all prompts. ``on_round`` gets the tasks so far after every round."""
        settings = self.settings
        remaining = settings.requests
        round_number = 0
        with ThreadPoolExecutor(settings.workers) as pool:
            while remaining > 0:
                round_number += 1
                plans = self._plan_round(min(settings.round_size, remaining))
                remaining -= len(plans)
                before = len(self.tasks)
                prompts = [self._prompt(focus) for focus in plans]
                for focus, outcome in zip(plans, pool.map(self._attempt, prompts), strict=True):
                    self._collect(focus, outcome)
                self.log(
                    f"round {round_number}: {len(plans)} prompts, "
                    f"{len(self.tasks) - before} new tasks, {len(self.tasks)} in total, "
                    f"{self.fields_used()} fields used"
                )
                if on_round is not None:
                    on_round(list(self.tasks.values()))
        return list(self.tasks.values())

    def fields_used(self) -> int:
        return sum(1 for coordinate in self._uses if "(" not in coordinate and "." in coordinate)

    # -- planning ----------------------------------------------------------------------

    def _plan_round(self, size: int) -> list[Focus]:
        plans = self._plan_targets(int(size * self.settings.target_share))
        while len(plans) < size:
            plans.append(self._plan_general())
        self.rng.shuffle(plans)
        return plans

    def _plan_targets(self, limit: int) -> list[Focus]:
        settings = self.settings
        open_targets = []
        for target in self.targets:
            quota = settings.broken_quota if target.broken else settings.deprecated_quota
            attempts = settings.broken_attempts if target.broken else settings.deprecated_attempts
            if self._uses[target.coordinate] < quota and self._asked[target.coordinate] < attempts:
                open_targets.append(target)
        self.rng.shuffle(open_targets)
        # Broken coordinates first, then whatever has been asked for least.
        open_targets.sort(key=lambda t: (not t.broken, self._asked[t.coordinate]))

        plans = []
        for target in open_targets[:limit]:
            type_name, field_name = self.rng.choice(_reachable_sites(self.graph, target.coordinate))
            if type_name == self.graph.query.name:
                focus = sample_focus(
                    self.graph, self.rng, field_name, feature=(target.coordinate,), expansions=2
                )
            else:
                root = self.rng.choice(self.graph.roots_reaching(type_name))
                focus = sample_focus(
                    self.graph,
                    self.rng,
                    root,
                    through=type_name,
                    feature=(target.coordinate,),
                    expansions=2,
                )
            self._asked[target.coordinate] += 1
            plans.append(focus)
        return plans

    def _plan_general(self) -> Focus:
        roots = self.graph.roots
        root = self.rng.choices(roots, [self._root_weight(root) for root in roots])[0]
        focus = sample_focus(self.graph, self.rng, root)

        candidates = [
            coordinate
            for coordinate in fields_of(self.schema, focus)
            if coordinate.split(".", 1)[1] not in PLUMBING
            and self._uses[coordinate] == 0
            and self._asked[coordinate] < self.settings.feature_attempts
        ]
        feature = self.rng.sample(candidates, min(self.settings.feature_fields, len(candidates)))
        self._asked.update(feature)
        return Focus(focus.root, focus.types, tuple(feature))

    def _root_weight(self, root: str) -> float:
        """Richer types get more prompts; fields that never yield a task get none."""
        prompts, accepted = self._root_prompts[root], self._root_accepted[root]
        if prompts >= 4 and accepted == 0:
            return 0.0
        weight = self._base_weight(root)
        if prompts >= 4:
            # Few accepted pairs per prompt means there is little data behind the field.
            weight *= min(1.0, 0.25 + accepted / (prompts * self.settings.pairs))
        return weight

    def _base_weight(self, root: str) -> float:
        """Number of fields of the records behind a root field."""
        if root not in self._weights:
            focus = sample_focus(self.graph, self.rng, root, expansions=0)
            record = self.schema.type_map[focus.types[-1]] if focus.types else None
            self._weights[root] = float(len(getattr(record, "fields", ())) or 1)
        return self._weights[root]

    # -- one prompt --------------------------------------------------------------------

    def _prompt(self, focus: Focus) -> str:
        return write_prompt(
            version=self.version,
            root=focus.root,
            schema=render(self.schema, focus),
            facts=self.store.facts(),
            records=format_records(self.store.records(focus.root, self.rng)),
            count=self.settings.pairs,
            style=self.rng.choice(STYLES),
            feature=focus.feature,
        )

    def _attempt(self, prompt: str) -> list[tuple[Pair, Checked]] | None:
        """Runs in a worker thread: one completion and the checks of its pairs."""
        try:
            completion = self.teacher.complete(prompt, response_format=PAIRS_FORMAT)
        except TeacherError as error:
            self.log(f"teacher failed: {error}")
            return None
        pairs = parse_pairs(completion)
        return [(pair, check(self.schema, self.client, pair.query)) for pair in pairs]

    def _collect(self, focus: Focus, outcome: list[tuple[Pair, Checked]] | None) -> None:
        self.report.prompts += 1
        self._root_prompts[focus.root] += 1
        if outcome is None:
            self.report.failed_prompts += 1
            return
        for pair, checked in outcome:
            self.report.pairs += 1
            if not checked.ok:
                self.report.rejections[checked.rejection.value] += 1
                self.rejected.append(
                    Rejected(pair.question, checked.query, checked.rejection.value, checked.detail)
                )
                continue
            coordinates = used_coordinates(self.schema, checked.document)
            changed = touched(coordinates, self.changes)
            task = Task(
                id=task_id(checked.query),
                question=pair.question,
                query=checked.query,
                answer=checked.data,
                focus=focus,
                coordinates=sorted(coordinates),
                broken=list(changed.broken),
                deprecated=list(changed.deprecated),
            )
            if self._keep(task):
                self.report.accepted += 1
                self._root_accepted[focus.root] += 1
            else:
                self.report.duplicates += 1

    def _keep(self, task: Task) -> bool:
        key = question_key(task.question)
        if task.id in self.tasks or key in self._questions:
            return False
        self.tasks[task.id] = task
        self._questions.add(key)
        self._uses.update(task.coordinates)
        return True


def _reachable_sites(graph: SchemaGraph, coordinate: str) -> list[tuple[str, str]]:
    return [
        (type_name, field_name)
        for type_name, field_name in graph.sites(coordinate)
        if (
            field_name in graph.roots
            if type_name == graph.query.name
            else graph.roots_reaching(type_name)
        )
    ]
