from collections import Counter

from restock.tasks.focus import Focus
from restock.tasks.split import TEST, TRAIN, assign_splits, group_tasks, query_shape
from restock.tasks.task import Task


def task(number: int, query: str, question: str | None = None, **fields) -> Task:
    return Task(
        id=str(number),
        question=question or f"Request number {number}",
        query=query,
        answer={},
        focus=Focus("shop", ()),
        coordinates=[],
        **fields,
    )


def distinct(count: int, start: int = 0, **fields) -> list[Task]:
    """Tasks that share neither request nor query shape."""
    return [task(n, f"{{ shop {{ field{n} }} }}", **fields) for n in range(start, start + count)]


def test_query_shape_blanks_values_but_not_structure():
    by_slug = query_shape('{ product(slug: "alpine-tee") { name } }')
    assert by_slug == query_shape('{ product(slug: "harbour-mug") { name } }')
    assert by_slug != query_shape('{ product(id: "alpine-tee") { name } }')
    assert by_slug != query_shape('{ product(slug: "alpine-tee") { slug } }')

    assert query_shape("{ products(first: 5) { totalCount } }") == query_shape(
        "{products(first: 20){totalCount}}"
    )
    assert query_shape("{ report(period: TODAY) { amount } }") != query_shape(
        "{ report(period: THIS_MONTH) { amount } }"
    )
    assert query_shape("{ products(filter: { price: { gte: 1.5 } }) { totalCount } }") == (
        query_shape("{ products(filter: { price: { gte: 20.0 } }) { totalCount } }")
    )


def test_tasks_with_the_same_shape_or_request_form_a_group():
    tasks = [
        task(0, '{ product(slug: "a") { name } }'),
        task(1, "{ shop { name } }", question="What is the shop called?"),
        task(2, '{ product(slug: "b") { name } }'),
        task(3, "{ shop { description } }", question="what is the shop called"),
        task(4, "{ shop { description } }"),
        task(5, "{ currencies }"),
    ]

    groups = [[t.id for t in group] for group in group_tasks(tasks)]

    assert groups == [["0", "2"], ["1", "3", "4"], ["5"]]


def test_groups_are_never_divided():
    tasks = []
    for number in range(40):
        tasks.append(task(2 * number, f'{{ product(slug: "a{number}") {{ field{number} }} }}'))
        tasks.append(task(2 * number + 1, f'{{ product(slug: "b{number}") {{ field{number} }} }}'))

    assign_splits(tasks, test_share=0.25)

    for first, second in zip(tasks[::2], tasks[1::2], strict=True):
        assert first.split == second.split
    assert Counter(t.split for t in tasks) == {TRAIN: 60, TEST: 20}


def test_changed_tasks_get_their_own_share():
    tasks = distinct(100) + distinct(20, start=100, broken=["Product.rating"])

    assign_splits(tasks, test_share=0.1, changed_test_share=0.5)

    assert Counter(t.split for t in tasks if not t.changed) == {TRAIN: 90, TEST: 10}
    assert Counter(t.split for t in tasks if t.changed) == {TRAIN: 10, TEST: 10}


def test_a_group_with_a_changed_task_counts_as_changed():
    tasks = [
        task(0, '{ product(slug: "a") { rating } }', broken=["Product.rating"]),
        task(1, '{ product(slug: "b") { rating } }'),
    ]

    assign_splits(tasks, test_share=0.0, changed_test_share=1.0)

    assert [t.split for t in tasks] == [TEST, TEST]


def test_tasks_that_failed_verification_get_no_split():
    tasks = distinct(10, verified="exact") + distinct(5, start=10, verified="failed")
    tasks += distinct(5, start=15)
    tasks[12].split = TRAIN

    assign_splits(tasks, test_share=0.2)

    assert [t.split for t in tasks[10:15]] == [None] * 5
    assert Counter(t.split for t in tasks[:10] + tasks[15:]) == {TRAIN: 12, TEST: 3}


def test_the_split_depends_only_on_the_seed():
    def splits(seed: int) -> list[str]:
        tasks = distinct(60)
        assign_splits(tasks, seed=seed)
        return [t.split for t in tasks]

    assert splits(0) == splits(0)
    assert splits(0) != splits(1)
