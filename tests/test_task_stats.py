from restock.tasks.focus import Focus
from restock.tasks.stats import kind, summary
from restock.tasks.task import Later, Task


def task(number: int, split: str | None, query: str = "{ shop { name } }", **fields) -> Task:
    return Task(
        id=str(number),
        question=f"Request {number}",
        query=query,
        answer={},
        focus=Focus("shop", ()),
        split=split,
        **{"coordinates": ["Query.shop", "Shop.name"], **fields},
    )


def test_kind():
    assert kind(task(0, None)) == "untouched"
    assert kind(task(0, None, deprecated=["Shop.description"])) == "deprecated"
    assert kind(task(0, None, broken=["Product.rating"], deprecated=["Shop.description"])) == (
        "changed"
    )


def test_summary(shop_schema):
    rating = ["Query.product", "Query.product(slug:)", "Product.rating"]
    tasks = [
        task(0, "train", verified="exact"),
        task(1, "train", verified="exact", deprecated=["Shop.description"]),
        task(
            2,
            "train",
            '{ product(slug: "alpine-tee") { rating } }',
            verified="values",
            coordinates=rating,
            broken=["Product.rating"],
        ),
        task(3, "test", verified="exact", later={"2.0": Later("same")}),
        task(
            4,
            "test",
            '{ product(slug: "harbour-mug") { rating } }',
            verified="exact",
            coordinates=rating,
            broken=["Product.rating"],
            later={"2.0": Later("gone")},
        ),
        task(5, None, verified="failed", coordinates=["Query.currencies"]),
    ]

    lines = summary(tasks, shop_schema)

    assert lines[0] == "tasks: 6"
    # Six tasks, but the two rating lookups differ only in the product they name.
    assert lines[1] == "query shapes: 2"
    assert lines[2] == "verification: exact 4, values 1, failed 1"
    assert lines[4].split() == ["split", "tasks", "changed", "deprecated", "untouched"]
    assert lines[5].split() == ["train", "3", "1", "1", "1"]
    assert lines[6].split() == ["test", "2", "1", "0", "1"]
    assert lines[7].split() == ["none", "1", "0", "0", "1"]
    assert lines[9] == "on 2.0, test set:"
    assert [line.split() for line in lines[10:13]] == [
        ["changed", "gone", "1"],
        ["deprecated", "-"],
        ["untouched", "same", "1"],
    ]
    assert lines[13] == "on 2.0, test set, by changed coordinate:"
    assert lines[14].split() == ["Product.rating", "gone", "1"]
    # Tasks without a split are not part of the set and do not count as coverage.
    assert lines[-2].startswith("fields used: 4 of ")
    assert lines[-1] == "changed coordinates with a task: 1"
