from restock.tasks.focus import Focus
from restock.tasks.task import Later, Task, read_tasks, task_id, write_tasks


def test_task_id_depends_only_on_the_query():
    assert task_id("{ shop { name } }") == task_id("{ shop { name } }")
    assert task_id("{ shop { name } }") != task_id("{ shop { description } }")
    assert len(task_id("{ shop { name } }")) == 12


def test_tasks_survive_a_round_trip(tmp_path):
    tasks = [
        Task(
            id="a1",
            question="Name of the shop “Zoë”?",
            query="{\n  shop {\n    name\n  }\n}",
            answer={"shop": {"name": "Zoë"}},
            focus=Focus("shop", ("Shop",), ("Shop.name",)),
            coordinates=["Query.shop", "Shop.name"],
            broken=["Shop.name"],
            verified="exact",
            later={"3.23": Later("ported", "{ shop { title } }", {"shop": {"title": "Zoë"}})},
            split="test",
        ),
        Task(
            id="b2",
            question="How many products?",
            query="{ products { totalCount } }",
            answer={"products": {"totalCount": 2}},
            focus=Focus("products", ()),
            coordinates=["Query.products"],
        ),
    ]
    path = tmp_path / "nested" / "tasks.jsonl"

    write_tasks(path, tasks)

    assert read_tasks(path) == tasks
    assert "Zoë" in path.read_text(encoding="utf-8")


def test_a_task_is_changed_when_it_uses_something_broken():
    task = Task("a", "q", "{ shop { name } }", {}, Focus("shop", ()), [])
    assert not task.changed
    task.broken.append("Shop.name")
    assert task.changed
