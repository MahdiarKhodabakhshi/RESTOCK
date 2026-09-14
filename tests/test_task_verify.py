from conftest import ScriptedTeacher

from restock.tasks.focus import Focus
from restock.tasks.task import Task
from restock.tasks.verify import EXACT, FAILED, VALUES, agreement, leaf_values, verify


def task(question: str, query: str, answer, root: str = "shop", **fields) -> Task:
    return Task(
        id=question,
        question=question,
        query=query,
        answer=answer,
        focus=Focus(root, (), ("Shop.name",)),
        coordinates=[],
        **fields,
    )


def test_agreement():
    answer = {"products": {"edges": [{"node": {"name": "Tee"}}, {"node": {"name": "Mug"}}]}}
    reordered = {"products": {"edges": [{"node": {"name": "Mug"}}, {"node": {"name": "Tee"}}]}}
    aliased = {"items": {"edges": [{"node": {"title": "Tee"}}, {"node": {"title": "Mug"}}]}}
    longer = {"products": {"edges": [{"node": {"name": "Tee", "id": "P1"}}]}}

    assert agreement(answer, answer) == EXACT
    assert agreement(answer, reordered) == EXACT
    assert agreement(answer, aliased) == VALUES
    assert agreement(answer, longer) == FAILED
    assert agreement(answer, None) == FAILED


def test_leaf_values_ignore_keys_shape_and_nulls():
    assert leaf_values({"a": {"b": [1, "x", None]}, "c": True}) == ['"x"', "1", "true"]
    assert leaf_values({"a": 1}) != leaf_values({"a": "1"})
    assert leaf_values(None) == []


def test_verify_marks_how_the_second_query_agrees(shop_schema, shop_client):
    tasks = [
        task("Name of the shop?", "{ shop { name } }", {"shop": {"name": "Sample shop"}}),
        task("Shop name?", "{ shop { title: name } }", {"shop": {"title": "Sample shop"}}),
        task("Which currencies?", "{ currencies }", {"currencies": ["USD"]}),
        task("Owner of the shop?", "{ shop { name } }", {"shop": {"name": "Sample shop"}}),
        task("Checked before", "{ shop { name } }", {}, verified=FAILED),
    ]

    def reply(prompt: str) -> str:
        if "Owner of the shop?" in prompt:
            return "{ shop { owner } }"
        if "Which currencies?" in prompt:
            return "GONE"
        return "```graphql\n{ shop { name } }\n```"

    teacher = ScriptedTeacher(reply)
    outcomes = verify(
        tasks, shop_schema, shop_client, teacher, version="1.0", log=lambda message: None
    )

    assert [t.verified for t in tasks] == [EXACT, VALUES, FAILED, FAILED, FAILED]
    assert dict(outcomes) == {EXACT: 1, VALUES: 1, FAILED: 2}
    assert len(teacher.prompts) == 4
    assert all(options == {"temperature": 0.0} for options in teacher.options)


def test_progress_is_saved_along_the_way(shop_schema, shop_client, monkeypatch):
    from restock.tasks import verify as verify_module

    monkeypatch.setattr(verify_module, "CHECKPOINT_EVERY", 2)
    tasks = [
        task(f"Name of the shop, {n}?", "{ shop { name } }", {"shop": {"name": "Sample shop"}})
        for n in range(5)
    ]
    saved = []

    verify(
        tasks,
        shop_schema,
        shop_client,
        ScriptedTeacher(lambda prompt: "{ shop { name } }"),
        version="1.0",
        log=lambda message: None,
        checkpoint=lambda: saved.append(sum(t.verified is not None for t in tasks)),
    )

    assert saved == [2, 4, 5]


def test_the_prompt_shows_the_request_but_not_what_was_featured(shop_schema, shop_client):
    teacher = ScriptedTeacher(lambda prompt: "{ shop { name } }")
    tasks = [task("Name of the shop?", "{ shop { name } }", {"shop": {"name": "Sample shop"}})]

    verify(tasks, shop_schema, shop_client, teacher, version="1.0")

    assert "Request: Name of the shop?" in teacher.prompts[0]
    assert "type Query {\n  shop: Shop!" in teacher.prompts[0]
    assert "Shop.name" not in teacher.prompts[0]


def test_an_empty_answer_can_still_agree(shop_schema, shop_client):
    # The second query is compared, not judged: it may return what the first returned.
    tasks = [task("Shop description?", "{ shop { description } }", {"shop": {"description": None}})]
    teacher = ScriptedTeacher(lambda prompt: "{ shop { description } }")

    verify(tasks, shop_schema, shop_client, teacher, version="1.0")

    assert tasks[0].verified == EXACT
