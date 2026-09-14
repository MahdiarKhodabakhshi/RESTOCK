import pytest
from conftest import SHOP_DATA, ScriptedTeacher, store_client

from restock.tasks.focus import Focus
from restock.tasks.port import (
    GONE,
    PORTED,
    SAME,
    UNRESOLVED,
    WEAK,
    informative,
    original_problem,
    port,
)
from restock.tasks.task import Later, Task

SKU_QUERY = '{ product(slug: "alpine-tee") { variants { sku } } }'
SKU_ANSWER = {"product": {"variants": [{"sku": "RS-0001-S"}]}}
SKU_COORDINATES = ["Product.variants", "Query.product", "Query.product(slug:)", "Variant.sku"]


def task(query: str, answer, coordinates=(), broken=()) -> Task:
    return Task(
        id=query,
        question=f"Request for {query}",
        query=query,
        answer=answer,
        focus=Focus("shop", ()),
        coordinates=list(coordinates),
        broken=list(broken),
    )


def settle(tasks, new_shop_schema, client, reply) -> tuple[ScriptedTeacher, dict]:
    teacher = ScriptedTeacher(reply)
    outcomes = port(
        tasks, new_shop_schema, client, teacher, old="1.0", new="2.0", log=lambda message: None
    )
    return teacher, dict(outcomes)


def test_a_query_that_still_gives_the_answer_stays(new_shop_schema, new_shop_client):
    tasks = [task("{ shop { name } }", {"shop": {"name": "Sample shop"}})]

    teacher, outcomes = settle(tasks, new_shop_schema, new_shop_client, lambda prompt: "GONE")

    assert tasks[0].later == {"2.0": Later(SAME, "{ shop { name } }")}
    assert outcomes == {SAME: 1}
    assert teacher.prompts == []


def test_a_renamed_field_is_ported_with_an_alias(new_shop_schema, new_shop_client):
    tasks = [task(SKU_QUERY, SKU_ANSWER, SKU_COORDINATES, broken=["Variant.sku"])]
    ported = '{ product(slug: "alpine-tee") { variants { sku: code } } }'

    teacher, outcomes = settle(tasks, new_shop_schema, new_shop_client, lambda prompt: ported)

    later = tasks[0].later["2.0"]
    assert later.status == PORTED
    assert "sku: code" in later.query
    assert later.answer is None
    assert outcomes == {PORTED: 1}
    assert "Cannot query field 'sku' on type 'Variant'" in teacher.prompts[0]
    assert "  code: String" in teacher.prompts[0]
    assert "type Query {\n  product(id: ID, slug: String): Product" in teacher.prompts[0]
    assert teacher.options[0] == {"temperature": 0.0, "thinking": False}


def test_a_port_with_other_keys_keeps_its_own_answer(new_shop_schema, new_shop_client):
    tasks = [task(SKU_QUERY, SKU_ANSWER, SKU_COORDINATES, broken=["Variant.sku"])]
    ported = '{ product(slug: "alpine-tee") { variants { code } } }'

    settle(tasks, new_shop_schema, new_shop_client, lambda prompt: ported)

    later = tasks[0].later["2.0"]
    assert later.status == PORTED
    assert later.answer == {"product": {"variants": [{"code": "RS-0001-S"}]}}


def test_a_second_attempt_is_made(new_shop_schema, new_shop_client):
    tasks = [task(SKU_QUERY, SKU_ANSWER, SKU_COORDINATES, broken=["Variant.sku"])]
    replies = iter(
        [
            '{ product(slug: "alpine-tee") { id } }',
            '{ product(slug: "alpine-tee") { variants { sku: code } } }',
        ]
    )

    teacher, outcomes = settle(
        tasks, new_shop_schema, new_shop_client, lambda prompt: next(replies)
    )

    assert outcomes == {PORTED: 1}
    assert [options["temperature"] for options in teacher.options] == [0.0, 0.7]
    assert [options["thinking"] for options in teacher.options] == [False, True]


def test_a_match_on_a_small_answer_is_not_a_reference(new_shop_schema, new_shop_client):
    # The answer is one small number. A query for another field that returns the
    # same number matches it, and nothing in the values says whether it is right.
    query = '{ product(slug: "alpine-tee") { variants { stock } } }'
    answer = {"product": {"variants": [{"stock": 4}]}}
    tasks = [task(query, answer, ["Variant.stock"], broken=["Variant.stock"])]
    tasks[0].answer = {"product": {"variants": [{"units": 4}]}}

    _, outcomes = settle(tasks, new_shop_schema, new_shop_client, lambda prompt: query)

    later = tasks[0].later["2.0"]
    assert later.status == WEAK
    assert later.answer == answer
    assert outcomes == {WEAK: 1}


def test_informative():
    assert informative({"product": {"variants": [{"sku": "RS-0001-S"}]}})
    assert informative({"orders": [{"number": "101"}, {"number": "102"}, {"number": "103"}]})
    assert not informative({"shop": {"automaticFulfillment": True}})
    assert not informative({"products": {"totalCount": 60}})
    assert not informative({"a": True, "b": False, "c": None})
    # Repeating a value adds nothing.
    assert not informative({"lines": [{"quantity": 2}, {"quantity": 2}, {"quantity": 2}]})


def test_progress_is_saved_along_the_way(new_shop_schema, new_shop_client, monkeypatch):
    from restock.tasks import port as port_module

    monkeypatch.setattr(port_module, "CHECKPOINT_EVERY", 2)
    tasks = [task("{ shop { name } }", {"shop": {"name": "Sample shop"}}) for _ in range(5)]
    saved = []

    port(
        tasks,
        new_shop_schema,
        new_shop_client,
        ScriptedTeacher(lambda prompt: "GONE"),
        old="1.0",
        new="2.0",
        log=lambda message: None,
        checkpoint=lambda: saved.append(sum("2.0" in t.later for t in tasks)),
    )

    # as_completed checkpoints every N and once more at the end.
    assert saved == [2, 4, 5]


def test_removed_information_is_gone(new_shop_schema, new_shop_client):
    query = '{ product(slug: "alpine-tee") { rating } }'
    tasks = [task(query, {"product": {"rating": 4.5}}, ["Product.rating"], ["Product.rating"])]

    _, outcomes = settle(tasks, new_shop_schema, new_shop_client, lambda prompt: "GONE")

    assert tasks[0].later["2.0"] == Later(GONE)
    assert outcomes == {GONE: 1}


def test_a_different_answer_without_a_schema_change_is_unresolved(new_shop_schema):
    data = {**SHOP_DATA, "shop": {**SHOP_DATA["shop"], "name": "Renamed shop"}}
    client = store_client(new_shop_schema, data)
    tasks = [task("{ shop { name } }", {"shop": {"name": "Sample shop"}}, ["Shop.name"])]

    teacher, outcomes = settle(tasks, new_shop_schema, client, lambda prompt: "{ shop { name } }")

    assert tasks[0].later["2.0"] == Later(UNRESOLVED)
    assert "it returns a different answer." in teacher.prompts[0]
    assert len(teacher.prompts) == 2


def test_settled_tasks_are_left_alone(new_shop_schema, new_shop_client):
    tasks = [task("{ shop { name } }", {"shop": {"name": "Sample shop"}})]
    tasks[0].later["2.0"] = Later(GONE)

    _, outcomes = settle(tasks, new_shop_schema, new_shop_client, lambda prompt: "GONE")

    assert tasks[0].later["2.0"] == Later(GONE)
    assert outcomes == {}


@pytest.mark.parametrize(
    ("query", "answer", "problem"),
    [
        ("{ shop { name } }", {"shop": {"name": "Sample shop"}}, None),
        ("{ shop { name } }", {"shop": {"name": "Other"}}, "it returns a different answer."),
        (SKU_QUERY, SKU_ANSWER, "it is rejected. Cannot query field 'sku' on type 'Variant'."),
    ],
)
def test_original_problem(new_shop_schema, new_shop_client, query, answer, problem):
    found = original_problem(task(query, answer), new_shop_schema, new_shop_client)
    assert found == problem or (problem and found.startswith(problem))
