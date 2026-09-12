import json

import pytest
from conftest import ScriptedTeacher

from restock.schema.diff import diff_schemas
from restock.tasks.focus import Focus, SchemaGraph
from restock.tasks.generate import Generator, Settings, Target, find_targets
from restock.tasks.task import Task
from restock.tasks.teacher import PAIRS_FORMAT


@pytest.fixture(scope="module")
def changes(shop_schema, new_shop_schema):
    return diff_schemas(shop_schema, new_shop_schema)


def pairs(*items: tuple[str, str]) -> str:
    return json.dumps([{"request": question, "query": query} for question, query in items])


def generator(shop_schema, changes, shop_client, reply, **settings) -> Generator:
    settings = Settings(**{"requests": 1, "round_size": 1, "workers": 1, **settings})
    return Generator(
        shop_schema,
        changes,
        shop_client,
        ScriptedTeacher(reply),
        version="1.0",
        settings=settings,
        log=lambda message: None,
    )


def test_targets_are_the_changed_coordinates_a_query_can_use(shop_schema, changes):
    targets = find_targets(SchemaGraph(shop_schema), changes)

    assert targets == [
        Target("Product.rating", broken=True),
        Target("ProductOrderField.RATING", broken=True),
        Target("Shop.description", broken=False),
        Target("Variant.sku", broken=True),
    ]


def test_accepted_pairs_become_tasks(shop_schema, changes, shop_client):
    reply = pairs(
        ("What is the shop called?", "query { shop { name } }"),
        ("Rating of the product alpine-tee?", '{ product(slug: "alpine-tee") { rating } }'),
    )
    gen = generator(shop_schema, changes, shop_client, lambda prompt: reply)

    tasks = gen.run()

    assert [task.question for task in tasks] == [
        "What is the shop called?",
        "Rating of the product alpine-tee?",
    ]
    shop, rating = tasks
    assert shop.query == "{\n  shop {\n    name\n  }\n}"
    assert shop.answer == {"shop": {"name": "Sample shop"}}
    assert shop.coordinates == ["Query.shop", "Shop.name"]
    assert (shop.broken, shop.deprecated) == ([], [])
    assert rating.broken == ["Product.rating"]
    assert rating.answer == {"product": {"rating": 4.5}}
    assert gen.report.accepted == 2


def test_rejected_pairs_are_counted_and_kept_apart(shop_schema, changes, shop_client):
    reply = pairs(
        ("What is the shop called?", "{ shop { name } }"),
        ("Who owns the shop?", "{ shop { owner } }"),
        ("Any orders?", "{ orders(first: 5) { edges { node { number } } } }"),
    )
    gen = generator(shop_schema, changes, shop_client, lambda prompt: reply)

    tasks = gen.run()

    assert len(tasks) == 1
    assert gen.report.pairs == 3
    assert dict(gen.report.rejections) == {"invalid": 1, "empty": 1}
    assert [rejected.rejection for rejected in gen.rejected] == ["invalid", "empty"]
    assert "owner" in gen.rejected[0].detail


def test_the_same_query_or_request_is_kept_once(shop_schema, changes, shop_client):
    reply = pairs(
        ("What is the shop called?", "{ shop { name } }"),
        ("Tell me the shop's name.", "{shop{name}}"),
        ("what is the shop called", "{ shop { name description } }"),
    )
    gen = generator(shop_schema, changes, shop_client, lambda prompt: reply)

    assert len(gen.run()) == 1
    assert gen.report.duplicates == 2


def test_an_earlier_run_is_continued(shop_schema, changes, shop_client):
    earlier = Task(
        id="0" * 12,
        question="What is the shop called?",
        query="{\n  shop {\n    name\n  }\n}",
        answer={"shop": {"name": "Sample shop"}},
        focus=Focus("shop", ("Shop",)),
        coordinates=["Query.shop", "Shop.name"],
    )
    reply = pairs(
        ("What is the shop called?", "{ shop { name description } }"),
        ("Which currencies?", "{ currencies }"),
    )
    gen = generator(shop_schema, changes, shop_client, lambda prompt: reply)
    gen.add([earlier])

    tasks = gen.run()

    assert [task.question for task in tasks] == ["What is the shop called?", "Which currencies?"]
    assert tasks[0] is earlier


def test_prompts_ask_for_changed_coordinates_first(shop_schema, changes, shop_client):
    gen = generator(
        shop_schema, changes, shop_client, lambda prompt: "[]", requests=4, round_size=4
    )

    gen.run()

    teacher = gen.teacher
    featured = [
        coordinate
        for coordinate in ("Product.rating", "ProductOrderField.RATING", "Variant.sku")
        if any(f"must use each of these: `{coordinate}`" in prompt for prompt in teacher.prompts)
    ]
    # Four prompts at a target share of 0.4 leave room for one target.
    assert len(featured) == 1
    assert all(options == {"response_format": PAIRS_FORMAT} for options in teacher.options)
    assert gen.report.prompts == 4


def test_a_coordinate_with_enough_tasks_is_not_asked_for_again(shop_schema, changes, shop_client):
    reply = pairs(
        ("Rating of the product alpine-tee?", '{ product(slug: "alpine-tee") { rating } }')
    )
    gen = generator(
        shop_schema,
        changes,
        shop_client,
        lambda prompt: reply,
        requests=6,
        round_size=3,
        broken_quota=1,
        target_share=0.34,
    )

    gen.run()

    asked = [prompt for prompt in gen.teacher.prompts if "`Product.rating`" in prompt]
    assert len(asked) <= 1


def test_general_prompts_feature_fields_no_task_uses_yet(shop_schema, changes, shop_client):
    gen = generator(
        shop_schema, changes, shop_client, lambda prompt: "[]", requests=3, target_share=0.0
    )

    gen.run()

    assert all("At least one example must use each of these" in p for p in gen.teacher.prompts)
    assert not any("`ProductConnection.edges`" in prompt for prompt in gen.teacher.prompts)


def test_a_teacher_failure_costs_only_that_prompt(shop_schema, changes, shop_client):
    from restock.tasks.teacher import TeacherError

    calls = []

    def reply(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            raise TeacherError("server went away")
        return pairs(("What is the shop called?", "{ shop { name } }"))

    gen = generator(shop_schema, changes, shop_client, reply, requests=2)

    assert len(gen.run()) == 1
    assert gen.report.failed_prompts == 1


def test_on_round_sees_the_tasks_so_far(shop_schema, changes, shop_client):
    replies = iter(
        [
            pairs(("What is the shop called?", "{ shop { name } }")),
            pairs(("Which currencies?", "{ currencies }")),
        ]
    )
    gen = generator(shop_schema, changes, shop_client, lambda prompt: next(replies), requests=2)
    seen = []

    gen.run(on_round=lambda tasks: seen.append(len(tasks)))

    assert seen == [1, 2]
