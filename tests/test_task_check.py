import httpx
import pytest
from conftest import SHOP_DATA, store_client

from restock.saleor.client import SaleorClient
from restock.tasks.check import (
    Rejection,
    check,
    has_content,
    normalise,
    question_key,
    stable,
)


def test_a_good_query_is_returned_in_normal_form_with_its_answer(shop_schema, shop_client):
    checked = check(shop_schema, shop_client, "query Named {shop{name}}")

    assert checked.ok
    assert checked.query == "{\n  shop {\n    name\n  }\n}"
    assert checked.data == {"shop": {"name": "Sample shop"}}
    assert checked.document is not None


@pytest.mark.parametrize(
    ("query", "rejection"),
    [
        ("{ shop { name }", Rejection.SYNTAX),
        ("mutation { shop { name } }", Rejection.NOT_ONE_QUERY),
        ("{ shop { ...F } } fragment F on Shop { name }", Rejection.NOT_ONE_QUERY),
        ("query ($n: Int) { products(first: $n) { totalCount } }", Rejection.VARIABLES),
        ("{ shop { owner } }", Rejection.INVALID),
        ("{ orders(first: 5) { totalCount edges { node { number } } } }", Rejection.EMPTY),
        ("{ shop { description } }", Rejection.EMPTY),
    ],
)
def test_rejections(shop_schema, shop_client, query, rejection):
    checked = check(shop_schema, shop_client, query)

    assert not checked.ok
    assert checked.rejection is rejection
    assert checked.data is None


def test_invalid_query_reports_the_validation_message(shop_schema, shop_client):
    checked = check(shop_schema, shop_client, "{ shop { owner } }")
    assert "Cannot query field 'owner' on type 'Shop'" in checked.detail


def test_an_empty_answer_is_accepted_when_content_is_not_required(shop_schema, shop_client):
    checked = check(shop_schema, shop_client, "{ shop { description } }", require_content=False)
    assert checked.ok
    assert checked.data == {"shop": {"description": None}}


def test_errors_from_the_server_reject_the_query(shop_schema):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": None, "errors": [{"message": "No permission"}]})

    client = SaleorClient("http://saleor.test/", transport=httpx.MockTransport(handler))
    checked = check(shop_schema, client, "{ shop { name } }")

    assert checked.rejection is Rejection.ERROR
    assert checked.detail == "No permission"


def test_an_answer_that_changes_between_runs_is_rejected(shop_schema):
    names = iter(["first", "second", "third"])
    data = {"shop": lambda *_: {"name": next(names)}}
    client = store_client(shop_schema, data)

    assert check(shop_schema, client, "{ shop { name } }").rejection is Rejection.UNSTABLE
    assert check(shop_schema, client, "{ shop { name } }", repeat=False).ok


def test_an_oversized_answer_is_rejected(shop_schema):
    data = {**SHOP_DATA, "shop": {"name": "x" * 30_000}}
    checked = check(shop_schema, store_client(shop_schema, data), "{ shop { name } }")
    assert checked.rejection is Rejection.TOO_LARGE


def test_local_urls_do_not_depend_on_the_port(shop_schema, shop_client):
    checked = check(
        shop_schema, shop_client, '{ product(slug: "alpine-tee") { thumbnail(size: 64) } }'
    )
    assert checked.data == {"product": {"thumbnail": "http://saleor.local/thumbnail/P1/64/"}}


def test_stable_leaves_other_urls_alone():
    value = {"a": ["http://localhost:8023/media/x.png", "https://example.com:8020/x"], "b": 8020}
    assert stable(value) == {
        "a": ["http://saleor.local/media/x.png", "https://example.com:8020/x"],
        "b": 8020,
    }


def test_normalise_ignores_layout_and_operation_name():
    assert normalise("query A { shop { name } }") == normalise("{shop{name}}")
    assert normalise("{ shop { name } }") != normalise("{ shop { description } }")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"shop": {"name": "x"}}, True),
        ({"shop": {"active": False}}, True),
        ({"stock": {"quantity": 0}}, True),
        ({"shop": None}, False),
        ({"shop": {"name": None, "description": ""}}, False),
        ({"orders": {"edges": [], "totalCount": 0}}, False),
        ({"orders": {"edges": [], "pageInfo": {"hasNextPage": False}}}, False),
        ({"orders": {"totalCount": 12}}, True),
        ({"orders": {"edges": [{"node": {"note": None}}]}}, False),
        ({"shop": {"name": "x"}, "orders": {"edges": []}}, False),
        ({}, False),
        (None, False),
    ],
)
def test_has_content(data, expected):
    assert has_content(data) is expected


def test_question_key_ignores_case_punctuation_and_spacing():
    assert question_key("Show me  the Shop's name!") == question_key("show me the shop s name")
    assert question_key("Orders of 2025") != question_key("Orders of 2026")
