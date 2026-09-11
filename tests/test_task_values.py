import json
from random import Random

from conftest import SHOP_DATA, store_client
from graphql import GraphQLError

from restock.tasks.values import (
    Store,
    compact,
    format_facts,
    format_records,
    print_selection,
    record_selection,
)


def test_record_selection_reads_values_and_what_identifies_related_records(shop_schema):
    selection = record_selection(shop_schema.type_map["Product"])

    assert selection == {
        "id": None,
        "name": None,
        "slug": None,
        "rating": None,
        "category": {"id": None, "name": None, "slug": None},
        "price": {"amount": None, "currency": None},
    }


def test_print_selection():
    selection = {"id": None, "price": {"amount": None}, "name": None}
    assert print_selection(selection) == "{ id price { amount } name }"


def test_records_come_from_a_connection(shop_schema, shop_client):
    records = Store(shop_schema, shop_client).records("products", Random(0), count=5)

    assert sorted(record["slug"] for record in records) == ["alpine-tee", "harbour-mug"]
    assert records[0].keys() == {"id", "name", "slug", "rating", "category", "price"}


def test_records_are_sampled_and_read_once(shop_schema):
    requests: list[str] = []
    store = Store(shop_schema, store_client(shop_schema, SHOP_DATA, requests))

    assert len(store.records("products", Random(0), count=1)) == 1
    assert len(store.records("products", Random(1), count=1)) == 1
    assert len(requests) == 1
    assert requests[0].startswith("{ products(first: 100) { edges { node { id name")


def test_a_lookup_field_takes_its_records_from_the_list_of_the_same_type(shop_schema):
    requests: list[str] = []
    store = Store(shop_schema, store_client(shop_schema, SHOP_DATA, requests))

    records = store.records("product", Random(0), count=5)

    assert sorted(record["id"] for record in records) == ["P1", "P2"]
    assert all(request.startswith("{ products(") for request in requests)


def test_plain_lists_and_single_objects(shop_schema, shop_client):
    store = Store(shop_schema, shop_client)

    assert store.records("channels", Random(0)) == [
        {"id": "CH1", "slug": "default-channel", "currency": "USD"}
    ]
    assert store.records("shop", Random(0)) == [
        {"name": "Sample shop", "description": None, "defaultChannel": None}
    ]


def test_fields_without_records(shop_schema, shop_client):
    store = Store(shop_schema, shop_client)

    assert store.records("orders", Random(0)) == []
    assert store.records("report", Random(0)) == []
    assert store.records("currencies", Random(0)) == []


def test_a_field_that_fails_is_left_out_and_the_read_repeated(shop_schema):
    def forbidden(*_):
        raise GraphQLError("No permission")

    products = SHOP_DATA["products"]
    guarded = [{**edge["node"], "rating": forbidden} for edge in products["edges"]]
    data = {
        **SHOP_DATA,
        "products": {**products, "edges": [{"node": node} for node in guarded]},
    }
    requests: list[str] = []
    store = Store(shop_schema, store_client(shop_schema, data, requests))

    records = store.records("products", Random(0), count=5)

    assert len(records) == 2
    assert all("rating" not in record for record in records)
    assert " rating " in requests[0]
    assert " rating " not in requests[-1]


def test_compact_drops_empty_values_and_cuts_long_ones():
    record = {
        "name": "Tee",
        "note": "",
        "parent": None,
        "tags": [],
        "price": {"amount": 0, "currency": None},
        "text": "x" * 200,
        "codes": list(range(40)),
    }
    assert compact(record) == {
        "name": "Tee",
        "price": {"amount": 0},
        "text": "x" * 120 + "...",
        "codes": list(range(12)),
    }


def test_format_records_is_one_json_object_per_line():
    text = format_records([{"id": "1", "name": None}, {"id": "2", "name": "Zoë"}])
    assert text == '{"id": "1"}\n{"id": "2", "name": "Zoë"}'
    assert [json.loads(line) for line in text.splitlines()][1]["name"] == "Zoë"


def test_format_facts():
    def nodes(*items):
        return {"edges": [{"node": item} for item in items]}

    data = {
        "channels": [{"slug": "default-channel", "name": "Default", "currencyCode": "USD"}],
        "categories": nodes({"id": "C1", "name": "Shirts", "slug": "shirts"}),
        "collections": nodes(),
        "productTypes": nodes(),
        "warehouses": nodes(),
        "attributes": nodes(),
        "first": nodes({"created": "2025-03-11T08:00:00+00:00"}),
        "last": {**nodes({"created": "2026-08-29T10:00:00+00:00"}), "totalCount": 420},
    }

    lines = format_facts(data).splitlines()

    assert lines[0] == 'Channels: "default-channel" (Default, USD)'
    assert lines[1] == 'Categories: Shirts (slug "shirts", id "C1")'
    assert lines[-1] == "Orders: 420, placed between 2025-03-11 and 2026-08-29"


def test_facts_are_empty_when_the_store_cannot_be_read(shop_schema, shop_client):
    # The sample schema has none of the Saleor fields the facts are read from.
    assert Store(shop_schema, shop_client).facts() == ""
