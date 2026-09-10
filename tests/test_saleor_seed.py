import json

import httpx
import pytest

from restock.saleor.client import SaleorClient
from restock.saleor.seed import (
    LINES,
    Attribute,
    Channel,
    Store,
    create_products,
    plan_products,
    read_store,
)

STORE = Store(
    categories={line.category: f"category:{line.category}" for line in LINES},
    product_types={line.product_type: f"type:{line.product_type}" for line in LINES},
    attributes={
        "material": Attribute("attr:material", ("cotton", "wool")),
        "size": Attribute("attr:size", ("s", "m", "l", "xl")),
        "shoe-size": Attribute("attr:shoe-size", ("40", "41", "42")),
        "flavor": Attribute("attr:flavor", ("apple", "orange")),
    },
    channels=(
        Channel("channel:usd", "default-channel", "USD"),
        Channel("channel:pln", "pln", "PLN"),
    ),
    warehouses=("warehouse:a", "warehouse:b", "warehouse:c", "warehouse:d"),
)


def test_plan_is_reproducible_for_a_seed():
    assert plan_products(STORE, 30, seed=7) == plan_products(STORE, 30, seed=7)
    assert plan_products(STORE, 30, seed=7) != plan_products(STORE, 30, seed=8)


def test_plan_has_unique_names_and_skus():
    plans = plan_products(STORE, 200)

    names = [plan.product["name"] for plan in plans]
    skus = [variant["sku"] for plan in plans for variant in plan.variants]
    assert len(set(names)) == 200
    assert len(set(skus)) == len(skus)


def test_plan_covers_every_product_line():
    plans = plan_products(STORE, len(LINES) * 3)

    categories = [plan.product["category"] for plan in plans]
    assert {categories.count(f"category:{line.category}") for line in LINES} == {3}


def test_products_get_the_attributes_their_type_needs():
    plans = plan_products(STORE, 90)

    for plan, line in zip(plans, LINES * 10, strict=True):
        attributes = plan.product.get("attributes", [])
        assert [a["id"] for a in attributes] == (
            [f"attr:{line.product_attribute}"] if line.product_attribute else []
        )
        for variant in plan.variants:
            assert [a["id"] for a in variant["attributes"]] == (
                [f"attr:{line.variant_attribute}"] if line.variant_attribute else []
            )
        if line.variant_attribute:
            options = [v["attributes"][0]["dropdown"]["value"] for v in plan.variants]
            assert 2 <= len(options) == len(set(options))
        else:
            assert len(plan.variants) == 1


def test_variants_are_priced_in_the_channels_the_product_is_listed_in():
    for plan in plan_products(STORE, 90):
        listed = [listing["channelId"] for listing in plan.listings]
        assert listed
        for variant in plan.variants:
            prices = {p["channelId"]: p for p in variant["channelListings"]}
            assert list(prices) == listed
            assert all(0 < p["costPrice"] < p["price"] for p in prices.values())
            if len(prices) == 2:
                assert prices["channel:pln"]["price"] > prices["channel:usd"]["price"]


def test_stock_includes_sold_out_and_low_quantities():
    quantities = [
        stock["quantity"]
        for plan in plan_products(STORE, 200)
        for variant in plan.variants
        for stock in variant["stocks"]
    ]

    assert min(quantities) == 0
    assert any(1 <= quantity <= 5 for quantity in quantities)
    assert max(quantities) > 100


def saleor_stub(calls: list[dict], failing: str | None = None) -> SaleorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        operation = next(
            name
            for name in ("productCreate", "productChannelListingUpdate", "productVariantBulkCreate")
            if name in body["query"]
        )
        errors = [{"field": "slug", "message": "taken"}] if operation == failing else []
        payload = {"errors": errors, "product": {"id": "product:1"}, "count": 1}
        return httpx.Response(200, json={"data": {operation: payload}})

    return SaleorClient("http://saleor.test/graphql/", transport=httpx.MockTransport(handler))


def test_create_products_creates_lists_and_adds_variants():
    calls: list[dict] = []
    plans = plan_products(STORE, 2)

    create_products(saleor_stub(calls), plans)

    assert len(calls) == 6
    assert calls[0]["variables"] == {"input": plans[0].product}
    assert calls[1]["variables"] == {
        "id": "product:1",
        "input": {"updateChannels": plans[0].listings},
    }
    assert calls[2]["variables"] == {"product": "product:1", "variants": plans[0].variants}
    assert calls[3]["variables"] == {"input": plans[1].product}


def test_create_products_stops_at_the_first_rejected_mutation():
    calls: list[dict] = []
    plans = plan_products(STORE, 2)

    with pytest.raises(RuntimeError, match=f"{plans[0].product['name']}: slug: taken"):
        create_products(saleor_stub(calls, failing="productChannelListingUpdate"), plans)

    assert len(calls) == 2


def test_read_store_collects_ids_by_name():
    data = {
        "categories": {"edges": [{"node": {"id": "c1", "name": "Juices"}}]},
        "productTypes": {"edges": [{"node": {"id": "t1", "name": "Juice"}}]},
        "attributes": {
            "edges": [
                {
                    "node": {
                        "id": "a1",
                        "slug": "flavor",
                        "choices": {"edges": [{"node": {"slug": "apple"}}]},
                    }
                }
            ]
        },
        "channels": [{"id": "ch1", "slug": "default-channel", "currencyCode": "USD"}],
        "warehouses": {"edges": [{"node": {"id": "w1"}}]},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": data}))

    store = read_store(SaleorClient("http://saleor.test/graphql/", transport=transport))

    assert store == Store(
        categories={"Juices": "c1"},
        product_types={"Juice": "t1"},
        attributes={"flavor": Attribute("a1", ("apple",))},
        channels=(Channel("ch1", "default-channel", "USD"),),
        warehouses=("w1",),
    )
