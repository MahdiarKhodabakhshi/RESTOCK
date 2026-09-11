from random import Random

import pytest

from restock.tasks.focus import Focus, SchemaGraph, fields_of, render, sample_focus


@pytest.fixture(scope="module")
def graph(shop_schema):
    return SchemaGraph(shop_schema)


def test_roots_leave_out_federation_fields(graph):
    assert "products" in graph.roots
    assert "_service" not in graph.roots


def test_route_goes_through_the_connection(graph):
    assert graph.route("products", "Category") == [
        "ProductConnection",
        "ProductEdge",
        "Product",
        "Category",
    ]
    assert graph.route("shop", "Order") is None


def test_going_through_a_connection_is_free(graph):
    assert graph.distance("product", "Product") == 0
    assert graph.distance("products", "Product") == 0
    assert graph.distance("products", "Warehouse") == 2
    assert graph.distance("channels", "Product") is None


def test_roots_reaching_prefers_the_nearest(graph):
    assert graph.roots_reaching("Product") == ["products", "product"]
    assert graph.roots_reaching("Variant", slack=0) == ["products", "product"]
    assert graph.roots_reaching("Variant", slack=1) == ["products", "product", "orders"]
    assert graph.roots_reaching("Period") == []


def test_sites_of_fields_and_arguments(graph):
    assert graph.sites("Product.rating") == [("Product", "rating")]
    assert graph.sites("Product.thumbnail(size:)") == [("Product", "thumbnail")]
    assert graph.sites("Product.missing") == []
    assert graph.sites("Missing.field") == []


def test_sites_of_inputs_are_the_fields_taking_them(graph):
    assert graph.sites("ProductOrderField.RATING") == [("Query", "products")]
    assert graph.sites("PriceRange.gte") == [("Query", "products")]
    assert graph.sites("OrderStatus.DRAFT") == []


def test_focus_starts_with_the_chain_to_the_record_type(graph):
    focus = sample_focus(graph, Random(0), "products")
    assert focus.root == "products"
    assert focus.types[:3] == ("ProductConnection", "ProductEdge", "Product")
    assert len(set(focus.types)) == len(focus.types)


def test_focus_without_expansions_is_only_the_chain(graph):
    focus = sample_focus(graph, Random(0), "orders", expansions=0)
    assert focus.types == ("OrderConnection", "OrderEdge", "Order")


def test_focus_through_a_type_shows_the_way_to_it(graph):
    focus = sample_focus(
        graph, Random(0), "products", through="Warehouse", feature=("Warehouse.name",)
    )
    assert focus.types[:5] == (
        "ProductConnection",
        "ProductEdge",
        "Product",
        "Variant",
        "Warehouse",
    )
    assert focus.feature == ("Warehouse.name",)


def test_focus_through_an_unreachable_type_is_an_error(graph):
    with pytest.raises(ValueError, match="not reachable"):
        sample_focus(graph, Random(0), "shop", through="Order")


def test_focus_on_a_field_without_selection_has_no_types(graph):
    assert sample_focus(graph, Random(0), "currencies") == Focus("currencies", ())


def test_focus_is_reproducible(graph):
    first = [sample_focus(graph, Random(3), root) for root in graph.roots]
    second = [sample_focus(graph, Random(3), root) for root in graph.roots]
    assert first == second


def test_render_shows_the_root_field_and_the_focus_types(shop_schema):
    text = render(shop_schema, Focus("products", ("ProductConnection", "ProductEdge", "Product")))
    assert text.startswith("type Query {\n  products(first: Int, filter: ProductFilter")
    assert "# List of the shop's products." in text
    assert "orders(" not in text
    assert "type Product implements Node {" in text
    assert "  name: String!  # Name of the product." in text
    assert "Shown in the storefront" not in text
    assert "  rating: Float @deprecated" in text
    assert "  thumbnail(size: Int!): String" in text


def test_render_adds_inputs_enums_and_small_types(shop_schema):
    text = render(shop_schema, Focus("products", ("ProductConnection", "ProductEdge", "Product")))
    assert "input ProductFilter {" in text
    assert "input PriceRange {" in text
    assert "enum ProductOrderField { NAME PRICE RATING }" in text
    assert "type Money {" in text
    assert "type PageInfo {" in text
    # Warehouse has too many fields to count as a value type and was not asked for.
    assert "type Warehouse" not in text
    assert "type Order " not in text


def test_render_names_the_implementations_of_an_interface(shop_schema):
    text = render(shop_schema, Focus("shop", ("Node",)))
    assert "interface Node {" in text
    assert "# Node is implemented by: Product, Category, Variant" in text


def test_render_keeps_main_types_and_drops_support_over_budget(shop_schema):
    focus = Focus("products", ("ProductConnection", "ProductEdge", "Product"))
    text = render(shop_schema, focus, budget=0)
    assert "type Product implements Node {" in text
    assert "input ProductFilter" not in text


def test_render_puts_featured_types_before_the_budget_runs_out(shop_schema):
    focus = Focus("products", ("ProductConnection",), feature=("Direction.DESC",))
    full = render(shop_schema, focus)
    budget = full.index("enum Direction") + len("enum Direction { ASC DESC }")
    text = render(shop_schema, focus, budget=budget)
    assert "enum Direction { ASC DESC }" in text
    assert "input ProductFilter" not in text


def test_long_enums_are_cut(shop_schema):
    from graphql import build_schema

    values = " ".join(f"V{i}" for i in range(60))
    schema = build_schema(f"type Query {{ pick(value: Big): Int }} enum Big {{ {values} }}")
    text = render(schema, Focus("pick", ()))
    assert "V14 ... 45 more }" in text
    assert "V15" not in text


def test_fields_of_lists_the_root_and_the_shown_types(shop_schema):
    focus = Focus("orders", ("OrderConnection", "OrderEdge", "OrderLine"))
    assert fields_of(shop_schema, focus) == [
        "Query.orders",
        "OrderConnection.edges",
        "OrderConnection.totalCount",
        "OrderConnection.pageInfo",
        "OrderEdge.node",
        "OrderEdge.cursor",
        "OrderLine.quantity",
        "OrderLine.variant",
    ]
