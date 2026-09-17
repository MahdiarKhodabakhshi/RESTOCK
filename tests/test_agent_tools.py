from graphql import build_schema

from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.reward import reward

SCHEMA = build_schema("""
type Query {
  product(id: ID!): Product
  products(first: Int, filter: ProductFilter): [Product!]!
}

type Product {
  id: ID!
  name: String!
  slug: String
  price: Money
}

type Money { amount: Float!, currency: String! }

input ProductFilter { search: String }
""")


def tools(**kwargs) -> SchemaTools:
    return SchemaTools(SCHEMA, version_signal("3.20.0", trained_on="3.20.0"), **kwargs)


def test_version_signal_does_not_charge_the_read_budget():
    session = tools()
    text = session.version_signal()
    assert "API version: 3.20.0" in text
    assert session.read_tokens == 0


def test_read_type_returns_sdl_and_accumulates_tokens():
    session = tools()
    first = session.read_type("Product")
    assert "type Product" in first.text
    assert "name: String!" in first.text
    assert first.tokens > 0
    assert session.read_tokens == first.tokens

    second = session.read_type("Money")
    assert session.read_tokens == first.tokens + second.tokens


def test_read_type_missing_name_still_charges_the_error_message():
    session = tools()
    result = session.read_type("Missing")
    assert result.text == "Type not found: Missing"
    assert result.tokens > 0
    assert session.read_tokens == result.tokens


def test_search_fields_matches_type_or_field_name():
    session = tools()
    result = session.search_fields("product")
    assert "Query.product: Product" in result.text
    assert "Query.products: [Product!]!" in result.text
    assert "Product.name: String!" in result.text
    assert result.tokens > 0
    assert session.read_tokens == result.tokens


def test_search_fields_rejects_an_empty_query():
    session = tools()
    result = session.search_fields("  ")
    assert result.text == "No matches."
    assert session.read_tokens == result.tokens


def test_search_fields_respects_the_limit():
    session = tools()
    result = session.search_fields("e", limit=2)
    assert len(result.text.splitlines()) == 2


def test_search_fields_lists_query_root_hits_first():
    schema = build_schema("""
    type Query { appExtensions: [AppExtension!]! }
    type AppExtension { id: ID! }
    type App { extensions: [AppExtension!]! }
    """)
    session = SchemaTools(schema, version_signal("3.23"))
    result = session.search_fields("app", limit=3)
    lines = result.text.splitlines()
    assert lines[0].startswith("Query.appExtensions")
    assert any(line.startswith("App.") or line.startswith("AppExtension.") for line in lines)


def test_custom_token_counter_is_used_for_accounting():
    session = tools(count_tokens=lambda text: 7)
    result = session.read_type("Money")
    assert result.tokens == 7
    assert session.read_tokens == 7


def test_read_tokens_plug_into_the_episode_reward():
    session = tools()
    session.read_type("Product")
    assert reward(True, session.read_tokens) < 1.0
    assert reward(True, session.read_tokens) > 0.0


def test_read_query_side_charges_every_reachable_type():
    session = tools()
    result = session.read_query_side()
    assert "type Product" in result.text
    assert "type Money" in result.text
    assert "input ProductFilter" in result.text
    assert session.read_tokens == result.tokens > 0
