import httpx
from graphql import build_schema

from restock.agent.efficiency import parse_efficiency, score_completion
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.saleor.client import SaleorClient

SCHEMA = build_schema("""
type Query { products(first: Int): [Product!]! }
type Product { id: ID!, name: String! }
""")


def test_parse_efficiency_reads_and_query_tags():
    text = """
    <read>Product</read>
    <read>Query</read>
    <read>Product</read>
    <query>{ products(first: 2) { name } }</query>
    """
    parsed = parse_efficiency(text)
    assert parsed.reads == ("Product", "Query")
    assert parsed.query == "{ products(first: 2) { name } }"


def test_parse_efficiency_falls_back_to_a_bare_query():
    parsed = parse_efficiency("{ shop { name } }")
    assert parsed.reads == ()
    assert parsed.query == "{ shop { name } }"


def test_score_completion_charges_reads_and_rewards_a_match():
    tools = SchemaTools(SCHEMA, version_signal("3.20"))
    answer = {"products": [{"name": "Mug"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": answer})

    client = SaleorClient("http://saleor.test/", transport=httpx.MockTransport(handler))
    text = "<read>Product</read><query>{ products(first: 1) { name } }</query>"
    value = score_completion(text, tools, client, answer)
    assert tools.read_tokens > 0
    assert 0.0 < value < 1.0
    client.close()


def test_score_completion_wrong_answer_is_zero():
    tools = SchemaTools(SCHEMA, version_signal("3.20"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"products": []}})

    client = SaleorClient("http://saleor.test/", transport=httpx.MockTransport(handler))
    value = score_completion(
        "<query>{ products(first: 1) { name } }</query>",
        tools,
        client,
        {"products": [{"name": "Mug"}]},
    )
    assert value == 0.0
    client.close()
