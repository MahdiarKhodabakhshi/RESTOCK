import json

import httpx
from graphql import build_schema

from restock.agent.model import Student
from restock.agent.policies import AlwaysRead, MemorisedOnly, Selective
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools

SCHEMA = build_schema("""
type Query {
  product(id: ID!): Product
  order(id: ID!): Order
}

type Product { id: ID!, name: String! }
type Order { id: ID!, number: String! }
""")


def tools() -> SchemaTools:
    return SchemaTools(SCHEMA, version_signal("3.20"))


def test_always_read_searches_keywords_and_reads_matching_types():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["messages"][-1]["content"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{ product(id: "1") { name } }'}}]},
        )

    student = Student(["http://student.test/v1"], transport=httpx.MockTransport(handler))
    session = tools()
    query = AlwaysRead().answer(student, session, "Show the product name")
    assert query is not None
    assert session.read_tokens > 0
    assert "type Product" in seen[0]
    assert "type Order" not in seen[0]


def test_memorised_prompt_contains_no_schema_sdl():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][-1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "{ __typename }"}}]})

    student = Student(["http://student.test/v1"], transport=httpx.MockTransport(handler))
    MemorisedOnly().answer(student, tools(), "Show the product name")
    assert "type Product" not in seen[0]
    assert "Show the product name" in seen[0]


def test_selective_executes_search_then_stops_on_content():
    responses = iter(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "search_fields",
                            "arguments": json.dumps({"query": "product"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": '{ product(id: "1") { id } }'},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": next(responses)}]})

    student = Student(["http://student.test/v1"], transport=httpx.MockTransport(handler))
    session = tools()
    query = Selective().answer(student, session, "product id")
    assert query == '{ product(id: "1") { id } }'
    assert session.read_tokens > 0
