import json

import httpx
from graphql import build_schema

from restock.agent.episode import run_episode
from restock.agent.model import Student
from restock.agent.policies import AlwaysRead, MemorisedOnly, Selective
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.saleor.client import SaleorClient

SCHEMA = build_schema("""
type Query {
  product(id: ID!): Product
  products(first: Int): [Product!]!
}

type Product {
  id: ID!
  name: String!
  slug: String
}
""")

QUERY = "{ products(first: 2) { id name } }"
ANSWER = {"products": [{"id": "1", "name": "Mug"}, {"id": "2", "name": "Cap"}]}


def student_returning(text: str, *, tool_calls: list | None = None) -> Student:
    def handler(request: httpx.Request) -> httpx.Response:
        message = {"role": "assistant", "content": text}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
            message["content"] = ""
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return Student(["http://student.test/v1"], transport=httpx.MockTransport(handler))


def saleor_returning(data) -> SaleorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": data})

    return SaleorClient("http://saleor.test/", transport=httpx.MockTransport(handler))


def tools() -> SchemaTools:
    return SchemaTools(SCHEMA, version_signal("3.20.0", trained_on="3.20.0"))


def test_memorised_episode_scores_a_correct_query_with_no_reads():
    student = student_returning(f"```graphql\n{QUERY}\n```")
    result = run_episode(
        MemorisedOnly(), student, tools(), saleor_returning(ANSWER), "list products", ANSWER
    )
    assert result.correct
    assert result.read_tokens == 0
    assert result.reward == 1.0
    assert result.query == QUERY


def test_always_read_charges_schema_lookups_before_answering():
    student = student_returning(f"```graphql\n{QUERY}\n```")
    session = tools()
    result = run_episode(
        AlwaysRead(),
        student,
        session,
        saleor_returning(ANSWER),
        "list product names",
        ANSWER,
    )
    assert result.correct
    assert result.read_tokens > 0
    assert result.reward < 1.0


def test_selective_runs_tool_calls_then_returns_a_query():
    calls = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "read_type", "arguments": json.dumps({"name": "Product"})},
        }
    ]
    responses = iter(
        [
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "assistant", "content": f"```graphql\n{QUERY}\n```"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": next(responses)}]})

    student = Student(["http://student.test/v1"], transport=httpx.MockTransport(handler))
    session = tools()
    result = run_episode(
        Selective(), student, session, saleor_returning(ANSWER), "list products", ANSWER
    )
    assert result.correct
    assert result.read_tokens > 0


def test_wrong_execution_result_scores_zero():
    student = student_returning(QUERY)
    result = run_episode(
        MemorisedOnly(),
        student,
        tools(),
        saleor_returning({"products": []}),
        "list products",
        ANSWER,
    )
    assert not result.correct
    assert result.reward == 0.0


def test_missing_query_scores_zero():
    student = student_returning("I do not know")
    result = run_episode(
        MemorisedOnly(), student, tools(), saleor_returning(ANSWER), "list products", ANSWER
    )
    assert result.query is None
    assert not result.correct
