import json

import httpx
import pytest

from restock.tasks.teacher import (
    PAIRS_FORMAT,
    Pair,
    Teacher,
    TeacherError,
    parse_pairs,
    parse_query,
    port_prompt,
    solve_prompt,
    write_prompt,
)


def teacher_answering(content: str, requests: list[httpx.Request], **options) -> Teacher:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    urls = options.pop("urls", ["http://gpu0/v1/"])
    return Teacher(urls, transport=httpx.MockTransport(handler), **options)


def test_complete_posts_a_chat_request():
    requests: list[httpx.Request] = []
    teacher = teacher_answering("an answer", requests, temperature=0.5, max_tokens=100)

    assert teacher.complete("a prompt") == "an answer"

    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "http://gpu0/v1/chat/completions"
    assert body["model"] == "teacher"
    assert body["messages"][-1] == {"role": "user", "content": "a prompt"}
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 100
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in body


def test_complete_passes_overrides():
    requests: list[httpx.Request] = []
    teacher = teacher_answering("[]", requests, max_tokens=100)

    teacher.complete("p", temperature=0.0, response_format=PAIRS_FORMAT, thinking=True)

    body = json.loads(requests[0].content)
    assert body["temperature"] == 0.0
    assert body["response_format"] == PAIRS_FORMAT
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 300


def test_complete_drops_inline_reasoning():
    teacher = teacher_answering("<think>\nfirst this, then that\n</think>\n\n{ shop { name } }", [])
    assert teacher.complete("p").strip() == "{ shop { name } }"


def test_requests_rotate_over_the_servers():
    requests: list[httpx.Request] = []
    teacher = teacher_answering("x", requests, urls=["http://gpu0/v1", "http://gpu1/v1"])

    for _ in range(3):
        teacher.complete("p")

    assert [request.url.host for request in requests] == ["gpu0", "gpu1", "gpu0"]


def test_a_server_error_raises():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="out of memory"))
    teacher = Teacher(["http://gpu0/v1"], transport=transport)

    with pytest.raises(TeacherError, match="gpu0"):
        teacher.complete("p")


def test_a_teacher_needs_a_server():
    with pytest.raises(ValueError):
        Teacher([])


def test_parse_pairs():
    text = json.dumps(
        [
            {"request": "  Name of\nthe shop? ", "query": " { shop { name } } "},
            {"request": "No query"},
            {"request": "", "query": "{ shop { name } }"},
            "not an object",
        ]
    )
    assert parse_pairs(f"Here you go:\n{text}\nDone.") == [
        Pair("Name of the shop?", "{ shop { name } }")
    ]


@pytest.mark.parametrize("text", ["", "no json here", "[{broken", '{"request": "x"}'])
def test_parse_pairs_returns_nothing_for_malformed_output(text):
    assert parse_pairs(text) == []


def test_parse_query():
    assert parse_query("{ shop { name } }") == "{ shop { name } }"
    assert parse_query("```graphql\n{ shop { name } }\n```") == "{ shop { name } }"
    assert parse_query("Sure:\n```\n{ shop { name } }\n```\nAnything else?") == "{ shop { name } }"


@pytest.mark.parametrize("text", ["GONE", " gone. ", "", "```graphql\nGONE\n```"])
def test_parse_query_understands_a_refusal(text):
    assert parse_query(text) is None


def test_write_prompt_names_the_root_and_the_featured_coordinates():
    prompt = write_prompt(
        version="3.20",
        root="products",
        schema="type Query { products: [Product] }",
        facts="Channels: default-channel",
        records='{"id": "P1"}',
        count=4,
        style="short commands",
        feature=("Product.rating", "Product.name"),
    )
    assert "Saleor 3.20 schema" in prompt
    assert "type Query { products: [Product] }" in prompt
    assert "starts at `Query.products`" in prompt
    assert "Write 4 examples" in prompt
    assert '{"id": "P1"}' in prompt
    assert (
        "At least one example must use each of these: `Product.rating`, `Product.name`." in prompt
    )


def test_write_prompt_without_records_or_feature():
    prompt = write_prompt(
        version="3.20", root="shop", schema="S", facts="", records="", count=2, style="x"
    )
    assert "At least one example" not in prompt
    assert "(none available" in prompt


def test_solve_prompt_holds_the_request_and_not_a_query():
    prompt = solve_prompt(
        version="3.20", root="shop", schema="SCHEMA", facts="FACTS", question="Name of the shop?"
    )
    assert "Request: Name of the shop?" in prompt
    assert "SCHEMA" in prompt and "FACTS" in prompt
    assert "starts at `Query.shop`" in prompt


def test_port_prompt_explains_what_went_wrong():
    prompt = port_prompt(
        old="3.20",
        new="3.23",
        question="SKU of variant V1?",
        query="{ variant { sku } }",
        problem="it is rejected. Cannot query field 'sku'.",
        schema="type Variant { code: String }",
    )
    assert "On Saleor 3.20 this request" in prompt
    assert "{ variant { sku } }" in prompt
    assert "does not give that answer any more: it is rejected. Cannot query field 'sku'." in prompt
    assert "type Variant { code: String }" in prompt
    assert "GONE" in prompt
