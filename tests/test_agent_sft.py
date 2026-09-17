from pathlib import Path

from graphql import build_schema

from restock.agent.sft import always_read_messages, sft_messages, write_sft
from restock.tasks.focus import Focus
from restock.tasks.task import Task

SCHEMA = build_schema("""
type Query { products(first: Int): [Product!]! }
type Product { id: ID!, name: String! }
""")


def _task() -> Task:
    return Task(
        id="t1",
        question="List product names",
        query="{ products(first: 5) { name } }",
        answer={},
        focus=Focus("products", ("Product",), ()),
        coordinates=["Query.products"],
    )


def test_sft_messages_hide_the_schema_and_show_the_query():
    messages = sft_messages(_task(), version="3.20")
    assert messages[0]["role"] == "system"
    assert "type Product" not in messages[1]["content"]
    assert "List product names" in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": _task().query}


def test_always_read_messages_include_a_schema_slice():
    messages = always_read_messages(_task(), SCHEMA, version="3.20")
    assert "type Product" in messages[1]["content"]
    assert "List product names" in messages[1]["content"]
    assert messages[2]["content"].startswith("{ products")


def test_write_sft_writes_one_json_object_per_task(tmp_path: Path):
    path = tmp_path / "sft.jsonl"
    assert write_sft(path, [_task()], version="3.20") == 1
    row = path.read_text().strip()
    assert '"id": "t1"' in row
    assert "messages" in row
