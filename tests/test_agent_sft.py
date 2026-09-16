from pathlib import Path

from restock.agent.sft import sft_messages, write_sft
from restock.tasks.focus import Focus
from restock.tasks.task import Task


def test_sft_messages_hide_the_schema_and_show_the_query():
    task = Task(
        id="t1",
        question="List product names",
        query="{ products(first: 5) { name } }",
        answer={},
        focus=Focus("products", ("Product",), ()),
        coordinates=["Query.products"],
    )
    messages = sft_messages(task, version="3.20")
    assert messages[0]["role"] == "system"
    assert "type Product" not in messages[1]["content"]
    assert "List product names" in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": task.query}


def test_write_sft_writes_one_json_object_per_task(tmp_path: Path):
    task = Task(
        id="t1",
        question="q",
        query="{ shop { name } }",
        answer={},
        focus=Focus("shop", ("Shop",), ()),
        coordinates=["Query.shop"],
    )
    path = tmp_path / "sft.jsonl"
    assert write_sft(path, [task], version="3.20") == 1
    row = path.read_text().strip()
    assert '"id": "t1"' in row
    assert "messages" in row
