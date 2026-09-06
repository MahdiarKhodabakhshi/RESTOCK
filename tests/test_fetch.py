import httpx
import pytest

from restock.schema.fetch import fetch_schema, load_schema, schema_path

SDL = "type Query { ping: String }\n"


def make_client(calls: list[str], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, text=SDL)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_writes_schema_for_the_release_tag(tmp_path):
    calls: list[str] = []

    path = fetch_schema("3.23.0", tmp_path, client=make_client(calls))

    assert path == schema_path("3.23.0", tmp_path)
    assert path.read_text() == SDL
    assert calls == [
        "https://raw.githubusercontent.com/saleor/saleor/3.23.0/saleor/graphql/schema.graphql"
    ]


def test_fetch_reuses_schema_already_on_disk(tmp_path):
    calls: list[str] = []
    client = make_client(calls)

    fetch_schema("3.23.0", tmp_path, client=client)
    fetch_schema("3.23.0", tmp_path, client=client)

    assert len(calls) == 1


def test_fetch_force_downloads_again(tmp_path):
    calls: list[str] = []
    client = make_client(calls)

    fetch_schema("3.23.0", tmp_path, client=client)
    fetch_schema("3.23.0", tmp_path, client=client, force=True)

    assert len(calls) == 2


def test_fetch_raises_for_unknown_tag(tmp_path):
    with pytest.raises(httpx.HTTPStatusError):
        fetch_schema("0.0.0", tmp_path, client=make_client([], status=404))

    assert not schema_path("0.0.0", tmp_path).exists()


def test_load_schema_requires_a_fetched_version(tmp_path):
    with pytest.raises(FileNotFoundError, match="restock fetch 3.23.0"):
        load_schema("3.23.0", tmp_path)
