import json

import httpx
import pytest

from restock.saleor.client import SaleorClient, SaleorError

URL = "http://saleor.test/graphql/"


def client_returning(response: httpx.Response, requests: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return response

    return SaleorClient(URL, "secret", transport=httpx.MockTransport(handler))


def test_execute_sends_the_operation_with_the_token():
    requests: list[httpx.Request] = []
    client = client_returning(httpx.Response(200, json={"data": {"shop": {"name": "x"}}}), requests)

    result = client.execute("query ($n: Int) { shop { name } }", {"n": 1})

    assert result.ok
    assert result.data == {"shop": {"name": "x"}}
    assert str(requests[0].url) == URL
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content) == {
        "query": "query ($n: Int) { shop { name } }",
        "variables": {"n": 1},
    }


def test_anonymous_client_sends_no_token():
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json={"data": {}})
    )

    SaleorClient(URL, transport=transport).execute("{ shop { name } }")

    assert "authorization" not in requests[0].headers


def test_graphql_errors_are_returned_not_raised():
    errors = [{"message": 'Cannot query field "nope" on type "Query".'}]
    client = client_returning(httpx.Response(400, json={"errors": errors}))

    result = client.execute("{ nope }")

    assert not result.ok
    assert result.data is None
    assert result.messages == ['Cannot query field "nope" on type "Query".']


def test_partial_data_with_errors_is_not_ok():
    body = {"data": {"shop": None}, "errors": [{"message": "PermissionDenied"}]}

    result = client_returning(httpx.Response(200, json=body)).execute("{ shop { version } }")

    assert not result.ok
    assert result.data == {"shop": None}


@pytest.mark.parametrize(
    "response",
    [httpx.Response(502, text="<html>Bad Gateway</html>"), httpx.Response(200, json=["nope"])],
)
def test_response_that_is_not_graphql_raises(response):
    with pytest.raises(SaleorError, match=str(response.status_code)):
        client_returning(response).execute("{ shop { name } }")
