"""Execute GraphQL operations against a Saleor API."""

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class Result:
    data: dict[str, Any] | None
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.errors

    @property
    def messages(self) -> list[str]:
        return [error.get("message", "") for error in self.errors]


class SaleorError(RuntimeError):
    """The server did not answer with a GraphQL response."""


class SaleorClient:
    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(headers=headers, timeout=timeout, transport=transport)
        self.url = url

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> Result:
        """Run one operation.

        Errors reported by GraphQL, such as an unknown field or a missing
        permission, come back in the result. Only a response that is not
        GraphQL at all raises.
        """
        response = self._http.post(self.url, json={"query": query, "variables": variables or {}})
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not body.keys() & {"data", "errors"}:
            raise SaleorError(f"HTTP {response.status_code}: {response.text[:200]}")
        return Result(body.get("data"), body.get("errors") or [])

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SaleorClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
