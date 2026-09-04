"""Download the GraphQL schema that Saleor publishes with each release."""

from pathlib import Path

import httpx

SCHEMA_URL = (
    "https://raw.githubusercontent.com/saleor/saleor/{version}/saleor/graphql/schema.graphql"
)
DEFAULT_ROOT = Path("data/schemas/saleor")


def schema_path(version: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / f"{version}.graphql"


def fetch_schema(
    version: str,
    root: Path = DEFAULT_ROOT,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
) -> Path:
    """Save the schema for a release tag under ``root`` and return its path.

    A schema that is already on disk is reused unless ``force`` is set.
    """
    path = schema_path(version, root)
    if path.exists() and not force:
        return path

    owns_client = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        response = client.get(SCHEMA_URL.format(version=version))
        response.raise_for_status()
    finally:
        if owns_client:
            client.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return path
