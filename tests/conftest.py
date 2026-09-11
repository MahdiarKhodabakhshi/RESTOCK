import json

import httpx
import pytest
from graphql import GraphQLSchema, build_schema, graphql_sync

from restock.saleor.client import SaleorClient

SHOP_SDL = """
type Query {
  "List of the shop's products."
  products(first: Int, filter: ProductFilter, sortBy: ProductOrder): ProductConnection
  "Look up a product by ID or slug."
  product(id: ID, slug: String): Product
  orders(first: Int): OrderConnection
  shop: Shop!
  channels: [Channel!]
  currencies: [String!]!
  "Sales of the given period."
  report(period: Period!): Money
  _service: Shop
}

type ProductConnection { edges: [ProductEdge!]!, totalCount: Int, pageInfo: PageInfo! }
type ProductEdge { node: Product!, cursor: String! }
type OrderConnection { edges: [OrderEdge!]!, totalCount: Int, pageInfo: PageInfo! }
type OrderEdge { node: Order!, cursor: String! }
type PageInfo { hasNextPage: Boolean!, endCursor: String }

interface Node { id: ID! }

type Product implements Node {
  id: ID!
  "Name of the product. Shown in the storefront and in the dashboard."
  name: String!
  slug: String!
  rating: Float @deprecated(reason: "Use reviews.")
  category: Category
  price: Money
  variants: [Variant!]
  thumbnail(size: Int!): String
  privateMetadata: [String!]
}

type Category implements Node {
  id: ID!
  name: String!
  slug: String!
  parent: Category
  products(first: Int): ProductConnection
}

type Variant implements Node {
  id: ID!
  sku: String
  stock: Int!
  warehouse: Warehouse
}

type Warehouse implements Node {
  id: ID!
  name: String!
  email: String
  address: Address
  city: String
  country: String
  isPrivate: Boolean
  shippingZones: [String!]
  capacity: Int
}

type Address { city: String, country: String }

type Order implements Node {
  id: ID!
  number: String!
  status: OrderStatus!
  total: Money!
  lines: [OrderLine!]!
  customer: Customer
}

type OrderLine { quantity: Int!, variant: Variant }
type Customer implements Node { id: ID!, email: String!, orders(first: Int): OrderConnection }
type Money { amount: Float!, currency: String! }
type Shop { name: String!, description: String, defaultChannel: Channel }
type Channel implements Node { id: ID!, slug: String!, currency: String! }

input ProductFilter { search: String, price: PriceRange, categories: [ID!] }
input PriceRange { gte: Float, lte: Float }
input ProductOrder { field: ProductOrderField!, direction: Direction! }
enum ProductOrderField { NAME PRICE RATING }
enum Direction { ASC DESC }
enum OrderStatus { DRAFT UNFULFILLED FULFILLED CANCELED }
enum Period { TODAY THIS_MONTH }
"""


@pytest.fixture(scope="session")
def shop_schema() -> GraphQLSchema:
    """A small store API with the shapes Saleor uses: connections, inputs, enums."""
    return build_schema(SHOP_SDL)


# A later version: the rating is gone, the SKU has a new name, a field is deprecated.
NEW_SHOP_SDL = (
    SHOP_SDL.replace('  rating: Float @deprecated(reason: "Use reviews.")\n', "")
    .replace("  sku: String\n", "  code: String\n")
    .replace("description: String,", 'description: String @deprecated(reason: "Unused."),')
    .replace(
        "enum ProductOrderField { NAME PRICE RATING }", "enum ProductOrderField { NAME PRICE }"
    )
)


def _connection(nodes: list[dict]) -> dict:
    return {
        "edges": [{"node": node, "cursor": f"c{index}"} for index, node in enumerate(nodes)],
        "totalCount": len(nodes),
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


AMERICAS = {
    "id": "W1",
    "name": "Americas",
    "email": None,
    "address": {"city": "Austin", "country": "US"},
    "city": "Austin",
    "country": "US",
    "isPrivate": False,
    "shippingZones": [],
    "capacity": 900,
}
SHIRTS = {"id": "C1", "name": "Shirts", "slug": "shirts", "parent": None}
TEE = {
    "id": "P1",
    "name": "Alpine Tee",
    "slug": "alpine-tee",
    "rating": 4.5,
    "category": SHIRTS,
    "price": {"amount": 19.99, "currency": "USD"},
    "variants": [
        {"id": "V1", "sku": "RS-0001-S", "code": "RS-0001-S", "stock": 4, "warehouse": AMERICAS}
    ],
    "thumbnail": "http://127.0.0.1:8020/thumbnail/P1/64/",
    "privateMetadata": [],
}
MUG = {
    "id": "P2",
    "name": "Harbour Mug",
    "slug": "harbour-mug",
    "rating": None,
    "category": None,
    "price": {"amount": 8.0, "currency": "USD"},
    "variants": [],
    "thumbnail": None,
    "privateMetadata": [],
}
SHOP_DATA = {
    "products": _connection([TEE, MUG]),
    "product": TEE,
    "orders": _connection([]),
    "shop": {"name": "Sample shop", "description": None, "defaultChannel": None},
    "channels": [{"id": "CH1", "slug": "default-channel", "currency": "USD"}],
    "currencies": ["USD"],
    "report": {"amount": 120.5, "currency": "USD"},
}


def store_client(schema: GraphQLSchema, data: dict, requests: list[str] | None = None):
    """A client whose server executes queries on ``schema`` over in-memory ``data``."""

    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        if requests is not None:
            requests.append(query)
        result = graphql_sync(schema, query, root_value=data)
        body = {"data": result.data}
        if result.errors:
            body["errors"] = [error.formatted for error in result.errors]
        return httpx.Response(200, json=body)

    return SaleorClient("http://saleor.test/graphql/", transport=httpx.MockTransport(handler))


@pytest.fixture
def shop_client(shop_schema) -> SaleorClient:
    """A store with two products, one channel and no orders."""
    return store_client(shop_schema, SHOP_DATA)


@pytest.fixture(scope="session")
def new_shop_schema() -> GraphQLSchema:
    return build_schema(NEW_SHOP_SDL)


@pytest.fixture
def new_shop_client(new_shop_schema) -> SaleorClient:
    """The same store served by the later version."""
    return store_client(new_shop_schema, SHOP_DATA)


class ScriptedTeacher:
    """Stands in for the model: answers each prompt with what ``reply`` returns."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []
        self.options: list[dict] = []

    def complete(self, prompt: str, **options) -> str:
        self.prompts.append(prompt)
        self.options.append(options)
        return self.reply(prompt)
