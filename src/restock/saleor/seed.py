"""Grow the sample store that ``populatedb`` creates.

The stock sample has 32 products, which leaves most filters and aggregates with
almost nothing to return. Products are added through the API from a seeded plan,
so a rebuilt store has the same catalogue.
"""

from dataclasses import dataclass
from random import Random
from typing import Any

from restock.saleor.client import Result, SaleorClient


@dataclass(frozen=True)
class Line:
    """A family of products that share a category, a product type and a price range."""

    category: str
    product_type: str
    nouns: tuple[str, ...]
    price: tuple[int, int]
    product_attribute: str | None = None
    variant_attribute: str | None = None


LINES = (
    Line("T-shirts", "Top", ("Tee", "Shirt", "Long Sleeve"), (15, 45), "material", "size"),
    Line("Polo shirts", "Top", ("Polo",), (25, 60), "material", "size"),
    Line("Sweatshirts", "Sweatshirt", ("Hoodie", "Crewneck", "Zip Hoodie"), (35, 90), "material"),
    Line("Sneakers", "Shoe", ("Runners", "Plimsolls", "High Tops"), (50, 160), None, "shoe-size"),
    Line("Beanies", "Beanies & Scarfs", ("Beanie",), (10, 30), "material"),
    Line("Scarfs", "Beanies & Scarfs", ("Scarf", "Neck Warmer"), (12, 40), "material"),
    Line("Sunglasses", "Simple", ("Sunglasses", "Aviators", "Sunnies"), (20, 120)),
    Line("Homewares", "Simple", ("Mug", "Cushion", "Blanket", "Candle", "Poster"), (8, 70)),
    Line("Juices", "Juice", ("Juice", "Smoothie"), (2, 8), "flavor"),
)  # fmt: skip

STYLES = (
    "Alpine", "Arctic", "Coastal", "Desert", "Harbour", "Midnight", "Nordic", "Prairie",
    "Summit", "Urban", "Vintage", "Meadow", "Canyon", "Lunar", "Solar", "Tidal",
)  # fmt: skip
COLOURS = (
    "Amber", "Black", "Coral", "Forest", "Indigo", "Ivory", "Olive", "Rust", "Sand", "Slate",
    "Teal", "Violet",
)  # fmt: skip

# Rough conversion from USD, enough to keep prices plausible in other channels.
EXCHANGE_RATES = {"USD": 1.0, "PLN": 4.0, "EUR": 0.9}


@dataclass(frozen=True)
class Attribute:
    id: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class Channel:
    id: str
    slug: str
    currency: str


@dataclass(frozen=True)
class Store:
    """The IDs a product needs to refer to."""

    categories: dict[str, str]
    product_types: dict[str, str]
    attributes: dict[str, Attribute]
    channels: tuple[Channel, ...]
    warehouses: tuple[str, ...]


@dataclass(frozen=True)
class ProductPlan:
    product: dict[str, Any]
    listings: list[dict[str, Any]]
    variants: list[dict[str, Any]]


STORE_QUERY = """
{
  categories(first: 100) { edges { node { id name } } }
  productTypes(first: 100) { edges { node { id name } } }
  attributes(first: 100) {
    edges { node { id slug choices(first: 100) { edges { node { slug } } } } }
  }
  channels { id slug currencyCode }
  warehouses(first: 100) { edges { node { id } } }
}
"""

CREATE_PRODUCT = """
mutation ($input: ProductCreateInput!) {
  productCreate(input: $input) { product { id } errors { field message } }
}
"""

LIST_PRODUCT = """
mutation ($id: ID!, $input: ProductChannelListingUpdateInput!) {
  productChannelListingUpdate(id: $id, input: $input) { errors { field message } }
}
"""

CREATE_VARIANTS = """
mutation ($product: ID!, $variants: [ProductVariantBulkCreateInput!]!) {
  productVariantBulkCreate(product: $product, variants: $variants) {
    count
    errors { field message }
  }
}
"""


def read_store(client: SaleorClient) -> Store:
    data = _data(client.execute(STORE_QUERY))
    nodes = lambda key: [edge["node"] for edge in data[key]["edges"]]  # noqa: E731
    return Store(
        categories={node["name"]: node["id"] for node in nodes("categories")},
        product_types={node["name"]: node["id"] for node in nodes("productTypes")},
        attributes={
            node["slug"]: Attribute(
                node["id"], tuple(edge["node"]["slug"] for edge in node["choices"]["edges"])
            )
            for node in nodes("attributes")
        },
        channels=tuple(
            Channel(channel["id"], channel["slug"], channel["currencyCode"])
            for channel in data["channels"]
        ),
        warehouses=tuple(node["id"] for node in nodes("warehouses")),
    )


def plan_products(store: Store, count: int, seed: int = 0) -> list[ProductPlan]:
    """Plan ``count`` products spread evenly over the product lines."""
    rng = Random(seed)
    names: set[str] = set()
    plans = []
    while len(plans) < count:
        line = LINES[len(plans) % len(LINES)]
        name = f"{rng.choice(STYLES)} {rng.choice(COLOURS)} {rng.choice(line.nouns)}"
        if name in names:
            continue
        names.add(name)
        plans.append(_plan_product(store, line, name, len(plans) + 1, rng))
    return plans


def create_products(client: SaleorClient, plans: list[ProductPlan]) -> None:
    for plan in plans:
        created = _data(client.execute(CREATE_PRODUCT, {"input": plan.product}))["productCreate"]
        _raise_for_errors(created, plan.product["name"])
        product_id = created["product"]["id"]

        listing = {"id": product_id, "input": {"updateChannels": plan.listings}}
        listed = _data(client.execute(LIST_PRODUCT, listing))["productChannelListingUpdate"]
        _raise_for_errors(listed, plan.product["name"])

        variants = {"product": product_id, "variants": plan.variants}
        added = _data(client.execute(CREATE_VARIANTS, variants))["productVariantBulkCreate"]
        _raise_for_errors(added, plan.product["name"])


def _plan_product(store: Store, line: Line, name: str, number: int, rng: Random) -> ProductPlan:
    product: dict[str, Any] = {
        "name": name,
        "slug": name.lower().replace(" ", "-"),
        "category": store.categories[line.category],
        "productType": store.product_types[line.product_type],
    }
    if line.product_attribute:
        attribute = store.attributes[line.product_attribute]
        product["attributes"] = [_attribute_value(attribute, rng.choice(attribute.choices))]

    # Most products sell in every channel; some miss one and a few are drafts.
    published = rng.random() < 0.95
    channels = list(store.channels)
    if len(channels) > 1 and rng.random() < 0.3:
        channels.remove(rng.choice(channels))
    listings = [
        {
            "channelId": channel.id,
            "isPublished": published,
            "visibleInListings": published,
            "isAvailableForPurchase": published,
        }
        for channel in channels
    ]

    if line.variant_attribute:
        attribute = store.attributes[line.variant_attribute]
        size = rng.randint(2, min(5, len(attribute.choices)))
        picked = set(rng.sample(attribute.choices, size))
        options = [choice for choice in attribute.choices if choice in picked]
    else:
        attribute, options = None, [None]

    price = rng.randint(*line.price) - 0.01
    cost_share = rng.uniform(0.4, 0.7)
    variants = []
    for option in options:
        suffix = f"-{option.upper()}" if option else ""
        variants.append(
            {
                "sku": f"RS-{number:04d}{suffix}",
                "name": option.upper() if option else name,
                "attributes": [_attribute_value(attribute, option)] if attribute else [],
                "stocks": _plan_stocks(store, rng),
                "channelListings": [
                    {
                        "channelId": channel.id,
                        "price": _convert(price, channel.currency),
                        "costPrice": _convert(price * cost_share, channel.currency),
                    }
                    for channel in channels
                ],
            }
        )
    return ProductPlan(product, listings, variants)


def _plan_stocks(store: Store, rng: Random) -> list[dict[str, Any]]:
    """Stock in one to three warehouses: some sold out, some running low."""
    warehouses = rng.sample(store.warehouses, rng.randint(1, min(3, len(store.warehouses))))
    stocks = []
    for warehouse in warehouses:
        roll = rng.random()
        quantity = 0 if roll < 0.1 else rng.randint(1, 5) if roll < 0.3 else rng.randint(6, 400)
        stocks.append({"warehouse": warehouse, "quantity": quantity})
    return stocks


def _attribute_value(attribute: Attribute, choice: str) -> dict[str, Any]:
    return {"id": attribute.id, "dropdown": {"value": choice}}


def _convert(usd: float, currency: str) -> float:
    return round(usd * EXCHANGE_RATES.get(currency, 1.0), 2)


def _data(result: Result) -> dict[str, Any]:
    if not result.ok:
        raise RuntimeError("; ".join(result.messages))
    return result.data


def _raise_for_errors(payload: dict[str, Any], name: str) -> None:
    if payload["errors"]:
        details = "; ".join(f"{e['field']}: {e['message']}" for e in payload["errors"])
        raise RuntimeError(f"{name}: {details}")
