from graphql import build_schema, parse

from restock.schema.diff import diff_schemas
from restock.tasks.coordinates import Touched, touched, used_coordinates

OLD = build_schema("""
type Query {
  products(first: Int, filter: ProductFilter, sortBy: ProductOrder): [Product!]!
  node(id: ID!): Node
  downloads: [Download!]!
}
interface Node { id: ID! }
type Product implements Node {
  id: ID!
  name: String!
  rating: Float
  download: Download
  price(currency: Currency): Float
}
type Download implements Node { id: ID!, url: String! }
input ProductFilter { search: String, price: PriceRange }
input PriceRange { gte: Float, lte: Float }
input ProductOrder { field: ProductOrderField!, direction: Direction! }
enum ProductOrderField { NAME RATING }
enum Direction { ASC DESC }
enum Currency { USD PLN }
""")

NEW = build_schema("""
type Query {
  products(first: Int, filter: ProductFilter, sortBy: ProductOrder): [Product!]!
  node(id: ID!): Node
}
interface Node { id: ID! }
type Product implements Node {
  id: ID!
  name: String!
  rating: Float @deprecated(reason: "Use reviews.")
  price(currency: Currency): Float
}
input ProductFilter { search: String, price: PriceRange }
input PriceRange { gte: Float, lte: Float }
input ProductOrder { field: ProductOrderField!, direction: Direction! }
enum ProductOrderField { NAME }
enum Direction { ASC DESC }
enum Currency { USD PLN }
""")


def coordinates(query: str) -> frozenset[str]:
    return used_coordinates(OLD, parse(query))


def test_fields_and_arguments():
    assert coordinates("{ products(first: 2) { name price(currency: USD) } }") == {
        "Query.products",
        "Query.products(first:)",
        "Product.name",
        "Product.price",
        "Product.price(currency:)",
        "Currency.USD",
    }


def test_input_fields_and_enum_values_inside_input_objects():
    used = coordinates("""
    {
      products(
        filter: { price: { gte: 5 } }
        sortBy: { field: RATING, direction: DESC }
      ) { id }
    }
    """)
    assert {
        "ProductFilter.price",
        "PriceRange.gte",
        "ProductOrder.field",
        "ProductOrder.direction",
        "ProductOrderField.RATING",
        "Direction.DESC",
    } <= used


def test_fragments_name_their_type_and_resolve_fields_on_it():
    used = coordinates('{ node(id: "1") { id ... on Product { name } } }')
    assert {"Node.id", "Product", "Product.name"} <= used
    assert "Node.name" not in used


def test_aliases_meta_fields_and_directives_are_not_coordinates():
    used = coordinates("{ items: products { __typename title: name @include(if: true) } }")
    assert used == {"Query.products", "Product.name"}


def test_touched_separates_broken_from_deprecated():
    changes = diff_schemas(OLD, NEW)
    query = "{ products(sortBy: { field: RATING, direction: ASC }) { rating download { url } } }"
    assert touched(coordinates(query), changes) == Touched(
        broken=("Download.url", "Product.download", "ProductOrderField.RATING"),
        deprecated=("Product.rating",),
    )


def test_untouched_query():
    changes = diff_schemas(OLD, NEW)
    assert touched(coordinates("{ products { id name } }"), changes) == Touched()
