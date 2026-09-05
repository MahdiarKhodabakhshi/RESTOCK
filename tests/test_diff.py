from graphql import build_schema, find_breaking_changes

from restock.schema.diff import Change, ChangeKind, diff_schemas, reachable_types

OLD = build_schema("""
type Query {
  product(id: ID!, channel: String): Product
  products(first: Int, filter: ProductFilter): [Product!]!
  node(id: ID!): Node
}

type Mutation {
  productDelete(id: ID!): Product
  tokenCreate(email: String!): Token
}

interface Node { id: ID! }

type Product implements Node {
  id: ID!
  name: String!
  slug: String
  price: Money
  status: ProductStatus
  media: [Media!]
}

type Money { amount: Float!, currency: String! }
type Media { url: String! }
type Token { value: String }

enum ProductStatus { DRAFT ACTIVE ARCHIVED }

input ProductFilter {
  search: String
  isPublished: Boolean
}
""")

NEW = build_schema("""
type Query {
  product(id: ID!, slug: String): Product
  products(first: Int!, filter: ProductFilter, channel: String!): [Product!]!
  node(id: ID!): Node
}

type Mutation {
  productDelete(id: ID!): Product
}

interface Node { id: ID! }

type Product {
  id: ID!
  name: String
  slug: String!
  price: Float
  status: ProductStatus @deprecated(reason: "Use `state`.")
  state: String
}

type Money { amount: Float!, currency: String! }

enum ProductStatus { DRAFT ACTIVE }

input ProductFilter {
  search: [String!]
  stock: Int!
  ids: [ID!]
}
""")


def changes_by_coordinate(old=OLD, new=NEW) -> dict[str, set[ChangeKind]]:
    found: dict[str, set[ChangeKind]] = {}
    for change in diff_schemas(old, new):
        found.setdefault(change.coordinate, set()).add(change.kind)
    return found


def test_identical_schemas_have_no_changes():
    assert diff_schemas(OLD, OLD) == []


def test_reports_each_change_at_its_coordinate():
    assert changes_by_coordinate() == {
        "Query.product(channel:)": {ChangeKind.ARG_REMOVED},
        "Query.products(first:)": {ChangeKind.ARG_TYPE_CHANGED},
        "Query.products(channel:)": {ChangeKind.REQUIRED_ARG_ADDED},
        "Mutation.tokenCreate": {ChangeKind.FIELD_REMOVED},
        "Token": {ChangeKind.TYPE_REMOVED},
        "Product.Node": {ChangeKind.INTERFACE_REMOVED},
        "Product.name": {ChangeKind.FIELD_TYPE_CHANGED},
        "Product.price": {ChangeKind.FIELD_TYPE_CHANGED},
        "Product.media": {ChangeKind.FIELD_REMOVED},
        "Product.state": {ChangeKind.FIELD_ADDED},
        "Media": {ChangeKind.TYPE_REMOVED},
        "ProductStatus.ARCHIVED": {ChangeKind.ENUM_VALUE_REMOVED},
        "ProductFilter.search": {ChangeKind.FIELD_TYPE_CHANGED},
        "ProductFilter.isPublished": {ChangeKind.FIELD_REMOVED},
        "ProductFilter.stock": {ChangeKind.REQUIRED_INPUT_FIELD_ADDED},
        "ProductFilter.ids": {ChangeKind.FIELD_ADDED},
    }


def test_stricter_output_and_looser_input_are_not_changes():
    old = build_schema("type Query { a(x: Int!, y: [Int!]): String, b: [String] }")
    new = build_schema("type Query { a(x: Int, y: [Int]): String!, b: [String!]! }")

    assert diff_schemas(old, new) == []


def test_breaking_changes_agree_with_graphql_core():
    ours = [change for change in diff_schemas(OLD, NEW) if change.breaking]

    assert len(ours) == len(find_breaking_changes(OLD, NEW))


def test_marks_changes_reachable_from_query():
    by_coordinate = {change.coordinate: change for change in diff_schemas(OLD, NEW)}

    assert by_coordinate["Product.media"].query_side
    assert by_coordinate["ProductFilter.isPublished"].query_side
    assert by_coordinate["ProductStatus.ARCHIVED"].query_side
    assert not by_coordinate["Mutation.tokenCreate"].query_side
    assert not by_coordinate["Token"].query_side


def test_reachable_types_follow_interfaces_and_arguments():
    assert reachable_types(OLD, OLD.query_type) == {
        "Query", "Node", "Product", "Money", "Media", "ProductStatus", "ProductFilter",
        "ID", "String", "Int", "Float", "Boolean",
    }  # fmt: skip
    assert reachable_types(OLD, None) == frozenset()


def test_change_coordinate_formats():
    assert Change(ChangeKind.TYPE_REMOVED, "Media").coordinate == "Media"
    assert Change(ChangeKind.FIELD_REMOVED, "Product", "media").coordinate == "Product.media"
    assert (
        Change(ChangeKind.ARG_REMOVED, "Query", "product", "channel").coordinate
        == "Query.product(channel:)"
    )
