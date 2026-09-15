from graphql import build_schema

from restock.agent.render import render_type

SCHEMA = build_schema("""
type Query {
  product(id: ID!): Product
}

interface Node { id: ID! }

type Product implements Node {
  id: ID!
  name: String!
  status: ProductStatus
}

enum ProductStatus { DRAFT ACTIVE }

input ProductFilter { search: String }

union SearchResult = Product

scalar Date
""")


def test_render_object_keeps_fields_args_and_interfaces():
    text = render_type(SCHEMA.type_map["Product"])
    assert text == (
        "type Product implements Node {\n  id: ID!\n  name: String!\n  status: ProductStatus\n}"
    )


def test_render_object_includes_field_arguments():
    text = render_type(SCHEMA.type_map["Query"])
    assert "product(id: ID!): Product" in text


def test_render_enum_input_union_and_scalar():
    assert render_type(SCHEMA.type_map["ProductStatus"]) == (
        "enum ProductStatus {\n  DRAFT\n  ACTIVE\n}"
    )
    assert render_type(SCHEMA.type_map["ProductFilter"]) == (
        "input ProductFilter {\n  search: String\n}"
    )
    assert render_type(SCHEMA.type_map["SearchResult"]) == "union SearchResult = Product"
    assert render_type(SCHEMA.type_map["Date"]) == "scalar Date"
