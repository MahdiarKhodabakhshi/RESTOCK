import pytest
from graphql import build_ast_schema, parse

from restock.schema.diff import ChangeKind, diff_schemas
from restock.schema.mutate import Remove, Rename, apply_mutations

SDL = """
type Query {
  product(id: ID!): Product
  node(id: ID!): Node
}

type Mutation {
  productDelete(id: ID!): Product
}

interface Node { id: ID! }

type Product implements Node {
  id: ID!
  name: String!
  slug: String
  price: Money
  variants: [Variant!]
}

type Variant implements Node {
  id: ID!
  sku: String
  price: Money
}

type Money { amount: Float!, currency: String! }
type Token { value: String }
"""

DOCUMENT = parse(SDL)
BASE = build_ast_schema(DOCUMENT)


MUTATIONS = [
    Rename("Product", "variants", "variantsData"),
    Rename("Variant", "sku", "skuValue"),
    Rename("Money", "amount", "amountInfo"),
    Remove("Product", "slug"),
]
MUTATED = build_ast_schema(apply_mutations(DOCUMENT, MUTATIONS))


def test_apply_mutations_changes_only_the_named_fields():
    changes = {(c.kind, c.coordinate) for c in diff_schemas(BASE, MUTATED)}

    assert changes == {
        (ChangeKind.FIELD_REMOVED, "Product.variants"),
        (ChangeKind.FIELD_ADDED, "Product.variantsData"),
        (ChangeKind.FIELD_REMOVED, "Variant.sku"),
        (ChangeKind.FIELD_ADDED, "Variant.skuValue"),
        (ChangeKind.FIELD_REMOVED, "Money.amount"),
        (ChangeKind.FIELD_ADDED, "Money.amountInfo"),
        (ChangeKind.FIELD_REMOVED, "Product.slug"),
    }
    assert diff_schemas(BASE, build_ast_schema(DOCUMENT)) == []


@pytest.mark.parametrize(
    "mutation", [Remove("Product", "missing"), Rename("Missing", "name", "title")]
)
def test_apply_mutations_rejects_unknown_targets(mutation):
    with pytest.raises(KeyError, match="[Mm]issing"):
        apply_mutations(DOCUMENT, [mutation])
