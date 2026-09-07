from random import Random

import pytest
from graphql import build_ast_schema, parse

from restock.schema.diff import ChangeKind, diff_schemas
from restock.schema.mutate import Remove, Rename, apply_mutations, mutable_fields, sample_mutations

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


def test_mutable_fields_skip_interface_fields_and_the_mutation_side():
    fields = mutable_fields(BASE)

    assert ("Product", "name") in fields
    assert ("Query", "product") in fields
    assert ("Product", "id") not in fields
    assert not {type_name for type_name, _ in fields} & {"Node", "Mutation", "Token"}


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


@pytest.mark.parametrize("seed", range(20))
def test_sampled_mutations_produce_exactly_the_declared_changes(seed):
    mutations = sample_mutations(BASE, Random(seed), renames=3, removals=2)

    mutated = build_ast_schema(apply_mutations(DOCUMENT, mutations))

    expected = set()
    for mutation in mutations:
        expected.add((ChangeKind.FIELD_REMOVED, f"{mutation.type_name}.{mutation.field}"))
        if isinstance(mutation, Rename):
            expected.add((ChangeKind.FIELD_ADDED, f"{mutation.type_name}.{mutation.new_name}"))
    assert {(c.kind, c.coordinate) for c in diff_schemas(BASE, mutated)} == expected
    assert len({(m.type_name, m.field) for m in mutations}) == 5


def test_sample_mutations_never_empties_a_type():
    with pytest.raises(ValueError, match="too few"):
        sample_mutations(BASE, Random(0), removals=len(mutable_fields(BASE)))
