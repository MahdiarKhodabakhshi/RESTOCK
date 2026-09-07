from random import Random

import pytest
from graphql import build_ast_schema, graphql_sync, parse

from restock.schema.diff import ChangeKind, diff_schemas
from restock.schema.mutate import (
    InvalidQuery,
    Remove,
    Rename,
    apply_mutations,
    mutable_fields,
    rewrite_query,
    sample_mutations,
)

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

DATA = {
    "product": {
        "id": "p1",
        "name": "Mug",
        "slug": "mug",
        "price": {"amount": 12.5, "currency": "USD"},
        "variants": [{"id": "v1", "sku": "MUG-S", "price": {"amount": 12.5, "currency": "USD"}}],
    }
}

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


def test_rewritten_query_runs_on_the_real_schema_with_the_requested_keys():
    query = """
    query ($id: ID!) {
      product(id: $id) {
        name
        price { amountInfo currency }
        items: variantsData { ...variant }
      }
    }
    fragment variant on Variant { skuValue price { amountInfo } }
    """

    result = graphql_sync(
        BASE, rewrite_query(query, MUTATED, MUTATIONS), DATA, variable_values={"id": "p1"}
    )

    assert result.errors is None
    assert result.data == {
        "product": {
            "name": "Mug",
            "price": {"amountInfo": 12.5, "currency": "USD"},
            "items": [{"skuValue": "MUG-S", "price": {"amountInfo": 12.5}}],
        }
    }


def test_untouched_query_is_unchanged_in_meaning():
    query = "{ product(id: 1) { name price { currency } } }"

    rewritten = rewrite_query(query, MUTATED, MUTATIONS)

    assert parse(rewritten, no_location=True) == parse(query, no_location=True)


@pytest.mark.parametrize(
    "stale_query",
    [
        "{ product(id: 1) { slug } }",
        "{ product(id: 1) { variants { id } } }",
        "{ product(id: 1) { price { amount } } }",
    ],
)
def test_query_using_the_old_schema_is_rejected(stale_query):
    with pytest.raises(InvalidQuery, match="Cannot query field"):
        rewrite_query(stale_query, MUTATED, MUTATIONS)
