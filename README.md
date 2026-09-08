# RESTOCK

**RE**inforcement-learned **S**elective **T**ool-reading **O**ver **C**ached **K**nowledge.

An agent that writes GraphQL queries spends most of its tokens reading the API
schema. Shopify reports about 40 tokens read for every token written by the
GraphQL expert behind Sidekick. A fine-tuned model could answer from memory
instead, but the API ships a new version every quarter and memory goes stale.

RESTOCK treats the model's weights as a cache of the API and uses reinforcement
learning to train the cache-validation policy: answer from memory when the
schema has not moved, re-read the part that has. The target is an expert that
reads far less than an always-read agent on a stable API and stays correct
after a version change without being retrained.

## Approach

1. **Memorise.** Supervised training on one API version, so the model can write
   queries without reading the schema.
2. **RL over simulated versions.** Each episode renames or removes fields in the
   schema the agent sees and tells it which version it is on. The reward is
   execution correctness, discounted by the schema tokens the agent chose to read.
3. **Test on real releases.** Evaluate on later Saleor releases the model never
   saw, against an always-read agent, a memorised-only model and an
   efficiency-RL model trained on a single version.

[Saleor](https://github.com/saleor/saleor) is the environment: an open-source
commerce API with a public schema for every release, so the version history is
real and queries can be executed.

## What is in this repository

The environment side of the method: the schema history, the simulated versions
and the reward. Model training is not part of this code.

| Module | What it does |
| --- | --- |
| `restock.schema.fetch` | Downloads the schema of any Saleor release tag |
| `restock.schema.diff` | Structured diff of two schema versions, with reachability from `Query` |
| `restock.schema.mutate` | Simulated versions: rename and remove fields, rewrite queries back to the real schema |
| `restock.reward` | Episode reward and result comparison |

## How much does the API really change?

This decides whether the problem is worth solving, so it was measured first.
Fields reachable from `Query` and how many of them break between releases:

| From | To | Fields | Broken | Share | Deprecated | Added |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3.14.0 | 3.15.0 | 1918 | 9 | 0.47% | 4 | 26 |
| 3.15.0 | 3.16.0 | 1967 | 0 | 0.00% | 0 | 3 |
| 3.16.0 | 3.17.0 | 1971 | 1 | 0.05% | 3 | 9 |
| 3.17.0 | 3.18.0 | 2075 | 2 | 0.10% | 0 | 7 |
| 3.18.0 | 3.19.0 | 2091 | 0 | 0.00% | 3 | 16 |
| 3.19.0 | 3.20.0 | 2109 | 0 | 0.00% | 1 | 33 |
| 3.20.0 | 3.21.0 | 2142 | 1 | 0.05% | 19 | 26 |
| 3.21.0 | 3.22.0 | 2177 | 0 | 0.00% | 23 | 26 |
| 3.22.0 | 3.23.0 | 2408 | 46 | 1.91% | 9 | 16 |
| 3.14.0 | 3.23.0 | 1918 | 52 | 2.71% | 58 | 138 |

"Added" counts fields added to types that already existed. The number of
breaking changes matches graphql-core's `find_breaking_changes` on all nine
consecutive pairs.

Two things follow:

- Over nine releases 97% of the query side kept working. Re-reading the schema
  on every request pays for a change that almost never happened, which is the
  case for reading selectively.
- Real breaking changes are few and arrive in bursts (3.15 and 3.23). A test set
  of random tasks would barely touch them, so the real-release evaluation has to
  be built from tasks that use changed fields, and reported separately from the
  untouched ones. It also means training has to rely on simulated changes.

## Usage

```bash
uv sync
uv run pytest

uv run restock fetch 3.22.0 3.23.0
uv run restock diff 3.22.0 3.23.0 -v
```

Simulated versions run against an unmodified server:

```python
from random import Random

from graphql import build_ast_schema, parse

from restock.schema.fetch import load_schema
from restock.schema.mutate import apply_mutations, rewrite_query, sample_mutations

document = parse(load_schema("3.23.0"))
mutations = sample_mutations(build_ast_schema(document), Random(0), renames=40, removals=20)
mutated = build_ast_schema(apply_mutations(document, mutations))

# Raises InvalidQuery if the query uses a field the mutated version no longer has.
query_for_saleor = rewrite_query(agent_query, mutated, mutations)
```
