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

The environment and the task set: the schema history, the simulated versions,
the reward, a local Saleor that serves one store from several releases, the
pipeline that builds (question, query) tasks from that store, and the schema
tools an agent uses during an episode. Model training is not part of this code.

| Module | What it does |
| --- | --- |
| `restock.schema.fetch` | Downloads the schema of any Saleor release tag |
| `restock.schema.diff` | Structured diff of two schema versions, with reachability from `Query` |
| `restock.schema.mutate` | Simulated versions: rename and remove fields, rewrite queries back to the real schema |
| `restock.reward` | Episode reward and result comparison |
| `restock.agent.tools` | Schema-reading tools (`read_type`, `search_fields`) with cumulative token counts |
| `restock.agent.signal` | Version signal: release id and query-side breaking changes since training |
| `restock.agent.policies` | Baselines: always-read, memorised-only, and selective tool use |
| `restock.agent.evaluate` | Episode loop and accuracy / read-token report by changed vs untouched |
| `restock.saleor.runtime` | Runs Postgres and Saleor releases with Apptainer |
| `restock.saleor.seed` | Adds products, customers, orders, digital content and app extensions to the sample store |
| `restock.saleor.build` | Builds the store once and upgrades a copy for each later release |
| `restock.saleor.client` | Executes GraphQL operations against a release |
| `restock.tasks` | Builds the task set: focus sampling, teacher prompts, execution checks, verification, porting to later releases and the train/test split |

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

An episode binds tools to the schema the agent may see. The version signal is
prompt context; only ``read_type`` and ``search_fields`` add to ``read_tokens``:

```python
from graphql import build_schema

from restock.agent import SchemaTools, version_signal
from restock.schema.diff import diff_schemas
from restock.schema.fetch import load_schema

old, new = build_schema(load_schema("3.20.0")), build_schema(load_schema("3.23.0"))
tools = SchemaTools(
    new,
    version_signal("3.23.0", trained_on="3.20.0", changes=diff_schemas(old, new)),
)
tools.version_signal()           # free
tools.read_type("Product")       # charged
tools.search_fields("checkout")  # charged
tools.read_tokens                # into restock.reward.reward
```

## Local Saleor

Queries are executed against real Saleor releases. They run with
[Apptainer](https://apptainer.org), so neither Docker nor root is needed.

```bash
uv run restock saleor build 3.20 3.21 3.22 3.23   # about ten minutes
uv run restock saleor up 3.20 3.23
uv run restock saleor status
uv run restock saleor down
```

`build` creates the sample store on the first release, adds products, customers,
orders, digital content, customer history and app extensions to it, and then
upgrades a copy of the database for each later release. Every release therefore
serves the same store, and a query gives a different answer on two of them only
when the API itself changed:

```python
from restock.saleor.client import SaleorClient
from restock.saleor.runtime import Runtime, api_url

token = Runtime().token()
query = "{ digitalContents(first: 1) { totalCount } }"

SaleorClient(api_url("3.22"), token).execute(query).ok  # True
SaleorClient(api_url("3.23"), token).execute(query).messages
# ['Cannot query field "digitalContents" on type "Query".']
```

## Task set

A local teacher model writes (question, query) pairs from a slice of the schema
and real values from the store. A pair is kept only when the query validates,
runs without errors and returns non-empty data. Each task is tagged with the
schema coordinates it uses and whether the 3.20 → 3.23 diff touches any of them.
Tasks whose request does not determine their answer are dropped; the rest are
split so that related questions stay on one side, and the test tasks get a
reference answer on 3.23.

The teacher is served with vLLM under Slurm (`scripts/serve_model.sh`), because
this machine kills GPU processes that are not in a Slurm job.

```bash
# Two replicas of the teacher, one GPU each
sbatch scripts/serve_model.sh Qwen/Qwen3.8-27B teacher 8100
sbatch scripts/serve_model.sh Qwen/Qwen3.8-27B teacher 8101

uv run restock saleor up 3.20 3.23
uv run restock tasks generate --requests 3000 \
  --teacher http://127.0.0.1:8100/v1 http://127.0.0.1:8101/v1
uv run restock tasks verify \
  --teacher http://127.0.0.1:8100/v1 http://127.0.0.1:8101/v1
uv run restock tasks split
uv run restock tasks port \
  --teacher http://127.0.0.1:8100/v1 http://127.0.0.1:8101/v1
uv run restock tasks stats
```

## Baselines

Serve the student, then score a policy on a task file. Reports accuracy and mean
schema tokens read, split into changed / deprecated / untouched. On a later
release only tasks with a ``same`` or ``ported`` reference are scored.

```bash
sbatch scripts/serve_model.sh Qwen/Qwen3-8B student 8200

uv run restock eval data/tasks/test.jsonl --policy always_read \
  --release 3.20 --student http://127.0.0.1:8200/v1 --limit 50
uv run restock eval data/tasks/test.jsonl --policy memorised \
  --release 3.23 --trained-on 3.20.0 --student http://127.0.0.1:8200/v1
```

Supervised examples for the memorised baseline (no schema in the prompt):

```bash
uv run restock sft data/tasks/train.jsonl --out data/sft/memorised_3.20.jsonl --version 3.20
uv run restock sft data/tasks/train.jsonl --out data/sft/always_read_3.20.jsonl \
  --version 3.20 --with-schema --schema-version 3.20.0

# LoRA SFT under Slurm, then serve the merged weights
sbatch scripts/sft_memorised.sh
sbatch scripts/serve_model.sh data/models/memorised-3.20/merged memorised 8201

# Efficiency-RL (GRPO) from the memorised weights; Saleor 3.20 must be up
sbatch scripts/grpo_efficiency.sh
sbatch scripts/serve_model.sh data/models/efficiency-3.20/merged efficiency 8200
uv run restock eval data/tasks/test.jsonl --policy efficiency \
  --release 3.20 --student http://127.0.0.1:8200/v1 --model efficiency
```
