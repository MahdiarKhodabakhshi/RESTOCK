# RESTOCK

**RE**inforcement-learned **S**elective **T**ool-reading **O**ver **C**ached **K**nowledge.

An agent that writes GraphQL queries spends most of its tokens reading the API
schema. RESTOCK trains the agent to keep the schema in its weights and to re-read
only the parts that have changed since training, so it stays cheap on a stable
API and correct after a new API version ships.

## Setup

```bash
uv sync
uv run pytest
```
