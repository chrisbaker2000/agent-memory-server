# agent-memory-server-fork — Claude Code Context

## What This Is

A fork of [redis/agent-memory-server](https://github.com/redis/agent-memory-server) (`server/v0.13.2`) that adds **multi-user attribution** and **memory lifecycle** fields. These changes support OpenClaw's family assistant use case, where a single memory server stores memories for multiple family members with per-user visibility controls.

**Upstream**: `origin` points to `https://github.com/redis/agent-memory-server.git`
**Fork branch**: `fork/openclaw-attribution` (branched from tag `server/v0.13.2`)
**Fork location**: `/Users/chris/Developer/homelab/openclaw/agent-memory-server-fork/`

## What Was Changed (8 commits, ~1,042 lines added)

### 4 New Fields on `MemoryRecord`

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `source_user` | `str \| None` | `None` | Username of the person who created/contributed this memory (e.g. `chris`, `lindsey`). Set at creation, propagated through merge and extraction. |
| `source_channel` | `str \| None` | `None` | Channel or integration where the memory originated (e.g. `discord`, `slack`, `whatsapp`). Enables per-channel recall filtering. |
| `visibility` | `str` | `"everyone"` | Access scope — who can see this memory. Values: `everyone`, `admin`, `restricted`, `kids`, or custom scopes. Most-restrictive wins on merge. |
| `stale_after` | `datetime \| None` | `None` | Optional expiry hint. The forgetting pipeline (`compact_long_term_memories`) deletes memories past this datetime. |

### Changes by Layer

1. **Models** (`models.py`) — Added 4 fields to `MemoryRecord`, `SearchRequest`, `CreateMemoryRecordRequest`, and `EditMemoryRecordRequest`. Search request builds filter expressions for the 3 TAG fields.

2. **Filters** (`filters.py`) — Added `SourceUser`, `SourceChannel`, and `VisibilityFilter` TAG filter types for RedisVL query building.

3. **Vectorstore** (`memory_vector_db.py`, `memory_vector_db_factory.py`) — Persist the 4 fields as Redis hash fields (3 TAGs + 1 NUMERIC for `stale_after`). Parse them back on read. Index schema updated with 4 new `FieldInfo` entries.

4. **Merge pipeline** (`long_term_memory.py`) — `merge_memories_with_llm()` propagates attribution from the kept memory. Visibility uses most-restrictive-wins logic (priority: `restricted` > `admin` > `kids` > `everyone`).

5. **Extraction pipeline** (`extraction.py`) — `index_long_term_memories()` copies parent's `source_user`, `source_channel`, and `visibility` to all extracted child memories.

6. **Forgetting pipeline** (`long_term_memory.py`) — `compact_long_term_memories()` now checks `stale_after` and deletes expired memories in batches of 100.

7. **API endpoints** (`api.py` via models, `mcp.py`) — Search, create, and edit endpoints accept the new fields. MCP `create_long_term_memories` and `search_long_term_memory` tools pass them through.

8. **Dedup fix** (`long_term_memory.py`) — Fixed variable shadowing bug in `deduplicate_by_semantic_search` where inner loop variable `memory` overwrote outer loop's `memory`.

### Test Coverage

- `tests/test_attribution.py` — 15 tests covering CRUD, search filtering, merge propagation, extraction inheritance, visibility semantics, and edge cases.
- `tests/test_forgetting.py` — 11 tests covering stale_after expiry, batch deletion, mixed stale/fresh memories, no-stale-memories no-op, and compact integration.

## How to Rebase on Upstream Releases

The fork is structured as linear commits on top of a tagged upstream release. To rebase onto a new upstream version (e.g. `server/v0.14.0`):

```bash
# 1. Fetch upstream tags
git fetch origin --tags

# 2. Rebase fork commits onto new tag
git checkout fork/openclaw-attribution
git rebase --onto server/v0.14.0 server/v0.13.2

# 3. Resolve conflicts — most likely in:
#    - models.py (if upstream adds fields near ours)
#    - long_term_memory.py (if merge/compact logic changes)
#    - memory_vector_db.py (if index schema changes)
#    - filters.py (if filter types change)

# 4. Run tests
source .venv/bin/activate
uv run pytest tests/test_attribution.py tests/test_forgetting.py -v

# 5. Update this doc with new base tag
```

**Conflict risk assessment**: Low-to-moderate. Our changes are additive (new fields, new pipeline steps). The highest-risk file is `long_term_memory.py` where we modify `merge_memories_with_llm` and `compact_long_term_memories` — these are also actively developed upstream.

## Running Tests

```bash
source .venv/bin/activate

# Fork-specific tests only
uv run pytest tests/test_attribution.py tests/test_forgetting.py -v

# Full suite (requires Redis running)
uv run pytest

# Full suite including API key-dependent tests
uv run pytest --run-api-tests
```

## Relationship to OpenClaw

This fork is consumed by the OpenClaw gateway's `openclaw-redis-agent-memory` plugin, which runs on the Mac Mini M4 Pro. The plugin sets attribution fields on every memory write based on the family registry (`~/.openclaw/family.json`) and identity resolver (`src/family.ts`).

The production memory server at `localhost:8000` runs from `~/Developer/homelab/openclaw/agent-memory-server/` (the deployed copy), not from this fork directory. After rebasing/updating this fork, the changes need to be deployed there.
