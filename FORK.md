# agent-memory-server-fork — Claude Code Context

## What This Is

A fork of [redis/agent-memory-server](https://github.com/redis/agent-memory-server) (`server/v0.13.2`) that adds **multi-user attribution**, **memory lifecycle** fields, **hybrid search**, **telemetry**, and **pipeline safety guards**. These changes support OpenClaw's family assistant use case, where a single memory server stores memories for multiple family members with per-user visibility controls.

**Upstream**: `origin` points to `https://github.com/redis/agent-memory-server.git`
**Fork branch**: `fork/openclaw-attribution` (branched from tag `server/v0.13.2`)
**Fork location**: `/Users/chris/Developer/homelab/openclaw/agent-memory-server-fork/`

## What Was Changed (12 commits)

### 4 New Fields on `MemoryRecord`

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `source_user` | `str \| None` | `None` | Username of the person who created/contributed this memory (e.g. `chris`, `lindsey`). Set at creation, propagated through merge and extraction. |
| `source_channel` | `str \| None` | `None` | Channel or integration where the memory originated (e.g. `discord`, `slack`, `whatsapp`). Enables per-channel recall filtering. |
| `visibility` | `str` | `"everyone"` | Access scope — who can see this memory. Values ranked by restrictiveness in `VISIBILITY_RANK` (models.py): `everyone` < `family` < `restricted` < `private` < `parents` < `admin`. Most-restrictive wins on merge. |
| `stale_after` | `datetime \| None` | `None` | Optional expiry hint. The forgetting pipeline deletes memories past this datetime (pinned memories exempt). |

### Changes by Layer

1. **Models** (`models.py`) — Added 4 fields to `MemoryRecord`, `SearchRequest`, `CreateMemoryRecordRequest`, and `EditMemoryRecordRequest`. `VISIBILITY_RANK` constant defines the single source of truth for visibility ordering. `StoreMemoryResponse` and `SimilarMemoryInfo` models for conflict detection.

2. **Filters** (`filters.py`) — Added `SourceUser`, `SourceChannel`, `VisibilityFilter` TAG filter types, `StaleAfter` NUMERIC filter for RedisVL query building.

3. **Vectorstore** (`memory_vector_db.py`, `memory_vector_db_factory.py`) — Persist the 4 fields as Redis hash fields (3 TAGs + 1 NUMERIC for `stale_after`). Parse them back on read. Index schema updated with 4 new fields. `RETURN_FIELDS` includes all attribution fields.

4. **Hybrid search** (`memory_vector_db.py`, `config.py`) — Runs both vector KNN and text BM25 searches in parallel, merges via Reciprocal Rank Fusion (RRF, k=60). Dramatically improves recall for short names, project titles, and proper nouns. Configurable via `hybrid_search_enabled`, `hybrid_search_rrf_k`, `hybrid_search_text_results_multiplier`.

5. **Merge pipeline** (`long_term_memory.py`) — `merge_memories_with_llm()` propagates attribution. Visibility uses `VISIBILITY_RANK` most-restrictive-wins. Refuses to merge memories with different `source_user` values. Size guards (`MAX_MEMORY_INPUT_CHARS=500`, `MAX_MEMORY_OUTPUT_CHARS=1000`, `MAX_ENTITY_COUNT=30`) prevent mega-memory creation.

6. **Extraction pipeline** (`extraction.py`) — `_resolve_parent_attribution()` copies parent's attribution to child memories. `clean_entities()` quality-filters entities (stop words, URLs, hex IDs, variant dedup). `enforce_topics()` enforces controlled vocabulary with synonym mapping and deterministic longest-first substring matching. Vocabulary loaded from `~/.openclaw/config/memory-vocabulary.json`.

7. **Forgetting pipeline** (`long_term_memory.py`) — `select_ids_for_forgetting()` checks `stale_after` and deletes expired memories (pinned exempt). Configurable via `stale_after_cleanup_enabled`.

8. **Conflict detection** (`long_term_memory.py`, `api.py`) — `detect_similar_memories()` searches without merging. `POST /v1/long-term-memory/?detect_conflicts=true` returns similar memories for caller-driven dedup.

9. **Safety defaults** (`config.py`) — Semantic dedup disabled (`semantic_dedup_enabled=False`, `deduplication_distance_threshold=0.0`). Automatic compaction disabled (`compaction_every_minutes=0`). Hash-based dedup remains active. `generation_model`, `slow_model`, and `fast_model` defaults changed from bare `"gpt-5"`/`"gpt-5-mini"` (which LiteLLM routes to paid OpenAI) to `"azure/gpt-5.4"`/`"azure/gpt-5-mini"` so the fallback stays on free Azure if env vars fail to export. Deep review 2026-04-16 H2.

10. **Telemetry** (`telemetry.py`, `main.py`, `llm/embeddings.py`) — OTLP HTTP metrics to SigNoz (localhost:4318). Instruments embedding calls, search operations, memory store. 30s flush interval, non-blocking (HTTP happens outside the buffer lock). Uses `httpx` (declared dependency).

11. **Nomic embedding prefixes** (`llm/embeddings.py`) — `aembed_documents()` prepends "search_document: " and `aembed_query()` prepends "search_query: " for nomic-embed-text models. No-op for other models.

12. **Recency tuning** — `semantic_weight` 0.8→0.9, `recency_weight` 0.2→0.1, `half_life_last_access_days` 7→14. Applied consistently across api.py, recency.py, redis_query.py.

13. **Dedup fix** (`long_term_memory.py`) — Fixed variable shadowing bug in `deduplicate_by_semantic_search`.

14. **Search-query clamp + recall relevance gate** (`long_term_memory.py`, `config.py`, `utils/relevance.py`) — Ported from wfr-finley (SOC2 OpenClaw). `search_long_term_memories` clamps the query to `max_search_query_chars` (2000, always on) before embedding so an oversized recall degrades gracefully. A deterministic, LLM-free relevance gate (`utils/relevance.py`) trims the weak-distance tail of results that share no salient whole-word term with the query — but ships **DEFAULT-OFF** (`recall_relevance_gate_enabled=False`) with a shadow mode (`recall_relevance_gate_shadow`) to protect this deployment's conceptual recall; strong semantic matches (`dist <= recall_relevance_gate_distance_floor`, 0.25) are always kept. Term matching is by tokenization, never substring (no "ben"↔"Bennett" leak). Tests: `tests/test_relevance_gate.py` (21), `tests/test_search_query_clamp.py` (3).

### Test Coverage

- `tests/test_attribution.py` — 29 tests: merge propagation, extraction inheritance, persistence round-trips, search filtering, API endpoints, MCP tools.
- `tests/test_forgetting.py` — 22 tests: stale_after expiry, pinned exemption, TTL/inactivity, budget pruning, `_parse_stale_after` parsing.
- `tests/test_memory_vector_db.py` — 41 tests: hybrid search (RRF merge, fallback, disable), text search, factory, embeddings, hash generation.
- `tests/test_review_fixes.py` — Tests for shared VISIBILITY_RANK, mock attribution propagation, RecencyAggregationQuery fields, telemetry platform detection.

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
#    - extraction.py (if topic/entity extraction changes)
#    - config.py (if settings change)

# 4. Run tests
source .venv/bin/activate
uv run pytest tests/test_attribution.py tests/test_forgetting.py tests/test_memory_vector_db.py tests/test_review_fixes.py -v

# 5. Update this doc with new base tag
```

**Conflict risk assessment**: Moderate. Our changes are mostly additive (new fields, new pipeline steps). Highest-risk files: `long_term_memory.py` (merge/compact modifications), `memory_vector_db.py` (hybrid search additions, index schema), `extraction.py` (vocabulary/quality filtering), `config.py` (new settings).

## Running Tests

```bash
source .venv/bin/activate

# Fork-specific tests only (fast, no Redis required)
uv run pytest tests/test_attribution.py tests/test_forgetting.py tests/test_memory_vector_db.py tests/test_review_fixes.py -v

# Full suite (requires Redis via docker-compose)
uv run pytest

# Full suite including API key-dependent tests
uv run pytest --run-api-tests
```

## Deployment

The production memory server at `localhost:8000` runs from `~/Developer/homelab/openclaw/agent-memory-server/` which has an editable install pointing back to this fork directory. Changes here are live after restarting:

```bash
launchctl kickstart -k gui/$(id -u)/ai.openclaw.memory-server
```

## Relationship to OpenClaw

This fork is consumed by the OpenClaw gateway's `openclaw-redis-agent-memory` plugin, which runs on the Mac Mini M4 Pro. The plugin sets attribution fields on every memory write based on the family registry (`~/.openclaw/family.json`) and identity resolver (`src/family.ts`).
