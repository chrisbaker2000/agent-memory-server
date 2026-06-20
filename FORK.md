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

7. **Forgetting pipeline** (`long_term_memory.py`) — `select_ids_for_forgetting()` checks `stale_after` and deletes expired memories (pinned exempt). Configurable via `stale_after_cleanup_enabled`. **Deployment note (2026-06-20):** the *deletion* path runs only via `periodic_forget_long_term_memories`, which is registered ONLY when Docket is enabled (`docket_tasks.py` early-returns when `use_docket=False`). This deployment runs `USE_DOCKET=false`, so stale_after is **flag-only** here: the nightly `memory-maintenance.py` `phase_stale_flag` tags past-due records with `stale_flagged` for operator review and never deletes them — consistent with the standing "never auto-destroy memories" principle (Docket was disabled after it destroyed ~2,700 memories in March 2026). This is intentional, not dormant; do not wire auto-deletion without explicit sign-off.

8. **Conflict detection** (`long_term_memory.py`, `api.py`) — `detect_similar_memories()` searches without merging. `POST /v1/long-term-memory/?detect_conflicts=true` returns similar memories for caller-driven dedup.

9. **Safety defaults** (`config.py`) — Semantic dedup disabled (`semantic_dedup_enabled=False`, `deduplication_distance_threshold=0.0`). Automatic compaction disabled (`compaction_every_minutes=0`). Hash-based dedup remains active. `generation_model`, `slow_model`, and `fast_model` defaults changed from bare `"gpt-5"`/`"gpt-5-mini"` (which LiteLLM routes to paid OpenAI) to `"azure/gpt-5.4"`/`"azure/gpt-5-mini"` so the fallback stays on free Azure if env vars fail to export. Deep review 2026-04-16 H2.

10. **Telemetry** (`telemetry.py`, `main.py`, `llm/embeddings.py`) — OTLP HTTP metrics to SigNoz (localhost:4318). Instruments embedding calls, search operations, memory store. 30s flush interval, non-blocking (HTTP happens outside the buffer lock). Uses `httpx` (declared dependency).

11. **Nomic embedding prefixes** (`llm/embeddings.py`) — `aembed_documents()` prepends "search_document: " and `aembed_query()` prepends "search_query: " for nomic-embed-text models. No-op for other models.

12. **Recency tuning** — `semantic_weight` 0.8→0.9, `recency_weight` 0.2→0.1, `half_life_last_access_days` 7→14. Applied consistently across api.py, recency.py, redis_query.py.

13. **Dedup fix** (`long_term_memory.py`) — Fixed variable shadowing bug in `deduplicate_by_semantic_search`.

14. **Search-query clamp + recall relevance gate** (`long_term_memory.py`, `config.py`, `utils/relevance.py`) — Ported from wfr-finley (SOC2 OpenClaw). `search_long_term_memories` clamps the query to `max_search_query_chars` (2000, always on) before embedding so an oversized recall degrades gracefully. A deterministic, LLM-free relevance gate (`utils/relevance.py`) trims the weak-distance tail of results that share no salient whole-word term with the query — but ships **DEFAULT-OFF** (`recall_relevance_gate_enabled=False`) with a shadow mode (`recall_relevance_gate_shadow`) to protect this deployment's conceptual recall; strong semantic matches (`dist <= recall_relevance_gate_distance_floor`, 0.25) are always kept. Term matching is by tokenization, never substring (no "ben"↔"Bennett" leak). Tests: `tests/test_relevance_gate.py` (21), `tests/test_search_query_clamp.py` (3).

15. **Write-time content security** (`utils/content_security.py`, `long_term_memory.py`, `config.py`) — Ported from **wfr-memory-commons** (`further-memory`, the SOC2 Further memory service) — the load-bearing, pure-Python halves of its `utils/dlp.py` (secret redaction) and `utils/content_trust.py` (Unicode sanitization + prompt-injection flagging). The multi-tenant machinery (per-consumer `ConsumerId` trust tiers, visibility-band `DataClassification`, per-consumer egress DLP) was deliberately NOT ported — this homelab is single-tenant. Three deterministic, dependency-free controls run in the universal write funnel `index_long_term_memories` (so every write path is covered), each independently config-gated:
    - **`sanitize_memory_text`** — NFC-normalize + strip C0/C1/DEL + format (Cf, zero-width/bidi/BOM) chars. Removal-only; defends against obfuscated-injection storage and log spoofing. Default ON (`memory_text_sanitization_enabled`).
    - **`redact_secrets`** — scrub known credential shapes (Anthropic/OpenAI/Stripe/GCP/AWS/GitHub/Slack keys + tokens, PEM private-key blocks) to `[REDACTED:<label>]` before the text is embedded or stored. A stored secret is a RAG re-injection/exfil target; redact-not-reject keeps the surrounding memory useful. Most→least-specific ordering so the narrowest label wins. Logs the secret *type* (never the value) + emits `memory_server.content_security.secrets_redacted` to SigNoz. Default ON (`memory_secret_redaction_enabled`).
    - **`scan_for_injection`** — flag (never reject/mutate) instruction-shaped text (instruction-override, wrapper-escape, tool-directive, exfiltration/destruction). Flag-and-keep: WARNING log + `memory_server.content_security.injection_flagged` counter so the operator can review. Default ON (`memory_injection_scan_enabled`).

    Tests: `tests/test_content_security.py` (28 — pure-function + funnel-integration, mocked vector DB, no Redis).

16. **Provenance & versioning** (`models.py`, `filters.py`, `memory_vector_db.py`, `memory_vector_db_factory.py`, `long_term_memory.py`, `api.py`, `scripts/migrate_provenance_versioning.py`) — Ported from **wfr-memory-commons** (`further-memory`). Adds seven additive `MemoryRecord` fields and a non-destructive supersede flow — the "link, don't merge" alternative to the LLM merge this fork forbids. The multi-tenant trust/PDP/attribution-chain cluster from further-memory was deliberately NOT ported.
    - **Fields**: `kind` (Literal fact/event/preference/summary), `confidence` (0–1, None = unscored), `derived_from` (provenance memory-id list), `observed_at`, `valid_from`, `valid_to`, `superseded_by`. All default None (additive; legacy records unaffected).
    - **Indexed** (TAG `kind`, `superseded_by`; NUMERIC `confidence_idx`, `valid_to_ts`) with null-sentinels: `CONFIDENCE_UNSCORED_SENTINEL=2.0` (unscored always passes a `min_confidence` floor) and `VALID_TO_SENTINEL=9_999_999_999` (open-ended). `derived_from`/`observed_at`/`valid_from` are store-and-return only.
    - **Write funnel** stamps `valid_from`/`observed_at` = `created_at` when omitted.
    - **Recall filters**: `Kind` + `MinConfidence` (opt-in, threaded through `SearchRequest`/`search_long_term_memories`). **Require the index rebuild** to function (migration below).
    - **Supersede**: `supersede_memory(target, replacement)` + `POST /v1/long-term-memory/{id}/supersede` (statuses superseded/idempotent/conflict/target_missing/replacement_missing/self; `force` to override, 409 on conflict). Superseded records are **hidden from default recall** via a post-filter on `superseded_by` (NOT a query predicate — so records written before the field existed are correctly kept). `SearchRequest.include_superseded=True` surfaces them.
    - **Key risk-control**: `FT.SEARCH RETURN` reads hash fields regardless of indexing, so storage + field round-trip + supersede-hiding work IMMEDIATELY on upgrade. Only the `kind`/`min_confidence` query filters need the index rebuilt. The rebuild + sentinel backfill over the ~16.8k live records is a DELIBERATE, dry-run-default, count-safeguarded migration (`scripts/migrate_provenance_versioning.py --execute`), not auto-run.
    - Tests: `tests/test_provenance_versioning.py` (24 — pure model/serialization/filter + funnel + supersede state machine, no Redis).

17. **Reference-record write protection — trust-rank mutation guard** (`utils/content_trust.py`, `models.py`, `memory_vector_db.py`, `config.py`, `long_term_memory.py`, `api.py`, `mcp.py`) — Ported from **wfr-memory-commons** `utils/content_trust.py` (FIN-466 / M-4). A lower-trust caller must never be able to **supersede or delete** a higher-trust (canonical / identity / reference) record. The audit (C1) claimed Pat "tags `trust_level` on write" — it does **not**; this port introduces the entire trust dimension. Defense-in-depth: the gateway already gates the agent's `memory_forget` tool (`safety-contract` plugin) and `supersede` is not an agent tool at all — this is the server-side backstop that survives a gateway bypass and keys on a signal the agent cannot forge.
    - **Single-tenant collapse**: Finley's per-`ConsumerId` three-tier model (system / first_party / agent) → two tiers **`OPERATOR`** vs **`AGENT`**, derived from the only agent-unforgeable signal under `auth_mode=disabled`: presence of a configured operator shared-secret (`X-Operator-Token`) on the mutating request. The multi-tenant `ConsumerId`/`DataClassification`/PDP machinery was NOT ported.
    - **`trust_level`** field on `MemoryRecord` (default None = lowest 'agent' tier). **SERVER-MANAGED**: stamped at the API create endpoint from the operator-token-derived caller tier, overriding any client value (the agent path can't mint an OPERATOR write). Excluded from PATCH `updatable_fields`. Stored on the hash + in `RETURN_FIELDS` but **NOT indexed** → no rebuild needed (same risk-control as #16).
    - **Gate**: `is_reference_protected_mutation(caller, record)` = `trust_rank(caller) < trust_rank(record)`. Wired into `supersede_memory` (→ `SUPERSEDE_PROTECTED` → 403) and `delete_long_term_memories` (atomic refuse: raises `ReferenceProtectedError` → 403, deletes nothing if ANY target out-ranks the caller). Legacy/unclassified records read as 'agent' (fail-safe) so the gate never over-blocks them; a missing target stays a 404/no-op, never a false 403.
    - **DORMANT BY DEFAULT** (`memory_operator_token=None`): no write is ever stamped OPERATOR, the per-record fetch on delete is skipped, and the guard can never fire — zero behavior change until an operator opts in by setting `MEMORY_OPERATOR_TOKEN` and having operator scripts send `X-Operator-Token`. `memory_reference_protection_enabled` (default True) is the kill switch even with a token set. Blocked mutations emit `memory_server.reference_protection.blocked` to SigNoz.
    - **MCP** is the agent surface (no token) — create/delete pass `TrustLevel.AGENT` explicitly (the route's `Depends()` default is unresolved on a direct call).
    - Tests: `tests/test_content_trust.py` (40 — pure tier/rank/derivation/token/fail-safe + storage round-trip + protection-active gating + delete/supersede guards incl. dormant + atomic-refuse + 404-not-403, no Redis).

18. **Embedding-order verification** (`utils/embed_verify.py`, `memory_vector_db.py`, `config.py`) — Ported from **wfr-memory-commons** `storage/embed_verify.py` (FIN-19 / FIN-295). `add_memories` batch-embeds (`aembed_documents([m.text …])`) then `zip(memories, embeddings, strict=True)` — the strict zip catches a length mismatch but NOT a silent *reorder* by the provider/LiteLLM/Ollama (wrong vector on wrong record, invisible at the boundary, corrupts recall for every affected record). A sample-based cosine probe re-embeds up to 3 leading texts individually and compares against the batched vectors; unrelated texts have cosine ~0, the 0.999 floor tolerates float rounding.
    - **Audit-premise correction**: C2 said "run unconditionally in the bulk re-embed/migration scripts" — but this fork has **no** bulk re-embed path (the provenance migration #16 reuses stored vectors; `rebuild-index` only recreates the schema). The one real batch-embed surface is the live `add_memories` write, which is where the probe is wired.
    - **Gated** by `memory_verify_embed_order` (default **OFF** — the hot path doesn't pay ~3 extra embeds per multi-record batch unless an operator opts in, e.g. to audit a provider). Runs **only for genuine multi-record batches** (`len(memories) > 1`; a batch of one cannot be reordered — a strict improvement over the reference, which samples even size-1).
    - **Fail-closed on confirmed reorder, fail-open on inability-to-verify**: `verify_embedding_order` raises `EmbeddingOrderError` only on a cosine-floor breach (→ propagates out of `add_memories`, the corrupted batch is never persisted, emits `memory_server.store.errors{error=EmbeddingOrderError}`); a transient single-embed failure or length disagreement returns `False` (logged, batch persisted unverified).
    - **Prefix caveat** (FIN-632): the single re-embed uses the **document** task path (`aembed_documents([text])[0]`) to match the batched call — nomic applies a different prefix to query vs document text, so a query-path re-embed would false-positive below the floor.
    - Tests: `tests/test_embed_verify.py` (16 — pure cosine, verifier pass/False/raise contract incl. reverse + rotate-by-one + float-rounding tolerance, and add_memories wiring: off-skips, single-not-checked, correct-order-persists, reorder-fails-closed, no Redis).

### Test Coverage

- `tests/test_provenance_versioning.py` — 24 tests: field defaults/validation, index-schema membership, serialization round-trip (sentinels for null), Kind/MinConfidence filters, write-funnel temporal stamping, supersede-hiding in recall (incl. legacy-record-kept), and the supersede_memory state machine (self/missing/idempotent/conflict/force).
- `tests/test_content_security.py` — 28 tests: sanitization (zero-width/bidi/NFC/control strip), secret redaction (label ordering, idempotency, multi-secret), injection flagging, orchestrator gating, and write-funnel wiring (redact-before-store, flag-and-keep, original-object-not-mutated).
- `tests/test_content_trust.py` — 40 tests: trust tiers/ranks/derivation, constant-time operator-token check (dormant + fail-safe), parse/read fail-safe (legacy records → lowest), the gate predicate, `trust_level` storage round-trip (return-field not indexed), `_reference_protection_active` gating, and the delete/supersede guards (block/allow, operator bypass, dormant no-op, atomic batch refuse, 404-not-403).
- `tests/test_embed_verify.py` — 16 tests: pure cosine (identical/orthogonal/scaled/degenerate), verifier pass/best-effort-False/fail-closed-raise contract (reverse, rotate-by-one, float-rounding tolerance, sample-size cap), and add_memories wiring (off-skips, single-record-not-checked, correct-order-persists, reorder-fails-closed-unpersisted).
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
