# MEMORY-MODEL.md — Epistemic + Source-Trust Schema Decision (F0 / LAB-395)

> **Status: DRAFT — awaiting Chris's sign-off.** This note ships *no behavioral
> code change* on its own. Several deep-dive tickets (LAB-397 populate
> kind/confidence, LAB-403 author-trust tier, LAB-399 supersession) implement
> against the decisions recorded here. Nothing downstream proceeds until this is
> approved.

This is the single agreed convention for the provenance/versioning fields added
in FORK.md #16 — specifically the two that are **100% unpopulated** today
(`kind`, `confidence`) plus the dormant `trust_level`. As of 2026-06-24 the live
corpus is 17,041 records; `FT.SEARCH @kind:{fact|event|preference|summary}` → 0
and `@confidence_idx:[2 2]` → 17,041 (every record carries the unscored
sentinel). So there is no legacy convention to honor — we are defining it.

Reference fields: `agent_memory_server/models.py:353-470` (all provenance
fields). Filters: `agent_memory_server/filters.py:305` (`Kind`), `:313`
(`MinConfidence`). Sentinel: `agent_memory_server/memory_vector_db.py:472`.

---

## Decision 1 — `kind` taxonomy

**Today:** `kind: Literal["fact", "event", "preference", "summary"] | None`
(`models.py:353`). `None` is treated as `fact` on read. It is a **recall FILTER
only** — it does NOT affect dedup or ranking — and is indexed as a TAG (a TAG
field accepts new values without a reindex; only the Pydantic `Literal` gates
them).

**Proposed:** extend to add two epistemic-status kinds:

```python
kind: Literal[
    "fact", "event", "preference", "opinion", "belief", "summary"
] | None
```

This is **net-new for us** — wfr-memory-commons treats `kind` as an opaque
client-supplied wire field and does not classify epistemic type
(`.../further_memory/routes/memories.py:177`).

### Definitions + decision matrix

| `kind` | Definition | Worked example (from the corpus / realistic) |
|---|---|---|
| **fact** | An objectively-verifiable state of the world. Truth-apt, not the speaker's stance. | "Christian Baker is a lightweight rower." / "Chris Baker's birthday is 1980-12-08." |
| **event** | A time-anchored occurrence. SHOULD carry an `event_date`. | "Grant Baker had a therapy appointment at 10am (2026-02-25)." |
| **preference** | What a *named person* likes / dislikes / wants. | "Chris prefers espresso." / "Lindsey prefers window seats." |
| **opinion** | A subjective *evaluative stance* a named person holds about something external. Attributed to the holder; not truth-apt about the world. | "Christian thinks rowing is boring." |
| **belief** | A held conviction about how the world *is* that is contestable / not established. Distinguished from `fact` by epistemic status, not topic. | "Chris believes the QNAP is unreliable." |
| **summary** | A condensed multi-message digest (the SummaryMemoryStrategy output). | "Chris discussed the marketplace launch; decided React+Postgres; target March 2025." |

**Disambiguation rules (for extraction + reviewers):**
- Prefer **fact** when the statement is verifiable and stated as such, even about
  a third party. ("Christian is a lightweight rower" = fact, not opinion.)
- Use **opinion** when the text contains an evaluative verb tied to a holder
  (*thinks / feels / finds it / likes-as-judgment*) about an external subject.
- Use **belief** for *believes / is convinced / suspects / assumes* about a state
  of the world. The fuzzy boundary (opinion↔belief) is acceptable because BOTH
  are downstream-grouped as "epistemic, holder-attributed, not a hard fact" — the
  recall consumer (Goal 2) only needs fact-vs-(opinion|belief) separation, so a
  mislabel between the two is low-cost.
- **preference** outranks **opinion** when the stance is about the holder's own
  taste ("Chris likes espresso" = preference, not opinion).
- `None` continues to mean *unspecified* and is read as `fact` — so legacy
  records and any path that doesn't classify stay safely in the default lane.

**Risk / cost:** extending the `Literal` is a one-line type change, no reindex
(TAG). The only behavioral surface is that `min`-style fact-vs-opinion recall
filters become *possible* once a writer populates the field (LAB-397). Low risk.
**Recommendation: ADD `opinion` + `belief`.**

---

## Decision 2 — `confidence` convention

`confidence: float | None`, `ge=0.0, le=1.0` (`models.py:362`). **`None` =
UNSCORED**, deliberately distinct from a low score.

### Verified safety property (the load-bearing invariant)

`confidence_idx` is **always** written: the real score when scored, else
`CONFIDENCE_UNSCORED_SENTINEL = 2.0` (above the [0,1] range) so an UNSCORED
record passes *any* inclusive `min_confidence` floor:

```python
# agent_memory_server/memory_vector_db.py:470-476
"confidence_idx": (
    memory.confidence
    if memory.confidence is not None
    else CONFIDENCE_UNSCORED_SENTINEL          # 2.0 — NOT 0.0
),
```

✅ **Confirmed: `None` is NOT coerced to `0`.** A `min_confidence=0.8` recall
keeps every first-hand (unscored) capture. This is locked by
`tests/test_provenance_versioning.py` (sentinel/floor cases) and is the property
LAB-397 must not regress.

### Who scores, and how

| Source of the claim | `confidence` | Rationale |
|---|---|---|
| **First-hand user statement** (a person said it in their own DM) | `None` (UNSCORED) | Highest trust; do NOT write `1.0` — keep the "first-hand" signal distinct, and the sentinel guarantees it survives any floor. |
| **Model-extracted from conversation** (discrete/summary strategy) | `0.7` | A grounded extraction, but a paraphrase that could drift. |
| **Cross-session synthesized / inferred** (derived from other memories) | `0.6` | Second-order; carries `derived_from` lineage. |
| **Speculative / weak-evidence** ("maybe", "I think it might be") | `0.3` | Hedged claims; rank below grounded facts. |

Scoring is the **writer's** responsibility (the extraction strategy on the
auto-capture path; the `memory_store` tool caller on the explicit path). The
write funnel `index_long_term_memories` passes `confidence` through unmodified
and applies the sentinel only when it is `None`.

**Recommendation: adopt the table above; first-hand stays UNSCORED (`None`).**

---

## Decision 3 — `trust_level` derivation (single-tenant, `source_channel`-keyed)

`trust_level` (`models.py:410`) is **server-managed** — derived at write time,
never accepted from a client write payload, fail-safe to the lowest tier. Tiers
(ported from wfr-memory-commons `utils/content_trust.py:49-63`):
`system` > `first_party` > `agent` (ranks 2 > 1 > 0).

wfr-memory-commons derives the tier from `ConsumerId` (multi-tenant — **dropped**
for homelab). The single-tenant replacement keys off **`source_channel`** (and,
where needed, family-registry membership of `source_user`):

| `source_channel` (or writer) | tier | Why |
|---|---|---|
| memory-manager / memory-curator / memory-maintenance / migration scripts; operator-token writes | `system` | Canonical maintenance writers; not exposed to untrusted input. Outrank everything for supersede/delete. |
| `discord` / `slack` / `whatsapp` / `imessage` / `signal` DM from a **family.json** member (incl. Chris) | `first_party` | Authenticated human input relayed by the gateway — not raw third-party content. |
| `mailgun` / `webhook-ingress` (inbound email/webhook) | `agent` | Semi-trusted external content; treat as the injection surface. |
| subagent / `firecrawl` / web-tool / any tool-derived or inferred write | `agent` | Processes untrusted tool output (the indirect-injection surface). |
| unknown / unmapped `source_channel` | `agent` (fail-safe) | An unclassified writer is never trusted above the lowest tier. |

**Deliberate choice — Chris is NOT privileged above other family members at the
trust layer.** Both map to `first_party`; the *person* a fact is about is carried
by `source_user`/text (Decision: subject attribution, LAB-396), not by trust.
`system` is reserved for non-human maintenance writers so a migration can
supersede a canonical record while a human DM cannot clobber another's.

This mapping feeds LAB-403 (`derive_trust_level(source_channel)` +
`is_reference_protected_mutation` re-keyed from `ConsumerId` → `source_channel`).
Injection-flagged writes are floored to `agent` + `flagged_for_review` (the
injection-scan half is already ported — content-security).

**Recommendation: adopt the table; reserve `system` for maintenance/migration.**

---

## Decision 4 — Backfill policy for the existing 17,041 records

**Guiding constraint (hard invariant):** NO bulk auto-correction of existing
records without a human-review gate. Prior LLM *and* deterministic-heuristic
sweeps over-flagged (reassigned Chris's legitimate rowing facts; false-matched
"1\" female fitting") — see `memory-subject-misattribution-rootcause`.

| Field | Backfill policy |
|---|---|
| `kind` | **Leave NULL (no bulk backfill).** `None` reads as `fact`, so recall is byte-for-byte unchanged. Populate **forward only** (LAB-397). *If* a backfill is later wanted, run a **deterministic, reversible** category-prefix heuristic (`"Preference:"→preference`, `"Event:"→event`, `"Decision:"/"Lesson:"→fact`) that writes a **human-review queue**, never auto-applies, and the un-prefixed majority stays `fact`. |
| `confidence` | **Leave UNSCORED (`None`).** Never bulk-assign — assigning would destroy the first-hand signal and could silently drop records under a floor. Forward-only via LAB-397. |
| `trust_level` | **Leave NULL** → resolves to `agent` (lowest) via the parse fail-safe, which never over-blocks the reference gate. Forward-only on write (LAB-403). An optional deterministic `source_channel→tier` backfill is reversible and lower-risk than `kind`, but still goes through a review queue, not a blind sweep. |

Any migration that rewrites existing records MUST run with
`EMBEDDING_MODEL=ollama/nomic-embed-text` set (the 768-dim dimension-mismatch
incident orphaned all records once) and is a **gated step confirmed with Chris**,
not automatic.

**Recommendation: forward-only population; existing records untouched; any future
backfill is deterministic + reversible + human-review-queued.**

---

## Summary of recommendations (for sign-off)

1. **Extend `kind`** → add `opinion` + `belief` (6 values; `None`=`fact`).
2. **`confidence`** → first-hand UNSCORED (`None`); inferred 0.7; synthesized 0.6;
   speculative 0.3. Sentinel-preservation verified (`None`≠0).
3. **`trust_level`** → `source_channel`-keyed: maintenance/migration=`system`,
   family DM=`first_party`, webhook/subagent/tool=`agent`, unknown=`agent`.
4. **Backfill** → forward-only; existing 17k untouched; any future backfill
   deterministic + reversible + human-review-gated; reindex only with the
   nomic-embed-text env set.

> The `MemoryKind` `Literal` extension (Decision 1) is the only code change this
> note authorizes, and it is deferred into the LAB-397 PR (which populates the
> field) rather than shipped here, keeping F0 a pure-doc change.
