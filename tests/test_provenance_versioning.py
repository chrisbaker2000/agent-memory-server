"""Tests for provenance & versioning (ported from wfr-memory-commons).

Covers the additive MemoryRecord fields (kind, confidence, derived_from,
observed_at, valid_from, valid_to, superseded_by), their Redis round-trip
serialization (sentinels for null), the Kind/MinConfidence filters, write-funnel
temporal stamping, supersede-hiding in recall, and the supersede_memory state
machine.

Pure + mock-based — no Redis/network/LLM. Run:
  uv run pytest tests/test_provenance_versioning.py -v
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.config import settings
from agent_memory_server.extraction import (
    VALID_MEMORY_KINDS,
    coerce_extracted_kind,
)
from agent_memory_server.filters import Kind, MinConfidence
from agent_memory_server.memory_strategies import DiscreteMemoryStrategy
from agent_memory_server.memory_vector_db import (
    CONFIDENCE_UNSCORED_SENTINEL,
    VALID_TO_SENTINEL,
    RedisVLMemoryVectorDatabase,
)
from agent_memory_server.memory_vector_db_factory import _build_redis_schema
from agent_memory_server.models import MemoryRecord, MemoryRecordResult
from tests.conftest import MockEmbeddings


def _db() -> RedisVLMemoryVectorDatabase:
    return RedisVLMemoryVectorDatabase(MagicMock(), MockEmbeddings())


# --- model defaults --------------------------------------------------------


def test_new_fields_default_none():
    m = MemoryRecord(id="x", text="t")
    assert m.kind is None
    assert m.confidence is None
    assert m.derived_from is None
    assert m.observed_at is None
    assert m.valid_from is None
    assert m.valid_to is None
    assert m.superseded_by is None


def test_confidence_bounds_enforced():
    with pytest.raises(ValueError):
        MemoryRecord(id="x", text="t", confidence=1.5)
    with pytest.raises(ValueError):
        MemoryRecord(id="x", text="t", confidence=-0.1)
    assert MemoryRecord(id="x", text="t", confidence=0.0).confidence == 0.0
    assert MemoryRecord(id="x", text="t", confidence=1.0).confidence == 1.0


def test_kind_literal_enforced():
    with pytest.raises(ValueError):
        MemoryRecord(id="x", text="t", kind="bogus")
    assert MemoryRecord(id="x", text="t", kind="preference").kind == "preference"


# --- index schema ----------------------------------------------------------


def test_schema_indexes_new_fields():
    names = {f["name"]: f["type"] for f in _build_redis_schema()["fields"]}
    assert names["kind"] == "tag"
    assert names["superseded_by"] == "tag"
    assert names["confidence_idx"] == "numeric"
    assert names["valid_to_ts"] == "numeric"
    # store-and-return fields must NOT be indexed
    assert "derived_from" not in names
    assert "observed_at" not in names
    assert "valid_from" not in names


# --- serialization round-trip ----------------------------------------------


def test_memory_to_data_writes_provenance():
    db = _db()
    now = datetime(2026, 6, 18, 12, 0, 0, tzinfo=UTC)
    m = MemoryRecord(
        id="p1",
        text="Chris prefers 2-space indent",
        kind="preference",
        confidence=0.8,
        derived_from=["a", "b"],
        observed_at=now,
        valid_from=now,
    )
    data = db._memory_to_data(m)
    assert data["kind"] == "preference"
    assert data["confidence_idx"] == 0.8
    assert data["derived_from"] == "a|b"
    assert data["superseded_by"] == ""
    assert data["valid_to_ts"] == VALID_TO_SENTINEL  # open-ended
    assert "valid_to" not in data  # ISO companion only when bounded
    assert data["observed_at"] == now.timestamp()
    assert data["valid_from"] == now.timestamp()


def test_memory_to_data_unscored_uses_sentinel():
    db = _db()
    data = db._memory_to_data(MemoryRecord(id="p2", text="t"))  # no confidence
    assert data["confidence_idx"] == CONFIDENCE_UNSCORED_SENTINEL
    assert data["kind"] == ""  # unset tag


def test_memory_to_data_bounded_valid_to():
    db = _db()
    vt = datetime(2026, 1, 1, tzinfo=UTC)
    data = db._memory_to_data(MemoryRecord(id="p3", text="t", valid_to=vt))
    assert data["valid_to_ts"] == vt.timestamp()
    assert data["valid_to"] == vt.timestamp()


def test_data_to_memory_result_parses_provenance():
    db = _db()
    now_ts = datetime(2026, 6, 18, tzinfo=UTC).timestamp()
    fields = {
        "id_": "p1",
        "text": "t",
        "created_at": str(now_ts),
        "last_accessed": str(now_ts),
        "updated_at": str(now_ts),
        "kind": "fact",
        "confidence_idx": "0.9",
        "derived_from": "a|b",
        "observed_at": str(now_ts),
        "valid_from": str(now_ts),
        "superseded_by": "newer-id",
    }
    r = db._data_to_memory_result(fields, score=0.1)
    assert r.kind == "fact"
    assert r.confidence == 0.9
    assert r.derived_from == ["a", "b"]
    assert r.observed_at is not None
    assert r.valid_from is not None
    assert r.superseded_by == "newer-id"


def test_data_to_memory_result_unscored_sentinel_reads_as_none():
    db = _db()
    fields = {
        "id_": "p2",
        "text": "t",
        "confidence_idx": str(CONFIDENCE_UNSCORED_SENTINEL),
    }
    r = db._data_to_memory_result(fields, score=0.1)
    assert r.confidence is None  # sentinel must NOT read back as a real score


def test_data_to_memory_result_empty_tags_read_as_none():
    db = _db()
    fields = {"id_": "p3", "text": "t", "kind": "", "superseded_by": ""}
    r = db._data_to_memory_result(fields, score=0.1)
    assert r.kind is None
    assert r.superseded_by is None


# --- filters ---------------------------------------------------------------


def test_kind_filter_builds_tag():
    assert "kind" in str(Kind(eq="fact").to_filter())


def test_min_confidence_filter_inclusive_floor():
    expr = str(MinConfidence(gte=0.7).to_filter())
    assert "confidence_idx" in expr
    # sentinel (2.0) for unscored records always exceeds any [0,1] floor
    assert CONFIDENCE_UNSCORED_SENTINEL > 1.0


# --- write-funnel temporal stamping ----------------------------------------


class _CaptureDB:
    def __init__(self):
        self.indexed: list[MemoryRecord] = []

    async def add_memories(self, memories):
        self.indexed.extend(memories)
        return [m.id for m in memories]


class _NoopBG:
    def add_task(self, *a, **k):
        return None


def _patch_funnel(db):
    return (
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)),
        patch.object(ltm, "get_background_tasks", lambda: _NoopBG()),
    )


@pytest.mark.asyncio
async def test_funnel_stamps_valid_from_and_observed_at():
    db = _CaptureDB()
    rec = MemoryRecord(id="s1", text="Chris likes coffee")
    p_db, p_bg = _patch_funnel(db)
    with p_db, p_bg:
        await ltm.index_long_term_memories([rec], deduplicate=False)
    stored = db.indexed[0]
    assert stored.valid_from == stored.created_at
    assert stored.observed_at == stored.created_at
    assert stored.valid_to is None  # still valid
    assert stored.superseded_by is None


@pytest.mark.asyncio
async def test_funnel_preserves_supplied_observed_at():
    db = _CaptureDB()
    obs = datetime(2025, 1, 1, tzinfo=UTC)
    rec = MemoryRecord(id="s2", text="An old fact", observed_at=obs)
    p_db, p_bg = _patch_funnel(db)
    with p_db, p_bg:
        await ltm.index_long_term_memories([rec], deduplicate=False)
    assert db.indexed[0].observed_at == obs


# --- supersede-hiding in recall --------------------------------------------


class _SearchDB:
    """Returns a fixed result set from search_memories.

    Captures the `limit` it was called with (codex F2 over-fetch lock).
    """

    def __init__(self, results, next_offset=None, raw_total=None):
        self._results = results
        self._next_offset = next_offset
        # raw_total simulates a backend total LARGER than the returned page (the
        # pre-filter over-fetched count) so a test can prove the as_of post-filter
        # recomputes total. Defaults to len(results).
        self._raw_total = raw_total if raw_total is not None else len(results)
        self.last_limit = None
        self.last_ssr = "unset"

    async def search_memories(self, *a, **k):
        from agent_memory_server.models import MemoryRecordResults

        self.last_limit = k.get("limit")
        self.last_ssr = k.get("server_side_recency")
        return MemoryRecordResults(
            total=self._raw_total,
            memories=list(self._results),
            next_offset=self._next_offset,
        )

    async def list_memories(self, *a, **k):
        from agent_memory_server.models import MemoryRecordResults

        # Mirror search_memories so the filter-only (empty-text) recall path is
        # exercisable — used by the as_of filter-only test (codex F2).
        self.last_limit = k.get("limit")
        return MemoryRecordResults(
            total=self._raw_total, memories=list(self._results), next_offset=None
        )


def _result(id_, text, superseded_by=None):
    base = MemoryRecord(id=id_, text=text, superseded_by=superseded_by)
    return MemoryRecordResult(**base.model_dump(), dist=0.1)


@pytest.mark.asyncio
async def test_recall_hides_superseded_by_default():
    db = _SearchDB(
        [
            _result("a", "current fact"),
            _result("b", "old fact", superseded_by="a"),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="fact", limit=10)
    ids = {m.id for m in res.memories}
    assert ids == {"a"}
    assert res.total == 1


@pytest.mark.asyncio
async def test_recall_include_superseded_surfaces_all():
    db = _SearchDB(
        [
            _result("a", "current fact"),
            _result("b", "old fact", superseded_by="a"),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(
            text="fact", limit=10, include_superseded=True
        )
    assert {m.id for m in res.memories} == {"a", "b"}


@pytest.mark.asyncio
async def test_recall_keeps_records_missing_superseded_field():
    # Old records (no superseded_by on hash → None) must never be hidden.
    db = _SearchDB([_result("legacy", "pre-versioning memory")])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="memory", limit=10)
    assert {m.id for m in res.memories} == {"legacy"}


# --- as_of time-travel recall (LAB-405) ------------------------------------


def _windowed(id_, text, valid_from=None, valid_to=None, superseded_by=None):
    """A search result with an explicit validity window."""
    base = MemoryRecord(
        id=id_,
        text=text,
        valid_from=valid_from,
        valid_to=valid_to,
        superseded_by=superseded_by,
    )
    return MemoryRecordResult(**base.model_dump(), dist=0.1)


_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 3, 1, tzinfo=UTC)
_T2 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_as_of_excludes_not_yet_valid_records():
    # A record whose validity STARTS after as_of must be excluded.
    db = _SearchDB(
        [
            _windowed("early", "valid early", valid_from=_T0),
            _windowed("future", "not valid yet", valid_from=_T2),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="valid", limit=10, as_of=_T1)
    assert {m.id for m in res.memories} == {"early"}
    assert res.total == 1


@pytest.mark.asyncio
async def test_as_of_filter_only_listing_applies_window():
    # codex F2: empty text → filter-only list_memories path, which returns BEFORE
    # the vector-path post-filter. The as_of window must still be applied there.
    db = _SearchDB(
        [
            _windowed("early", "valid early", valid_from=_T0),
            _windowed("future", "not valid yet", valid_from=_T2),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="", limit=10, as_of=_T1)
    assert {m.id for m in res.memories} == {"early"}
    assert res.total == 1


@pytest.mark.asyncio
async def test_as_of_includes_superseded_record_valid_then():
    # A record valid at as_of but superseded AFTERWARDS (valid_to after as_of)
    # must be included — as_of overrides the default superseded-hide.
    db = _SearchDB(
        [
            _windowed("new", "current value", valid_from=_T2),
            _windowed(
                "old",
                "old value",
                valid_from=_T0,
                valid_to=_T2,
                superseded_by="new",
            ),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="value", limit=10, as_of=_T1)
    # at _T1: "old" is valid (T0 <= T1 < T2); "new" not yet valid (valid_from T2).
    assert {m.id for m in res.memories} == {"old"}


@pytest.mark.asyncio
async def test_as_of_excludes_window_ended_at_or_before():
    # `valid_to <= as_of` (end-exclusive) → excluded.
    db = _SearchDB(
        [
            _windowed("ended", "ended exactly at as_of", valid_from=_T0, valid_to=_T1),
            _windowed("open", "still valid", valid_from=_T0),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="valid", limit=10, as_of=_T1)
    assert {m.id for m in res.memories} == {"open"}


@pytest.mark.asyncio
async def test_as_of_open_ended_record_always_currently_valid():
    # valid_to None (open) → always currently-valid at any as_of >= valid_from.
    db = _SearchDB([_windowed("open", "open record", valid_from=_T0)])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="open", limit=10, as_of=_T2)
    assert {m.id for m in res.memories} == {"open"}


@pytest.mark.asyncio
async def test_as_of_legacy_record_no_valid_from_has_no_lower_bound():
    # Legacy record (valid_from None, valid_to None) → always valid at any as_of.
    db = _SearchDB([_windowed("legacy", "pre-versioning")])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="legacy", limit=10, as_of=_T0)
    assert {m.id for m in res.memories} == {"legacy"}


@pytest.mark.asyncio
async def test_as_of_overfetches_db_limit():
    # codex F2: with as_of set, the DB is queried with an over-fetched limit so
    # the window post-filter has a larger candidate pool than the caller's limit.
    db = _SearchDB([_windowed("a", "x", valid_from=_T0)])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        await ltm.search_long_term_memories(text="x", limit=5, as_of=_T1)
    assert db.last_limit > 5
    assert db.last_limit == min(max(5, 5 * 5), 200)


@pytest.mark.asyncio
async def test_no_as_of_queries_exact_limit():
    # Default (no as_of) must NOT over-fetch — db limit == caller limit.
    db = _SearchDB([_windowed("a", "x")])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        await ltm.search_long_term_memories(text="x", limit=5)
    assert db.last_limit == 5


@pytest.mark.asyncio
async def test_as_of_truncates_overfetched_pool_to_limit():
    # An over-fetched pool with more valid-at-as_of records than `limit` is
    # truncated back to `limit` after the window filter (codex F2).
    db = _SearchDB(
        [
            _windowed("v1", "valid 1", valid_from=_T0),
            _windowed("v2", "valid 2", valid_from=_T0),
            _windowed("v3", "valid 3", valid_from=_T0),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="valid", limit=2, as_of=_T1)
    assert len(res.memories) == 2
    assert res.total == 2


@pytest.mark.asyncio
async def test_as_of_forces_off_server_side_recency():
    # codex F1: the SSR aggregation backend omits valid_from/valid_to, so as_of
    # must NOT use it — the function downgrades server_side_recency to None.
    db = _SearchDB([_windowed("a", "x", valid_from=_T0)])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        await ltm.search_long_term_memories(
            text="x", limit=5, as_of=_T1, server_side_recency=True
        )
    assert db.last_ssr is None


def test_utc_timestamp_treats_naive_as_utc():
    # codex F1: BOTH as_of and the stored bounds normalize through _utc_timestamp.
    naive = datetime(2026, 3, 1, 12, 0, 0)
    aware = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    assert ltm._utc_timestamp(naive) == ltm._utc_timestamp(aware) == aware.timestamp()


@pytest.mark.asyncio
async def test_as_of_naive_stored_valid_from_boundary():
    # A record with a NAIVE valid_from must be compared as UTC (not host-local),
    # so an as_of exactly at that wall-clock instant is start-inclusive (codex F1).
    naive_from = datetime(2026, 3, 1, 0, 0, 0)  # _T1 wall clock, no tz
    db = _SearchDB([_windowed("a", "x", valid_from=naive_from)])
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=5, as_of=_T1)
    assert {m.id for m in res.memories} == {"a"}


def test_as_of_timestamp_treats_naive_as_utc():
    # codex F1: a timezone-less as_of must be interpreted as UTC, not host-local,
    # so it matches the UTC-stored validity windows at the boundary.
    naive = datetime(2026, 3, 1, 12, 0, 0)
    aware = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    assert ltm._as_of_timestamp(naive) == ltm._as_of_timestamp(aware)
    assert ltm._as_of_timestamp(aware) == aware.timestamp()


@pytest.mark.asyncio
async def test_as_of_naive_boundary_matches_utc_window():
    # A naive as_of exactly at a UTC valid_from is included (start-inclusive),
    # proving naive→UTC normalization (host-local would shift the boundary).
    db = _SearchDB([_windowed("a", "x", valid_from=_T1)])
    naive_t1 = datetime(2026, 3, 1, tzinfo=None)  # == _T1 wall clock, no tz
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=5, as_of=naive_t1)
    assert {m.id for m in res.memories} == {"a"}


@pytest.mark.asyncio
async def test_as_of_recall_is_single_page_next_offset_none():
    # codex F2: the over-fetch + window post-filter breaks the raw next_offset
    # mapping, so as_of recall nulls next_offset (single-page) rather than skip or
    # re-scan valid rows across pages.
    db = _SearchDB([_windowed("a", "x", valid_from=_T0)], next_offset=99)
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=5, as_of=_T1)
    assert res.next_offset is None


@pytest.mark.asyncio
async def test_as_of_total_matches_window_result_all_valid():
    # codex F1 follow-up: when the window filter drops nothing, total must still
    # equal the returned (windowed/truncated) count — not the raw over-fetched
    # DB total. Mock returns total=99 but only the page is windowed.
    # raw_total=99 simulates an inflated backend total; the window drops nothing,
    # so total must be recomputed to the returned count (2), not left at 99.
    db = _SearchDB(
        [_windowed("a", "x", valid_from=_T0), _windowed("b", "y", valid_from=_T0)],
        raw_total=99,
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=5, as_of=_T1)
    assert res.total == len(res.memories) == 2


@pytest.mark.asyncio
async def test_as_of_empty_page_still_nulls_next_offset():
    # codex follow-up: if an earlier filter empties the page before the as_of block,
    # next_offset must still be nulled (never leak a stale DB cursor under as_of).
    db = _SearchDB([], next_offset=42)
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=5, as_of=_T1)
    assert res.memories == []
    assert res.next_offset is None


@pytest.mark.asyncio
async def test_no_as_of_recall_unchanged_still_hides_superseded():
    # Default (no as_of): byte-for-byte unchanged — superseded-hide still applies.
    db = _SearchDB(
        [
            _windowed("a", "current"),
            _windowed("b", "old", superseded_by="a"),
        ]
    )
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        res = await ltm.search_long_term_memories(text="x", limit=10)
    assert {m.id for m in res.memories} == {"a"}


# --- supersede_memory state machine ----------------------------------------


@pytest.mark.asyncio
async def test_supersede_self_rejected():
    status, rec = await ltm.supersede_memory("x", "x")
    assert status == ltm.SUPERSEDE_SELF
    assert rec is None


@pytest.mark.asyncio
async def test_supersede_target_missing():
    with patch.object(ltm, "get_long_term_memory_by_id", AsyncMock(return_value=None)):
        status, rec = await ltm.supersede_memory("missing", "repl")
    assert status == ltm.SUPERSEDE_TARGET_MISSING


@pytest.mark.asyncio
async def test_supersede_replacement_missing():
    target = MemoryRecord(id="t", text="old")
    get = AsyncMock(side_effect=[target, None])  # target found, replacement not
    with patch.object(ltm, "get_long_term_memory_by_id", get):
        status, rec = await ltm.supersede_memory("t", "repl")
    assert status == ltm.SUPERSEDE_REPLACEMENT_MISSING


@pytest.mark.asyncio
async def test_supersede_success_marks_and_persists():
    target = MemoryRecord(id="t", text="old")
    repl = MemoryRecord(id="r", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, rec = await ltm.supersede_memory("t", "r")
    assert status == ltm.SUPERSEDE_OK
    assert rec.superseded_by == "r"
    assert rec.valid_to is not None
    fake_db.update_memories.assert_awaited_once()


@pytest.mark.asyncio
async def test_supersede_idempotent_same_replacement():
    target = MemoryRecord(id="t", text="old", superseded_by="r")
    get = AsyncMock(side_effect=[target, MemoryRecord(id="r", text="new")])
    with patch.object(ltm, "get_long_term_memory_by_id", get):
        status, rec = await ltm.supersede_memory("t", "r")
    assert status == ltm.SUPERSEDE_IDEMPOTENT


@pytest.mark.asyncio
async def test_supersede_conflict_different_replacement():
    target = MemoryRecord(id="t", text="old", superseded_by="other")
    get = AsyncMock(side_effect=[target, MemoryRecord(id="r", text="new")])
    with patch.object(ltm, "get_long_term_memory_by_id", get):
        status, rec = await ltm.supersede_memory("t", "r")
    assert status == ltm.SUPERSEDE_CONFLICT
    assert rec.superseded_by == "other"  # unchanged


@pytest.mark.asyncio
async def test_supersede_force_overrides_conflict():
    target = MemoryRecord(id="t", text="old", superseded_by="other")
    repl = MemoryRecord(id="r", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, rec = await ltm.supersede_memory("t", "r", force=True)
    assert status == ltm.SUPERSEDE_OK
    assert rec.superseded_by == "r"


# --- LAB-397: populate kind/confidence on extraction -----------------------
# F0 (LAB-395) extended `kind` with epistemic opinion/belief and set extraction
# confidence to the model-paraphrase tier (0.7), distinct from first-hand
# (unscored) writes. These lock the schema + the extraction-stamping convention.


def test_kind_literal_accepts_epistemic_values():
    # The F0-added epistemic kinds must be accepted on MemoryRecord.
    assert (
        MemoryRecord(
            id="o", text="Christian thinks rowing is boring", kind="opinion"
        ).kind
        == "opinion"
    )
    assert (
        MemoryRecord(
            id="b", text="Chris believes the QNAP is unreliable", kind="belief"
        ).kind
        == "belief"
    )
    # And the original four still work.
    for k in ("fact", "event", "preference", "summary"):
        assert MemoryRecord(id=k, text="t", kind=k).kind == k


def test_valid_memory_kinds_matches_literal():
    # The coercion allowlist must stay in lockstep with the model Literal.
    assert {
        "fact",
        "event",
        "preference",
        "opinion",
        "belief",
        "summary",
    } == VALID_MEMORY_KINDS


def test_coerce_extracted_kind():
    assert coerce_extracted_kind("opinion") == "opinion"
    assert coerce_extracted_kind("belief") == "belief"
    assert coerce_extracted_kind("fact") == "fact"
    # Off-vocabulary / missing / wrong-type → None (read path treats None as 'fact').
    assert coerce_extracted_kind("bogus") is None
    assert coerce_extracted_kind(None) is None
    assert coerce_extracted_kind(123) is None
    assert coerce_extracted_kind("") is None
    # Unhashable LLM output must NOT raise (a bare `in frozenset` would TypeError).
    assert coerce_extracted_kind(["fact"]) is None
    assert coerce_extracted_kind({"kind": "fact"}) is None
    # Capitalized/padded LLM emissions normalize to the canonical lowercase value.
    assert coerce_extracted_kind("Opinion") == "opinion"
    assert coerce_extracted_kind("FACT ") == "fact"
    assert coerce_extracted_kind("  Belief  ") == "belief"


def test_extraction_confidence_default_is_scored():
    # Extraction is a model paraphrase → SCORED (not the unscored first-hand tier).
    assert settings.extraction_confidence == 0.7
    assert 0.0 < settings.extraction_confidence < 1.0


def test_extraction_confidence_bounded_at_config_boundary():
    # A mis-set env must fail LOUD at config load (the boundary), not deep in
    # extraction when stamped onto MemoryRecord.confidence (ge=0/le=1).
    from agent_memory_server.config import Settings

    with pytest.raises(ValueError):
        Settings(extraction_confidence=1.5)
    with pytest.raises(ValueError):
        Settings(extraction_confidence=-0.1)


def test_opinion_record_serializes_kind_tag_and_scored_confidence():
    # An extracted opinion (kind=opinion, confidence=0.7) must serialize a real
    # confidence_idx (NOT the unscored sentinel) and the kind TAG — so a
    # min_confidence floor and an @kind:{opinion} filter both work on it.
    db = _db()
    m = MemoryRecord(
        id="op1",
        text="Christian thinks rowing is boring",
        kind="opinion",
        confidence=settings.extraction_confidence,
    )
    data = db._memory_to_data(m)
    assert data["kind"] == "opinion"
    assert data["confidence_idx"] == 0.7
    assert data["confidence_idx"] != CONFIDENCE_UNSCORED_SENTINEL


def test_discrete_prompt_emits_kind():
    # The discrete extraction prompt must instruct the LLM to classify `kind`
    # (with the epistemic values) and carry it in the example objects, else the
    # live path would never populate the field.
    p = DiscreteMemoryStrategy.EXTRACTION_PROMPT
    assert "kind: str" in p
    assert '"opinion"' in p and '"belief"' in p
    assert '"kind": "preference"' in p  # example object carries kind
    # The episodic (time-anchored) example must be labeled kind="event", NOT "fact"
    # — an example contradicting the event-vs-fact rule would teach the LLM wrong.
    assert '"type": "episodic",\n                "kind": "event"' in p
    assert '"type": "episodic",\n                "kind": "fact"' not in p


@pytest.mark.asyncio
async def test_session_thread_extraction_stamps_kind_and_confidence():
    """The LIVE auto-capture path (extract_memories_from_session_thread, which
    hard-codes the discrete strategy) must stamp kind=coerce_extracted_kind(...)
    and confidence=settings.extraction_confidence onto each constructed record."""
    from types import SimpleNamespace

    wm = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="Christian thinks rowing is boring")
        ]
    )

    class _Strategy:
        async def extract_memories(self, text, source_user_name=None):
            return [
                {
                    "text": "Christian thinks rowing is boring",
                    "type": "semantic",
                    "kind": "opinion",
                },
                {
                    "text": "Christian Baker is a lightweight rower",
                    "type": "semantic",
                },  # no kind
                {"text": "bad kind ignored", "type": "semantic", "kind": "BOGUS"},
            ]

    with (
        patch(
            "agent_memory_server.working_memory.get_working_memory",
            AsyncMock(return_value=wm),
        ),
        patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=_Strategy(),
        ),
    ):
        records = await ltm.extract_memories_from_session_thread(
            session_id="test-session", source_user="chris"
        )

    assert len(records) == 3
    # Every extracted record is stamped with the scored extraction-confidence tier.
    assert all(r.confidence == settings.extraction_confidence for r in records)
    # kind: emitted opinion kept; missing → None (reads as fact); off-vocab → None.
    assert records[0].kind == "opinion"
    assert records[1].kind is None
    assert records[2].kind is None


@pytest.mark.asyncio
async def test_strategy_aware_extraction_stamps_kind_and_confidence():
    """extract_memories_with_strategy must stamp confidence + a kind on each
    record: the LLM-emitted kind when present (discrete), else the strategy's
    invariant default (summary→summary, preferences→preference)."""
    import agent_memory_server.extraction as ext

    captured: list = []

    async def _capture_index(memories, **kwargs):
        captured.extend(memories)

    class _Strategy:
        def __init__(self, out):
            self._out = out

        async def extract_memories(self, text, source_user_name=None):
            return self._out

    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)

    # A 'summary'-strategy parent whose extracted memory carries NO kind → the
    # invariant default ("summary") must be stamped (not left None).
    parent = MemoryRecord(
        id="msg1",
        text="a long conversation",
        memory_type="message",
        extraction_strategy="summary",
        extraction_strategy_config={},
        discrete_memory_extracted="f",
    )
    # The LLM emits a CONFLICTING kind="fact"; the invariant summary default must
    # OVERRIDE it (a summary is always a summary), not merely fill when absent.
    strategy = _Strategy(
        [{"text": "a concise summary", "type": "semantic", "kind": "fact"}]
    )

    with (
        patch(
            "agent_memory_server.memory_vector_db_factory.get_memory_vector_db",
            AsyncMock(return_value=fake_db),
        ),
        patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=strategy,
        ),
        patch(
            "agent_memory_server.long_term_memory.index_long_term_memories",
            _capture_index,
        ),
    ):
        await ext.extract_memories_with_strategy(memories=[parent], deduplicate=False)

    assert len(captured) == 1
    assert captured[0].kind == "summary"  # invariant strategy default applied
    assert captured[0].confidence == settings.extraction_confidence


def test_default_kind_for_strategy():
    from agent_memory_server.extraction import default_kind_for_strategy

    assert default_kind_for_strategy("summary") == "summary"
    assert default_kind_for_strategy("preferences") == "preference"
    # Discrete kinds vary per memory → no invariant default.
    assert default_kind_for_strategy("discrete") is None
    assert default_kind_for_strategy(None) is None
    assert default_kind_for_strategy("bogus") is None
