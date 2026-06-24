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
    """Returns a fixed result set from search_memories."""

    def __init__(self, results):
        self._results = results

    async def search_memories(self, *a, **k):
        from agent_memory_server.models import MemoryRecordResults

        return MemoryRecordResults(
            total=len(self._results), memories=list(self._results), next_offset=None
        )

    async def list_memories(self, *a, **k):
        from agent_memory_server.models import MemoryRecordResults

        return MemoryRecordResults(total=0, memories=[], next_offset=None)


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
