"""Tests for reference-record write protection — trust-rank mutation guard (C1).

Ported (single-tenant collapse) from wfr-memory-commons content_trust.py.
Covers the pure trust module, the storage round-trip of the ``trust_level``
field, write-time stamping at the API create endpoint, and the supersede/delete
mutation guards — including the dormant-by-default behavior (no operator token
configured ⇒ guard never fires).

No Redis required: storage is exercised via ``_memory_to_data`` /
``_data_to_memory_result``, and the guards via mocked ``get_long_term_memory_by_id``
and DB, mirroring tests/test_provenance_versioning.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.config import settings
from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase
from agent_memory_server.models import MemoryRecord
from agent_memory_server.utils.content_trust import (
    LOWEST_TRUST_LEVEL,
    ReferenceProtectedError,
    TrustLevel,
    derive_trust_level,
    is_operator_token,
    is_reference_protected_mutation,
    parse_trust_level,
    record_trust_level,
    trust_rank,
)
from tests.conftest import MockEmbeddings


def _db() -> RedisVLMemoryVectorDatabase:
    return RedisVLMemoryVectorDatabase(MagicMock(), MockEmbeddings())


# --- pure module: tiers, ranks, derivation ---------------------------------


def test_two_tiers_only():
    assert set(TrustLevel) == {TrustLevel.AGENT, TrustLevel.OPERATOR}


def test_lowest_trust_is_agent():
    assert LOWEST_TRUST_LEVEL is TrustLevel.AGENT


def test_trust_rank_is_total_and_ordered():
    # Total over every tier (a newly added tier would KeyError here).
    ranks = {level: trust_rank(level) for level in TrustLevel}
    assert ranks[TrustLevel.AGENT] < ranks[TrustLevel.OPERATOR]


def test_derive_trust_level_maps_operator_signal():
    assert derive_trust_level(is_operator=True) is TrustLevel.OPERATOR
    assert derive_trust_level(is_operator=False) is TrustLevel.AGENT


# --- operator token check (constant-time, fail-safe) -----------------------


def test_operator_token_exact_match():
    assert is_operator_token("s3cret", "s3cret") is True


def test_operator_token_mismatch():
    assert is_operator_token("wrong", "s3cret") is False


def test_operator_token_dormant_when_unconfigured():
    # No configured token ⇒ nobody is ever operator (the dormant default).
    assert is_operator_token("anything", None) is False
    assert is_operator_token("anything", "") is False


def test_operator_token_missing_header():
    assert is_operator_token(None, "s3cret") is False
    assert is_operator_token("", "s3cret") is False


# --- parse / read fail-safe ------------------------------------------------


def test_parse_trust_level_recognized():
    assert parse_trust_level("operator") is TrustLevel.OPERATOR
    assert parse_trust_level("agent") is TrustLevel.AGENT
    assert parse_trust_level(TrustLevel.OPERATOR) is TrustLevel.OPERATOR


def test_parse_trust_level_failsafe():
    for bad in (None, "garbage", 123, ["operator"], {}):
        assert parse_trust_level(bad) is LOWEST_TRUST_LEVEL


def test_record_trust_level_dict_and_object():
    assert record_trust_level({"trust_level": "operator"}) is TrustLevel.OPERATOR

    class R:
        trust_level = "operator"

    assert record_trust_level(R()) is TrustLevel.OPERATOR


def test_record_trust_level_legacy_records_are_lowest():
    # A record predating the field (no trust_level) must read as the lowest tier
    # so the gate never over-blocks a legacy record.
    assert record_trust_level({}) is LOWEST_TRUST_LEVEL

    class Legacy:
        pass

    assert record_trust_level(Legacy()) is LOWEST_TRUST_LEVEL


# --- the gate predicate ----------------------------------------------------


def test_agent_blocked_from_operator_record():
    assert (
        is_reference_protected_mutation(
            caller=TrustLevel.AGENT, record={"trust_level": "operator"}
        )
        is True
    )


def test_operator_can_mutate_operator_record():
    assert (
        is_reference_protected_mutation(
            caller=TrustLevel.OPERATOR, record={"trust_level": "operator"}
        )
        is False
    )


def test_equal_rank_allowed():
    assert (
        is_reference_protected_mutation(
            caller=TrustLevel.AGENT, record={"trust_level": "agent"}
        )
        is False
    )


def test_operator_can_mutate_agent_record():
    assert (
        is_reference_protected_mutation(
            caller=TrustLevel.OPERATOR, record={"trust_level": "agent"}
        )
        is False
    )


def test_legacy_record_not_protected():
    assert is_reference_protected_mutation(caller=TrustLevel.AGENT, record={}) is False


def test_reference_protected_error_carries_context():
    e = ReferenceProtectedError(["a", "b"], TrustLevel.AGENT)
    assert e.blocked_ids == ["a", "b"]
    assert e.caller == "agent"


# --- model field + storage round-trip --------------------------------------


def test_model_default_trust_level_none():
    assert MemoryRecord(id="x", text="t").trust_level is None


def test_memory_to_data_writes_trust_level():
    db = _db()
    data = db._memory_to_data(
        MemoryRecord(id="o1", text="canonical", trust_level="operator")
    )
    assert data["trust_level"] == "operator"


def test_memory_to_data_unset_trust_level_is_empty():
    db = _db()
    data = db._memory_to_data(MemoryRecord(id="a1", text="agent write"))
    assert data["trust_level"] == ""  # empty = unset = agent tier on read


def test_data_to_memory_result_parses_trust_level():
    db = _db()
    r = db._data_to_memory_result(
        {"id_": "o1", "text": "t", "trust_level": "operator"}, score=0.0
    )
    assert r.trust_level == "operator"


def test_data_to_memory_result_empty_trust_reads_as_none():
    db = _db()
    r = db._data_to_memory_result(
        {"id_": "a1", "text": "t", "trust_level": ""}, score=0.0
    )
    assert r.trust_level is None


def test_trust_level_in_return_fields_not_in_index_schema():
    # Round-trips via FT.SEARCH RETURN (stored hash field) without an index
    # rebuild — must be a return field but NOT an indexed field.
    from agent_memory_server.memory_vector_db_factory import _build_redis_schema

    assert "trust_level" in RedisVLMemoryVectorDatabase.RETURN_FIELDS
    indexed = {f["name"] for f in _build_redis_schema()["fields"]}
    assert "trust_level" not in indexed


# --- _reference_protection_active gating -----------------------------------


@pytest.fixture
def _operator_token(monkeypatch):
    monkeypatch.setattr(settings, "memory_operator_token", "op-secret")
    monkeypatch.setattr(settings, "memory_reference_protection_enabled", True)


def test_protection_inactive_when_no_token(monkeypatch):
    monkeypatch.setattr(settings, "memory_operator_token", None)
    monkeypatch.setattr(settings, "memory_reference_protection_enabled", True)
    assert ltm._reference_protection_active(TrustLevel.AGENT) is False


def test_protection_inactive_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "memory_operator_token", "op-secret")
    monkeypatch.setattr(settings, "memory_reference_protection_enabled", False)
    assert ltm._reference_protection_active(TrustLevel.AGENT) is False


def test_protection_inactive_for_none_caller(_operator_token):
    # Internal/library calls (no caller tier) are trusted — guard skipped.
    assert ltm._reference_protection_active(None) is False


def test_protection_inactive_for_operator_caller(_operator_token):
    # Operator outranks everything; the per-record fetch would be wasted.
    assert ltm._reference_protection_active(TrustLevel.OPERATOR) is False


def test_protection_active_for_agent_caller_with_token(_operator_token):
    assert ltm._reference_protection_active(TrustLevel.AGENT) is True


# --- delete guard ----------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_blocks_agent_on_operator_record(_operator_token):
    protected = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=1)
    with (
        patch.object(
            ltm, "get_long_term_memory_by_id", AsyncMock(return_value=protected)
        ),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
        pytest.raises(ReferenceProtectedError) as ei,
    ):
        await ltm.delete_long_term_memories(["o1"], caller_trust_level=TrustLevel.AGENT)
    assert ei.value.blocked_ids == ["o1"]
    # Atomic refuse: nothing deleted.
    fake_db.delete_memories.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_atomic_refuse_mixed_batch(_operator_token):
    protected = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    agent_rec = MemoryRecord(id="a1", text="agent")
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=2)
    get = AsyncMock(side_effect=[agent_rec, protected])  # a1 ok, o1 protected
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
        pytest.raises(ReferenceProtectedError) as ei,
    ):
        await ltm.delete_long_term_memories(
            ["a1", "o1"], caller_trust_level=TrustLevel.AGENT
        )
    assert ei.value.blocked_ids == ["o1"]
    fake_db.delete_memories.assert_not_awaited()  # whole batch refused


@pytest.mark.asyncio
async def test_delete_allows_agent_on_agent_record(_operator_token):
    agent_rec = MemoryRecord(id="a1", text="agent")
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=1)
    with (
        patch.object(
            ltm, "get_long_term_memory_by_id", AsyncMock(return_value=agent_rec)
        ),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        count = await ltm.delete_long_term_memories(
            ["a1"], caller_trust_level=TrustLevel.AGENT
        )
    assert count == 1
    fake_db.delete_memories.assert_awaited_once_with(["a1"])


@pytest.mark.asyncio
async def test_delete_operator_caller_bypasses_fetch(_operator_token):
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=1)
    get = AsyncMock()
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        count = await ltm.delete_long_term_memories(
            ["o1"], caller_trust_level=TrustLevel.OPERATOR
        )
    assert count == 1
    get.assert_not_awaited()  # operator outranks all — no per-record fetch


@pytest.mark.asyncio
async def test_delete_dormant_no_guard(monkeypatch):
    monkeypatch.setattr(settings, "memory_operator_token", None)
    operator_rec = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=1)
    get = AsyncMock(return_value=operator_rec)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        # Even an AGENT caller deletes an operator record when dormant.
        count = await ltm.delete_long_term_memories(
            ["o1"], caller_trust_level=TrustLevel.AGENT
        )
    assert count == 1
    get.assert_not_awaited()  # dormant ⇒ no fetch overhead


@pytest.mark.asyncio
async def test_delete_missing_target_not_blocked(_operator_token):
    fake_db = MagicMock()
    fake_db.delete_memories = AsyncMock(return_value=0)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", AsyncMock(return_value=None)),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        count = await ltm.delete_long_term_memories(
            ["gone"], caller_trust_level=TrustLevel.AGENT
        )
    assert count == 0  # missing target is a delete no-op, never a 403


# --- supersede guard -------------------------------------------------------


@pytest.mark.asyncio
async def test_supersede_blocks_agent_on_operator_record(_operator_token):
    target = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    repl = MemoryRecord(id="r1", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, rec = await ltm.supersede_memory(
            "o1", "r1", caller_trust_level=TrustLevel.AGENT
        )
    assert status == ltm.SUPERSEDE_PROTECTED
    assert rec.id == "o1"
    fake_db.update_memories.assert_not_awaited()  # not mutated


@pytest.mark.asyncio
async def test_supersede_target_missing_404_not_403(_operator_token):
    # A missing target must 404 (target_missing), not 403 — the protection check
    # runs only after the target is fetched.
    with patch.object(ltm, "get_long_term_memory_by_id", AsyncMock(return_value=None)):
        status, _ = await ltm.supersede_memory(
            "gone", "r1", caller_trust_level=TrustLevel.AGENT
        )
    assert status == ltm.SUPERSEDE_TARGET_MISSING


@pytest.mark.asyncio
async def test_supersede_operator_caller_allowed(_operator_token):
    target = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    repl = MemoryRecord(id="r1", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, rec = await ltm.supersede_memory(
            "o1", "r1", caller_trust_level=TrustLevel.OPERATOR
        )
    assert status == ltm.SUPERSEDE_OK
    assert rec.superseded_by == "r1"


@pytest.mark.asyncio
async def test_supersede_agent_on_agent_record_allowed(_operator_token):
    target = MemoryRecord(id="a1", text="old")  # no trust_level = agent
    repl = MemoryRecord(id="r1", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, _ = await ltm.supersede_memory(
            "a1", "r1", caller_trust_level=TrustLevel.AGENT
        )
    assert status == ltm.SUPERSEDE_OK


@pytest.mark.asyncio
async def test_supersede_dormant_allows_agent_on_operator(monkeypatch):
    monkeypatch.setattr(settings, "memory_operator_token", None)
    target = MemoryRecord(id="o1", text="canonical", trust_level="operator")
    repl = MemoryRecord(id="r1", text="new")
    get = AsyncMock(side_effect=[target, repl])
    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    with (
        patch.object(ltm, "get_long_term_memory_by_id", get),
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=fake_db)),
    ):
        status, _ = await ltm.supersede_memory(
            "o1", "r1", caller_trust_level=TrustLevel.AGENT
        )
    assert status == ltm.SUPERSEDE_OK  # dormant ⇒ guard never fires
