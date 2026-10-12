"""Importance is a stored recall weight, not a topic tag.

Scores were discarded from 2026-04-12 because the edit model had no field.
These tests lock the field: 1–5 or unset, round-trips through the hash codec,
comes back on search (RETURN, not an index field — no rebuild), and a PATCH
of only ``importance`` is a real update.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.memory_vector_db import (
    RedisVLMemoryVectorDatabase,
    _parse_importance,
)
from agent_memory_server.models import EditMemoryRecordRequest, MemoryRecord
from tests.conftest import MockEmbeddings


def _db() -> RedisVLMemoryVectorDatabase:
    return RedisVLMemoryVectorDatabase(MagicMock(), MockEmbeddings())


def test_default_is_unscored():
    assert MemoryRecord(id="x", text="t").importance is None
    assert EditMemoryRecordRequest().importance is None


@pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
def test_accepts_one_through_five(score: int):
    assert MemoryRecord(id="x", text="t", importance=score).importance == score
    assert EditMemoryRecordRequest(importance=score).importance == score


@pytest.mark.parametrize("score", [0, 6, -1, 3.5])
def test_rejects_out_of_range(score: float):
    with pytest.raises(ValidationError):
        MemoryRecord(id="x", text="t", importance=score)
    with pytest.raises(ValidationError):
        EditMemoryRecordRequest(importance=score)


def test_memory_to_data_writes_score_and_blank_when_unset():
    db = _db()
    scored = db._memory_to_data(MemoryRecord(id="a", text="t", importance=5))
    unset = db._memory_to_data(MemoryRecord(id="b", text="t"))
    assert scored["importance"] == 5
    assert unset["importance"] == ""


def test_data_to_memory_result_parses_score():
    db = _db()
    scored = db._data_to_memory_result(
        {"id_": "a", "text": "t", "importance": "5"}, score=0.0
    )
    assert scored.importance == 5


@pytest.mark.parametrize("raw", [None, "", "0", "6", "nope", "4.5", True, 9])
def test_garbage_and_missing_read_as_unscored(raw: object):
    assert _parse_importance(raw) is None
    db = _db()
    fields = {"id_": "a", "text": "t"}
    if raw is not None:
        fields["importance"] = raw
    assert db._data_to_memory_result(fields, score=0.0).importance is None


def test_importance_is_returned_but_not_indexed():
    # FT.SEARCH RETURN reads the hash field. Indexing it would require a
    # rebuild before any search that LOADs it, and recall only needs the value.
    from agent_memory_server.memory_vector_db_factory import _build_redis_schema

    assert "importance" in RedisVLMemoryVectorDatabase.RETURN_FIELDS
    indexed = {field["name"] for field in _build_redis_schema()["fields"]}
    assert "importance" not in indexed


async def test_update_persists_importance_alone():
    existing = MemoryRecord(id="m1", text="hello")
    db = MagicMock()
    db.update_memories = AsyncMock()
    with (
        patch.object(
            ltm, "get_long_term_memory_by_id", new=AsyncMock(return_value=existing)
        ),
        patch.object(ltm, "get_memory_vector_db", new=AsyncMock(return_value=db)),
    ):
        updated = await ltm.update_long_term_memory("m1", {"importance": 4})
    assert updated is not None
    assert updated.importance == 4
    assert updated.text == "hello"
    written = db.update_memories.await_args.args[0][0]
    assert written.importance == 4


async def test_update_still_rejects_unknown_fields():
    existing = MemoryRecord(id="m1", text="hello")
    with (
        patch.object(
            ltm, "get_long_term_memory_by_id", new=AsyncMock(return_value=existing)
        ),
        pytest.raises(ValueError, match="Cannot update fields"),
    ):
        await ltm.update_long_term_memory("m1", {"not_a_field": 1})
