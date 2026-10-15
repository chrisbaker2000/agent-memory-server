"""Curator salience (`importance`) is stored and returned, not indexed.

The curator PATCHes ``{"importance": N}``. Until the edit model grew the
field, pydantic dropped the key and the score never reached Redis. The
field is an unindexed hash value (same rule as ``trust_level``): FT.SEARCH
RETURN reads it with no index rebuild.

No Redis. Storage goes through ``_memory_to_data`` / ``_data_to_memory_result``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.memory_vector_db import (
    RedisVLMemoryVectorDatabase,
    parse_stored_importance,
)
from agent_memory_server.models import EditMemoryRecordRequest, MemoryRecord
from tests.conftest import MockEmbeddings


def _db() -> RedisVLMemoryVectorDatabase:
    return RedisVLMemoryVectorDatabase(MagicMock(), MockEmbeddings())


def test_model_default_importance_is_none():
    assert MemoryRecord(id="x", text="t").importance is None


def test_model_accepts_bounds_and_rejects_out_of_range():
    assert MemoryRecord(id="x", text="t", importance=1).importance == 1
    assert MemoryRecord(id="x", text="t", importance=5).importance == 5
    with pytest.raises(ValidationError):
        MemoryRecord(id="x", text="t", importance=0)
    with pytest.raises(ValidationError):
        MemoryRecord(id="x", text="t", importance=6)


def test_edit_request_dumps_only_the_set_score():
    assert EditMemoryRecordRequest(importance=4).model_dump(exclude_unset=True) == {
        "importance": 4
    }
    assert EditMemoryRecordRequest().model_dump(exclude_unset=True) == {}
    with pytest.raises(ValidationError):
        EditMemoryRecordRequest(importance=0)


def test_parse_stored_importance_accepts_redis_strings():
    assert parse_stored_importance("4") == 4
    assert parse_stored_importance(" 5 ") == 5
    assert parse_stored_importance(1) == 1
    assert parse_stored_importance(4.0) == 4


@pytest.mark.parametrize(
    "raw",
    ["", "0", "6", "9", "4.5", "nope", None, True, False, 0, 6, 4.5],
)
def test_parse_stored_importance_rejects_invalid(raw):
    assert parse_stored_importance(raw) is None


def test_memory_to_data_writes_score_and_omits_unset():
    db = _db()
    scored = db._memory_to_data(MemoryRecord(id="s", text="t", importance=4))
    assert scored["importance"] == 4
    unset = db._memory_to_data(MemoryRecord(id="u", text="t"))
    assert "importance" not in unset


def test_data_to_memory_result_round_trip():
    db = _db()
    scored = db._data_to_memory_result(
        {"id_": "s", "text": "t", "importance": "4"}, score=0.0
    )
    assert scored.importance == 4
    missing = db._data_to_memory_result({"id_": "m", "text": "t"}, score=0.0)
    assert missing.importance is None
    corrupt = db._data_to_memory_result(
        {"id_": "c", "text": "t", "importance": "9"}, score=0.0
    )
    assert corrupt.importance is None


def test_importance_is_returned_and_not_indexed():
    from agent_memory_server.memory_vector_db_factory import _build_redis_schema
    from agent_memory_server.utils.redis_query import RecencyAggregationQuery

    assert "importance" in RedisVLMemoryVectorDatabase.RETURN_FIELDS
    assert "importance" in RecencyAggregationQuery.DEFAULT_RETURN_FIELDS
    indexed = {field["name"] for field in _build_redis_schema()["fields"]}
    assert "importance" not in indexed


async def test_update_persists_importance(monkeypatch):
    existing = MemoryRecord(id="m1", text="a fact")

    async def fake_get(memory_id: str) -> MemoryRecord:
        assert memory_id == "m1"
        return existing

    db = MagicMock()
    db.update_memories = AsyncMock()

    async def fake_db():
        return db

    monkeypatch.setattr(ltm, "get_long_term_memory_by_id", fake_get)
    monkeypatch.setattr(ltm, "get_memory_vector_db", fake_db)

    updated = await ltm.update_long_term_memory("m1", {"importance": 4})

    assert updated is not None
    assert updated.importance == 4
    stored = db.update_memories.await_args.args[0][0]
    assert stored.importance == 4
