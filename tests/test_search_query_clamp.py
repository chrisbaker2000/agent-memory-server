"""Integration test for the search-query length clamp in
search_long_term_memories. Mocks the vector DB so no Redis is required.

Run: uv run pytest tests/test_search_query_clamp.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.config import settings
from agent_memory_server.models import MemoryRecordResults


class _FakeDB:
    """Captures the query length that reaches the vector DB."""

    def __init__(self):
        self.last_query_len = None

    async def search_memories(self, query, **kw):
        self.last_query_len = len(query)
        return MemoryRecordResults(total=0, memories=[], next_offset=None)

    async def list_memories(self, **kw):
        return MemoryRecordResults(total=0, memories=[], next_offset=None)


@pytest.mark.asyncio
async def test_oversized_query_is_clamped_to_cap():
    cap = settings.max_search_query_chars
    assert cap == 2000  # regression-lock the default
    db = _FakeDB()
    big = "mortgage rate " * 215  # ~3010 chars, well over the cap
    assert len(big) > cap
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        await ltm.search_long_term_memories(text=big, limit=1)
    assert db.last_query_len == cap


@pytest.mark.asyncio
async def test_under_cap_query_is_not_clamped():
    db = _FakeDB()
    small = "what is the mortgage rate"
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=db)):
        await ltm.search_long_term_memories(text=small, limit=1)
    assert db.last_query_len == len(small)


@pytest.mark.asyncio
async def test_relevance_gate_off_by_default_keeps_all():
    """With the gate disabled (default), no results are dropped even when the
    weak tail has no lexical anchor — protects conceptual recall."""
    from agent_memory_server.models import MemoryRecord, MemoryRecordResult

    assert settings.recall_relevance_gate_enabled is False

    def _rec(text, dist):
        base = MemoryRecord(id=f"id-{dist}", text=text)
        return MemoryRecordResult(**base.model_dump(), dist=dist)

    results = MemoryRecordResults(
        total=2,
        next_offset=None,
        memories=[
            _rec("mortgage rate is 6.5", 0.12),
            _rec("the dog ate dinner", 0.45),  # weak, no anchor — would drop if gate ON
        ],
    )

    class _DB:
        async def search_memories(self, query, **kw):
            return results

        async def list_memories(self, **kw):
            return results

    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=_DB())):
        out = await ltm.search_long_term_memories(text="mortgage rate", limit=10)
    # Gate is OFF → both kept.
    assert out.total == 2
