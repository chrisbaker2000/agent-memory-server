"""Integration tests for the `bypass_recall_filters` search param (LAB-281).

When the recall relevance gate is enabled globally, a trusted "list all" caller
(the curator full-corpus backup) must be able to read the entire corpus instead
of the salient-term subset. `bypass_recall_filters=True` skips the gate for that
single request. Mocks the vector DB so no Redis is required.

Run: uv run pytest tests/test_bypass_recall_filters.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.config import settings
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
)


def _rec(text, dist):
    base = MemoryRecord(id=f"id-{dist}", text=text)
    return MemoryRecordResult(**base.model_dump(), dist=dist)


def _results():
    # One result anchored to the query term ("memory"), one weak-tail result with
    # no shared salient term — the latter is what the gate drops.
    return MemoryRecordResults(
        total=2,
        next_offset=None,
        memories=[
            _rec("memory backup of all records", 0.12),
            _rec("the dog ate dinner", 0.45),  # no anchor, above floor → gate drops
        ],
    )


class _DB:
    async def search_memories(self, query, **kw):
        return _results()

    async def list_memories(self, **kw):
        return _results()


@pytest.mark.asyncio
async def test_gate_enabled_trims_weak_tail_without_bypass(monkeypatch):
    """Regression-lock the bug surface: with the gate ON and no bypass, the
    text-anchored 'list all' query is trimmed to the salient-term subset."""
    monkeypatch.setattr(settings, "recall_relevance_gate_enabled", True)
    monkeypatch.setattr(settings, "recall_relevance_gate_shadow", False)
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=_DB())):
        out = await ltm.search_long_term_memories(text="memory", limit=10)
    # Weak no-anchor record dropped → only the anchored one survives.
    assert out.total == 1
    assert [m.text for m in out.memories] == ["memory backup of all records"]


@pytest.mark.asyncio
async def test_bypass_keeps_full_corpus_with_gate_enabled(monkeypatch):
    """The fix: bypass_recall_filters=True skips the gate even when it is enabled
    globally, so a full-corpus backup reads every record."""
    monkeypatch.setattr(settings, "recall_relevance_gate_enabled", True)
    monkeypatch.setattr(settings, "recall_relevance_gate_shadow", False)
    with patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=_DB())):
        out = await ltm.search_long_term_memories(
            text="memory", limit=10, bypass_recall_filters=True
        )
    # Nothing trimmed — both records returned.
    assert out.total == 2
    assert {m.text for m in out.memories} == {
        "memory backup of all records",
        "the dog ate dinner",
    }
