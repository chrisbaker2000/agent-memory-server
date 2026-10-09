"""The conftest MockMemoryVectorDatabase must accept what the real one accepts.

LAB-397 added `kind` / `min_confidence` (and earlier `hybrid_search`) to
MemoryVectorDatabase.search_memories, but not to the mock, so every API / MCP /
no-worker test that searched through the mock failed with "got an unexpected
keyword argument 'kind'" (4 tests red on fork/openclaw-attribution until
2026-10-09). This locks the public method signatures together.
"""

import inspect

import pytest

from agent_memory_server.memory_vector_db import MemoryVectorDatabase
from tests.conftest import MockMemoryVectorDatabase


def _public_methods(cls):
    return {
        name: fn
        for name, fn in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


@pytest.mark.parametrize("name", sorted(_public_methods(MockMemoryVectorDatabase)))
def test_mock_overrides_accept_every_real_parameter(name):
    real = _public_methods(MemoryVectorDatabase).get(name)
    mock = getattr(MockMemoryVectorDatabase, name)
    if real is None or mock is real:
        pytest.skip("inherited or mock-only")
    real_params = set(inspect.signature(real).parameters)
    mock_sig = inspect.signature(mock)
    if any(p.kind is p.VAR_KEYWORD for p in mock_sig.parameters.values()):
        return
    missing = real_params - set(mock_sig.parameters)
    assert not missing, f"MockMemoryVectorDatabase.{name} lacks {sorted(missing)}"


@pytest.mark.asyncio
async def test_mock_honours_kind_and_min_confidence():
    from agent_memory_server.filters import Kind, MinConfidence
    from agent_memory_server.models import MemoryRecord

    db = MockMemoryVectorDatabase()
    await db.add_memories(
        [
            MemoryRecord(id="f1", text="a fact", kind="fact", confidence=0.9),
            MemoryRecord(id="f2", text="weak fact", kind="fact", confidence=0.2),
            MemoryRecord(id="e1", text="an event", kind="event"),
        ]
    )
    facts = await db.search_memories("x", kind=Kind(eq="fact"))
    assert sorted(m.id for m in facts.memories) == ["f1", "f2"]
    floored = await db.search_memories("x", min_confidence=MinConfidence(gte=0.5))
    # Unscored (confidence None) always passes the floor.
    assert sorted(m.id for m in floored.memories) == ["e1", "f1"]
