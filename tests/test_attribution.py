"""Comprehensive tests for attribution fields on MemoryRecord.

Covers merge propagation, extraction inheritance, persistence round-trips,
and search filtering for source_user, source_channel, visibility, and stale_after.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from ulid import ULID

from agent_memory_server.extraction import _resolve_parent_attribution
from agent_memory_server.filters import SourceChannel, SourceUser, VisibilityFilter
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(**overrides) -> MemoryRecord:
    """Build a MemoryRecord with sensible defaults, allowing overrides."""
    defaults = {
        "id": str(ULID()),
        "text": "test memory",
        "memory_type": MemoryTypeEnum.SEMANTIC,
        "discrete_memory_extracted": "t",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ---------------------------------------------------------------------------
# Group 1: Merge attribution (6 tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMergeAttribution:
    """Tests for merge_memories_with_llm attribution field propagation."""

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_same_source_user(self, mock_llm_cls):
        """When all memories share the same source_user, the merged memory preserves it."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris", visibility="everyone")
        m2 = _make_memory(source_user="chris", visibility="everyone")

        merged = await merge_memories_with_llm([m1, m2])

        assert merged.source_user == "chris"

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_different_source_user(self, mock_llm_cls):
        """Merging memories with different source_user values raises ValueError."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris")
        m2 = _make_memory(source_user="lindsey")

        with pytest.raises(ValueError, match="different source_user"):
            await merge_memories_with_llm([m1, m2])

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_visibility_most_restrictive(self, mock_llm_cls):
        """Merged visibility is the most restrictive across inputs."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris", visibility="parents")
        m2 = _make_memory(source_user="chris", visibility="everyone")

        merged = await merge_memories_with_llm([m1, m2])

        assert merged.visibility == "parents"

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_stale_after_earliest(self, mock_llm_cls):
        """Merged stale_after takes the earliest (most conservative) datetime."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        early = datetime(2026, 6, 1, tzinfo=UTC)
        late = datetime(2026, 12, 1, tzinfo=UTC)

        m1 = _make_memory(source_user="chris", stale_after=early)
        m2 = _make_memory(source_user="chris", stale_after=late)

        merged = await merge_memories_with_llm([m1, m2])

        assert merged.stale_after == early

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_source_channel_first_wins(self, mock_llm_cls):
        """Merged source_channel takes the first non-None value."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris", source_channel="discord")
        m2 = _make_memory(source_user="chris", source_channel="slack")

        merged = await merge_memories_with_llm([m1, m2])

        assert merged.source_channel == "discord"

    @patch("agent_memory_server.long_term_memory.LLMClient")
    async def test_merge_missing_attribution(self, mock_llm_cls):
        """Merging memories that lack attribution fields uses safe defaults (getattr fallback)."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        # Build memories with default attribution (source_user=None, visibility="everyone")
        m1 = _make_memory()
        m2 = _make_memory()

        merged = await merge_memories_with_llm([m1, m2])

        assert merged.source_user is None
        assert merged.source_channel is None
        assert merged.visibility == "everyone"
        assert merged.stale_after is None


# ---------------------------------------------------------------------------
# Group 2: Extraction attribution (4 tests)
# ---------------------------------------------------------------------------

class TestExtractionAttribution:
    """Tests for _resolve_parent_attribution and extraction inheritance."""

    def test_extraction_inherits_parent_source_user(self):
        """Child memories inherit source_user from parent when not explicitly set."""
        parent = _make_memory(source_user="lindsey")

        resolved_user, _, _ = _resolve_parent_attribution([parent])

        assert resolved_user == "lindsey"

    def test_extraction_inherits_parent_visibility(self):
        """Child memories inherit most-restrictive visibility from parents."""
        p1 = _make_memory(visibility="family")
        p2 = _make_memory(visibility="admin")

        _, _, resolved_vis = _resolve_parent_attribution([p1, p2])

        assert resolved_vis == "admin"

    def test_extraction_explicit_params_override_parent(self):
        """Explicit source_user/source_channel/visibility override parent values."""
        parent = _make_memory(
            source_user="chris",
            source_channel="discord",
            visibility="admin",
        )

        resolved_user, resolved_channel, resolved_vis = _resolve_parent_attribution(
            [parent],
            source_user="lindsey",
            source_channel="slack",
            visibility="everyone",
        )

        assert resolved_user == "lindsey"
        assert resolved_channel == "slack"
        assert resolved_vis == "everyone"

    def test_extraction_no_parent_uses_defaults(self):
        """With no parents and no explicit params, defaults are safe."""
        resolved_user, resolved_channel, resolved_vis = _resolve_parent_attribution([])

        assert resolved_user is None
        assert resolved_channel is None
        assert resolved_vis == "everyone"


# ---------------------------------------------------------------------------
# Group 3: Persistence round-trip (3 tests)
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    """Tests for MemoryRecord → Redis data dict → MemoryRecordResult serialization."""

    def _make_db_instance(self):
        """Create a RedisVLMemoryVectorDatabase with mock index/embeddings for unit tests."""
        from unittest.mock import MagicMock

        from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase

        mock_index = MagicMock()
        mock_embeddings = MagicMock()
        return RedisVLMemoryVectorDatabase(index=mock_index, embeddings=mock_embeddings)

    def test_round_trip_all_attribution_fields(self):
        """All 4 attribution fields survive serialize → deserialize round-trip."""
        db = self._make_db_instance()
        stale_dt = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)

        memory = _make_memory(
            source_user="chris",
            source_channel="discord",
            visibility="parents",
            stale_after=stale_dt,
        )

        # Serialize: MemoryRecord → data dict (as stored in Redis)
        data = db._memory_to_data(memory)

        assert data["source_user"] == "chris"
        assert data["source_channel"] == "discord"
        assert data["visibility"] == "parents"
        assert data["stale_after"] == pytest.approx(stale_dt.timestamp(), abs=1)

        # Deserialize: data dict → MemoryRecordResult
        result = db._data_to_memory_result(data, score=0.0)

        assert result.source_user == "chris"
        assert result.source_channel == "discord"
        assert result.visibility == "parents"
        assert result.stale_after is not None
        assert abs(result.stale_after.timestamp() - stale_dt.timestamp()) < 1

    def test_round_trip_missing_fields_get_defaults(self):
        """Data without attribution fields deserializes with safe defaults."""
        db = self._make_db_instance()

        # Simulate legacy Redis data without attribution fields
        data = {
            "text": "old memory",
            "id_": str(ULID()),
            "session_id": "",
            "user_id": "",
            "namespace": "",
            "memory_type": "semantic",
            "topics": "",
            "entities": "",
            "memory_hash": "",
            "discrete_memory_extracted": "t",
            "pinned": 0,
            "access_count": 0,
            "extracted_from": "",
            "created_at": datetime.now(UTC).timestamp(),
            "last_accessed": datetime.now(UTC).timestamp(),
            "updated_at": datetime.now(UTC).timestamp(),
            # No source_user, source_channel, visibility, stale_after
        }

        result = db._data_to_memory_result(data, score=0.0)

        assert result.source_user is None
        assert result.source_channel is None
        assert result.visibility == "everyone"
        assert result.stale_after is None

    def test_round_trip_stale_after_timestamp_parsing(self):
        """stale_after survives float timestamp serialization/deserialization."""
        db = self._make_db_instance()

        stale_dt = datetime(2027, 1, 15, 8, 30, 0, tzinfo=UTC)
        memory = _make_memory(stale_after=stale_dt)

        data = db._memory_to_data(memory)

        # Verify it's stored as a float timestamp
        assert isinstance(data["stale_after"], float)
        assert data["stale_after"] == pytest.approx(stale_dt.timestamp(), abs=1)

        result = db._data_to_memory_result(data, score=0.0)

        assert result.stale_after is not None
        assert result.stale_after.tzinfo is not None  # timezone-aware
        assert abs(result.stale_after.timestamp() - stale_dt.timestamp()) < 1


# ---------------------------------------------------------------------------
# Group 4: Search filtering (3 tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSearchFiltering:
    """Tests for attribution-based search filtering via MockMemoryVectorDatabase."""

    def _build_mock_db(self):
        """Build a MockMemoryVectorDatabase with seeded test memories."""
        from tests.conftest import MockMemoryVectorDatabase

        db = MockMemoryVectorDatabase()

        # Seed memories with different attribution profiles
        db.memories["m1"] = _make_memory(
            id="m1",
            text="Chris's note",
            source_user="chris",
            source_channel="discord",
            visibility="everyone",
        )
        db.memories["m2"] = _make_memory(
            id="m2",
            text="Lindsey's note",
            source_user="lindsey",
            source_channel="slack",
            visibility="family",
        )
        db.memories["m3"] = _make_memory(
            id="m3",
            text="Admin-only note",
            source_user="chris",
            source_channel="slack",
            visibility="admin",
        )

        return db

    async def test_search_filter_source_user(self):
        """Filtering by source_user returns only memories from that user."""
        db = self._build_mock_db()

        results = await db.search_memories(
            query="note",
            source_user=SourceUser(eq="chris"),
        )

        # Only m1 and m3 are from chris
        returned_ids = {m.id for m in results.memories}
        assert returned_ids == {"m1", "m3"}
        assert results.total == 2

    async def test_search_filter_visibility(self):
        """Filtering by visibility returns only matching memories."""
        db = self._build_mock_db()

        results = await db.search_memories(
            query="note",
            visibility=VisibilityFilter(eq="family"),
        )

        returned_ids = {m.id for m in results.memories}
        assert returned_ids == {"m2"}
        assert results.total == 1

    async def test_search_filter_combined(self):
        """Combined source_user + source_channel filter narrows correctly."""
        db = self._build_mock_db()

        results = await db.search_memories(
            query="note",
            source_user=SourceUser(eq="chris"),
            source_channel=SourceChannel(eq="slack"),
        )

        # Only m3 matches chris + slack
        returned_ids = {m.id for m in results.memories}
        assert returned_ids == {"m3"}
        assert results.total == 1
