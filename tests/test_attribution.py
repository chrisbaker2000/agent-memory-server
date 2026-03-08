"""Comprehensive tests for attribution fields on MemoryRecord.

Covers model creation, defaults, merge propagation, extraction inheritance,
persistence round-trips, search filtering, API endpoints, and MCP tools
for source_user, source_channel, visibility, and stale_after.
"""

from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from ulid import ULID

from agent_memory_server.extraction import _resolve_parent_attribution
from agent_memory_server.filters import SourceChannel, SourceUser, VisibilityFilter
from agent_memory_server.models import (
    EditMemoryRecordRequest,
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
    SearchRequest,
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


# ---------------------------------------------------------------------------
# Group 5: Model creation and defaults (6 tests)
# ---------------------------------------------------------------------------

class TestModelAttributionFields:
    """Tests for MemoryRecord attribution field creation, defaults, and serialization."""

    def test_memory_record_accepts_all_attribution_fields(self):
        """MemoryRecord constructor accepts all 4 attribution fields."""
        stale_dt = datetime(2026, 9, 1, tzinfo=UTC)
        m = MemoryRecord(
            id=str(ULID()),
            text="test",
            source_user="chris",
            source_channel="discord",
            visibility="admin",
            stale_after=stale_dt,
        )
        assert m.source_user == "chris"
        assert m.source_channel == "discord"
        assert m.visibility == "admin"
        assert m.stale_after == stale_dt

    def test_memory_record_defaults(self):
        """MemoryRecord defaults: source_user=None, source_channel=None, visibility='everyone', stale_after=None."""
        m = MemoryRecord(id=str(ULID()), text="test")
        assert m.source_user is None
        assert m.source_channel is None
        assert m.visibility == "everyone"
        assert m.stale_after is None

    def test_memory_record_model_dump_includes_attribution(self):
        """model_dump() output includes all 4 attribution fields."""
        m = _make_memory(
            source_user="lindsey",
            source_channel="slack",
            visibility="family",
            stale_after=datetime(2026, 12, 1, tzinfo=UTC),
        )
        d = m.model_dump()
        assert "source_user" in d
        assert "source_channel" in d
        assert "visibility" in d
        assert "stale_after" in d
        assert d["source_user"] == "lindsey"
        assert d["source_channel"] == "slack"
        assert d["visibility"] == "family"

    def test_memory_record_result_inherits_attribution(self):
        """MemoryRecordResult (subclass) exposes attribution fields."""
        now = datetime.now(UTC)
        stale_dt = datetime(2027, 1, 1, tzinfo=UTC)
        r = MemoryRecordResult(
            id="test-id",
            text="test",
            dist=0.1,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            user_id="u1",
            session_id=None,
            namespace="ns1",
            topics=[],
            entities=[],
            memory_hash="",
            memory_type="semantic",
            persisted_at=None,
            source_user="chris",
            source_channel="discord",
            visibility="parents",
            stale_after=stale_dt,
        )
        assert r.source_user == "chris"
        assert r.source_channel == "discord"
        assert r.visibility == "parents"
        assert r.stale_after == stale_dt

    def test_edit_memory_record_request_includes_attribution(self):
        """EditMemoryRecordRequest model accepts all 4 attribution fields."""
        req = EditMemoryRecordRequest(
            source_user="chris",
            source_channel="discord",
            visibility="admin",
            stale_after=datetime(2026, 6, 1, tzinfo=UTC),
        )
        assert req.source_user == "chris"
        assert req.source_channel == "discord"
        assert req.visibility == "admin"
        assert req.stale_after is not None

    def test_search_request_includes_attribution_filters(self):
        """SearchRequest generates attribution filters via get_filters()."""
        req = SearchRequest(
            text="test query",
            source_user=SourceUser(eq="chris"),
            source_channel=SourceChannel(eq="discord"),
            visibility=VisibilityFilter(eq="family"),
        )
        filters = req.get_filters()
        assert "source_user" in filters
        assert filters["source_user"].eq == "chris"
        assert "source_channel" in filters
        assert filters["source_channel"].eq == "discord"
        assert "visibility" in filters
        assert filters["visibility"].eq == "family"


# ---------------------------------------------------------------------------
# Group 6: API endpoint tests (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAPIAttribution:
    """Tests for attribution fields in REST API create, search, and edit endpoints."""

    async def test_api_create_with_attribution_fields(self, client):
        """POST /v1/long-term-memory/ accepts attribution fields on memory records."""
        payload = {
            "memories": [
                {
                    "id": "attr-test-1",
                    "text": "Chris said hello via Discord",
                    "memory_type": "semantic",
                    "source_user": "chris",
                    "source_channel": "discord",
                    "visibility": "family",
                    "stale_after": "2026-12-01T00:00:00Z",
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_api_search_with_attribution_filters(self, client):
        """POST /v1/long-term-memory/search accepts attribution filter objects."""
        from agent_memory_server.config import Settings

        mock_settings = Settings(long_term_memory=True)
        mock_results = MemoryRecordResults(total=0, memories=[])

        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.search_long_term_memories",
                return_value=mock_results,
            ) as mock_search,
        ):
            payload = {
                "text": "family notes",
                "source_user": {"eq": "chris"},
                "source_channel": {"eq": "discord"},
                "visibility": {"eq": "family"},
            }

            response = await client.post(
                "/v1/long-term-memory/search", json=payload
            )
            assert response.status_code == 200

            # Verify filters were forwarded to the search function
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["source_user"].eq == "chris"
            assert call_kwargs["source_channel"].eq == "discord"
            assert call_kwargs["visibility"].eq == "family"

    async def test_api_edit_attribution_fields(self, client):
        """PATCH /v1/long-term-memory/{id} accepts attribution field updates."""
        from agent_memory_server.config import Settings

        mock_settings = Settings(long_term_memory=True)
        updated_memory = _make_memory(
            id="edit-test-1",
            source_user="lindsey",
            source_channel="slack",
            visibility="parents",
            stale_after=datetime(2027, 1, 1, tzinfo=UTC),
        )

        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.update_long_term_memory",
                return_value=updated_memory,
            ) as mock_update,
        ):
            payload = {
                "source_user": "lindsey",
                "source_channel": "slack",
                "visibility": "parents",
                "stale_after": "2027-01-01T00:00:00Z",
            }

            response = await client.patch(
                "/v1/long-term-memory/edit-test-1", json=payload
            )
            assert response.status_code == 200

            # Verify update was called with our attribution fields
            mock_update.assert_called_once()
            update_dict = mock_update.call_args[0][1]
            assert update_dict["source_user"] == "lindsey"
            assert update_dict["source_channel"] == "slack"
            assert update_dict["visibility"] == "parents"
            assert update_dict["stale_after"] is not None

    async def test_api_edit_partial_attribution(self, client):
        """PATCH with only one attribution field does not send the others."""
        from agent_memory_server.config import Settings

        mock_settings = Settings(long_term_memory=True)
        updated_memory = _make_memory(id="edit-partial-1", visibility="admin")

        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.update_long_term_memory",
                return_value=updated_memory,
            ) as mock_update,
        ):
            payload = {"visibility": "admin"}

            response = await client.patch(
                "/v1/long-term-memory/edit-partial-1", json=payload
            )
            assert response.status_code == 200

            update_dict = mock_update.call_args[0][1]
            assert update_dict["visibility"] == "admin"
            # Other attribution fields should not be present (they were None → excluded)
            assert "source_user" not in update_dict
            assert "source_channel" not in update_dict
            assert "stale_after" not in update_dict


# ---------------------------------------------------------------------------
# Group 7: MCP tool tests (3 tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMCPAttribution:
    """Tests for attribution fields in MCP search and edit tools."""

    async def test_mcp_search_passes_attribution_filters(
        self, session, mock_memory_vector_db
    ):
        """MCP search_long_term_memory forwards source_user/source_channel/visibility filters."""
        from mcp.shared.memory import (
            create_connected_server_and_client_session as client_session,
        )

        from agent_memory_server.mcp import mcp_app

        async with client_session(mcp_app._mcp_server) as client:
            with mock.patch(
                "agent_memory_server.mcp.core_search_long_term_memory"
            ) as mock_search:
                mock_search.return_value = MemoryRecordResults(
                    total=0, memories=[]
                )

                await client.call_tool(
                    "search_long_term_memory",
                    {
                        "text": "family notes",
                        "source_user": {"eq": "chris"},
                        "source_channel": {"eq": "discord"},
                        "visibility": {"eq": "everyone"},
                    },
                )

                mock_search.assert_called_once()
                call_args = mock_search.call_args
                # The first positional arg is the SearchRequest
                search_req = call_args[0][0]
                filters = search_req.get_filters()
                assert "source_user" in filters
                assert filters["source_user"].eq == "chris"
                assert "source_channel" in filters
                assert filters["source_channel"].eq == "discord"
                assert "visibility" in filters
                assert filters["visibility"].eq == "everyone"

    async def test_mcp_edit_with_attribution_fields(
        self, session, mock_memory_vector_db
    ):
        """MCP edit_long_term_memory forwards attribution fields to update."""
        from mcp.shared.memory import (
            create_connected_server_and_client_session as client_session,
        )

        from agent_memory_server.mcp import mcp_app

        updated_memory = _make_memory(
            id="mcp-edit-1",
            source_user="chris",
            source_channel="slack",
            visibility="admin",
            stale_after=datetime(2027, 6, 1, tzinfo=UTC),
        )

        async with client_session(mcp_app._mcp_server) as client:
            # core_update_long_term_memory is api.update_long_term_memory aliased in mcp.py
            with mock.patch(
                "agent_memory_server.mcp.core_update_long_term_memory",
                return_value=updated_memory,
            ) as mock_update:
                await client.call_tool(
                    "edit_long_term_memory",
                    {
                        "memory_id": "mcp-edit-1",
                        "source_user": "chris",
                        "source_channel": "slack",
                        "visibility": "admin",
                        "stale_after": "2027-06-01T00:00:00Z",
                    },
                )

                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["memory_id"] == "mcp-edit-1"
                # updates is an EditMemoryRecordRequest
                updates = call_kwargs["updates"]
                assert updates.source_user == "chris"
                assert updates.source_channel == "slack"
                assert updates.visibility == "admin"
                assert updates.stale_after is not None

    async def test_mcp_edit_partial_attribution(
        self, session, mock_memory_vector_db
    ):
        """MCP edit_long_term_memory with only visibility does not include other attribution fields."""
        from mcp.shared.memory import (
            create_connected_server_and_client_session as client_session,
        )

        from agent_memory_server.mcp import mcp_app

        updated_memory = _make_memory(id="mcp-partial-1", visibility="family")

        async with client_session(mcp_app._mcp_server) as client:
            with mock.patch(
                "agent_memory_server.mcp.core_update_long_term_memory",
                return_value=updated_memory,
            ) as mock_update:
                await client.call_tool(
                    "edit_long_term_memory",
                    {
                        "memory_id": "mcp-partial-1",
                        "visibility": "family",
                    },
                )

                mock_update.assert_called_once()
                updates = mock_update.call_args[1]["updates"]
                assert updates.visibility == "family"
                # None-valued fields should remain None on the request object
                assert updates.source_user is None
                assert updates.source_channel is None
                assert updates.stale_after is None
