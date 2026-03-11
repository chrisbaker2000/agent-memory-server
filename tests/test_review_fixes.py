"""Tests for specific fixes from code review.

Covers:
  1. VISIBILITY_RANK shared constant (models.py) — shape, ordering, and
     behavioral consistency between extraction._resolve_parent_attribution
     and long_term_memory.merge_memories_with_llm.
  2. MockMemoryVectorDatabase attribution propagation — verifies that
     source_user, source_channel, visibility, and stale_after survive
     add_memories -> search_memories / list_memories round-trips.
  3. RecencyAggregationQuery.DEFAULT_RETURN_FIELDS — verifies the four
     attribution fields are present.
  4. Telemetry RESOURCE_ATTRS — verifies host.arch and os.type use
     platform module values (not hardcoded strings).
"""

import platform
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from ulid import ULID

from agent_memory_server.models import VISIBILITY_RANK, MemoryRecord, MemoryTypeEnum


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


# ===========================================================================
# 1. VISIBILITY_RANK shared constant
# ===========================================================================


class TestVisibilityRankConstant:
    """Verify VISIBILITY_RANK exists, has the expected keys/ordering,
    and that both extraction and merge use the same ranking behavior."""

    def test_visibility_rank_has_expected_keys(self):
        """VISIBILITY_RANK contains all expected visibility levels."""
        expected_keys = {
            "everyone",
            "family",
            "restricted",
            "private",
            "parents",
            "admin",
        }
        assert set(VISIBILITY_RANK.keys()) == expected_keys

    def test_visibility_rank_ordering(self):
        """Ranks increase monotonically from least to most restrictive."""
        assert VISIBILITY_RANK["everyone"] < VISIBILITY_RANK["family"]
        assert VISIBILITY_RANK["family"] < VISIBILITY_RANK["restricted"]
        assert VISIBILITY_RANK["restricted"] < VISIBILITY_RANK["private"]
        assert VISIBILITY_RANK["private"] < VISIBILITY_RANK["parents"]
        assert VISIBILITY_RANK["parents"] < VISIBILITY_RANK["admin"]

    def test_visibility_rank_everyone_is_zero(self):
        """The least restrictive level ('everyone') has rank 0."""
        assert VISIBILITY_RANK["everyone"] == 0

    def test_visibility_rank_admin_is_highest(self):
        """'admin' has the highest rank value."""
        assert VISIBILITY_RANK["admin"] == max(VISIBILITY_RANK.values())

    def test_resolve_parent_attribution_uses_same_ranking(self):
        """_resolve_parent_attribution picks the most restrictive visibility
        using the same ranking as VISIBILITY_RANK."""
        from agent_memory_server.extraction import _resolve_parent_attribution

        # Create parents with 'family' and 'admin' — admin should win
        p1 = _make_memory(visibility="family")
        p2 = _make_memory(visibility="admin")

        _, _, resolved = _resolve_parent_attribution([p1, p2])
        assert resolved == "admin"

        # Reverse order should give the same result
        _, _, resolved2 = _resolve_parent_attribution([p2, p1])
        assert resolved2 == "admin"

    def test_resolve_parent_attribution_everyone_vs_private(self):
        """'private' wins over 'everyone' in _resolve_parent_attribution."""
        from agent_memory_server.extraction import _resolve_parent_attribution

        p1 = _make_memory(visibility="everyone")
        p2 = _make_memory(visibility="private")

        _, _, resolved = _resolve_parent_attribution([p1, p2])
        assert resolved == "private"

    @patch("agent_memory_server.long_term_memory.LLMClient")
    @pytest.mark.asyncio
    async def test_merge_uses_same_ranking_as_extraction(self, mock_llm_cls):
        """merge_memories_with_llm selects the same most-restrictive visibility
        that _resolve_parent_attribution would for the same inputs."""
        from agent_memory_server.extraction import _resolve_parent_attribution
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris", visibility="restricted")
        m2 = _make_memory(source_user="chris", visibility="parents")

        # Extraction path
        _, _, extraction_vis = _resolve_parent_attribution([m1, m2])

        # Merge path
        merged = await merge_memories_with_llm([m1, m2])

        assert extraction_vis == merged.visibility
        assert merged.visibility == "parents"

    @patch("agent_memory_server.long_term_memory.LLMClient")
    @pytest.mark.asyncio
    async def test_merge_all_everyone_stays_everyone(self, mock_llm_cls):
        """When all memories have 'everyone' visibility, merged result is 'everyone'."""
        from agent_memory_server.long_term_memory import merge_memories_with_llm

        mock_llm_cls.create_chat_completion = AsyncMock(
            return_value=AsyncMock(content="merged text")
        )

        m1 = _make_memory(source_user="chris", visibility="everyone")
        m2 = _make_memory(source_user="chris", visibility="everyone")

        merged = await merge_memories_with_llm([m1, m2])
        assert merged.visibility == "everyone"


# ===========================================================================
# 2. MockMemoryVectorDatabase attribution propagation
# ===========================================================================


@pytest.mark.asyncio
class TestMockVectorDbAttributionPropagation:
    """Verify that MockMemoryVectorDatabase carries attribution fields
    through add_memories -> search_memories / list_memories."""

    async def _build_db_with_memory(self):
        """Helper: create a MockMemoryVectorDatabase, add one attributed memory."""
        from tests.conftest import MockMemoryVectorDatabase

        db = MockMemoryVectorDatabase()
        stale_dt = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)

        memory = _make_memory(
            id="attr-prop-1",
            text="attributed memory",
            source_user="chris",
            source_channel="discord",
            visibility="parents",
            stale_after=stale_dt,
        )
        await db.add_memories([memory])
        return db, stale_dt

    async def test_add_memories_stores_attribution(self):
        """add_memories stores all four attribution fields in the internal dict."""
        db, stale_dt = await self._build_db_with_memory()

        stored = db.memories["attr-prop-1"]
        assert stored.source_user == "chris"
        assert stored.source_channel == "discord"
        assert stored.visibility == "parents"
        assert stored.stale_after == stale_dt

    async def test_search_memories_returns_attribution(self):
        """search_memories results include all four attribution fields."""
        db, stale_dt = await self._build_db_with_memory()

        results = await db.search_memories(query="attributed")
        assert results.total == 1

        result = results.memories[0]
        assert result.source_user == "chris"
        assert result.source_channel == "discord"
        assert result.visibility == "parents"
        assert result.stale_after == stale_dt

    async def test_list_memories_returns_attribution(self):
        """list_memories results include all four attribution fields."""
        db, stale_dt = await self._build_db_with_memory()

        results = await db.list_memories()
        assert results.total == 1

        result = results.memories[0]
        assert result.source_user == "chris"
        assert result.source_channel == "discord"
        assert result.visibility == "parents"
        assert result.stale_after == stale_dt

    async def test_search_memories_none_attribution_preserved(self):
        """Memories with None attribution fields propagate None through search."""
        from tests.conftest import MockMemoryVectorDatabase

        db = MockMemoryVectorDatabase()

        memory = _make_memory(
            id="none-attr",
            text="no attribution",
            source_user=None,
            source_channel=None,
            visibility="everyone",
            stale_after=None,
        )
        await db.add_memories([memory])

        results = await db.search_memories(query="no attribution")
        result = results.memories[0]
        assert result.source_user is None
        assert result.source_channel is None
        assert result.visibility == "everyone"
        assert result.stale_after is None

    async def test_multiple_memories_with_different_attribution(self):
        """Multiple memories with different attribution fields are each preserved."""
        from tests.conftest import MockMemoryVectorDatabase

        db = MockMemoryVectorDatabase()

        m1 = _make_memory(
            id="m-chris",
            text="chris memory",
            source_user="chris",
            source_channel="slack",
            visibility="admin",
            stale_after=datetime(2027, 1, 1, tzinfo=UTC),
        )
        m2 = _make_memory(
            id="m-lindsey",
            text="lindsey memory",
            source_user="lindsey",
            source_channel="discord",
            visibility="family",
            stale_after=None,
        )
        await db.add_memories([m1, m2])

        results = await db.search_memories(query="memory")
        assert results.total == 2

        by_id = {r.id: r for r in results.memories}
        assert by_id["m-chris"].source_user == "chris"
        assert by_id["m-chris"].source_channel == "slack"
        assert by_id["m-chris"].visibility == "admin"
        assert by_id["m-chris"].stale_after == datetime(2027, 1, 1, tzinfo=UTC)

        assert by_id["m-lindsey"].source_user == "lindsey"
        assert by_id["m-lindsey"].source_channel == "discord"
        assert by_id["m-lindsey"].visibility == "family"
        assert by_id["m-lindsey"].stale_after is None


# ===========================================================================
# 3. RecencyAggregationQuery.DEFAULT_RETURN_FIELDS
# ===========================================================================


class TestRecencyAggregationQueryFields:
    """Verify DEFAULT_RETURN_FIELDS includes attribution fields."""

    def test_default_return_fields_includes_source_user(self):
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        assert "source_user" in RecencyAggregationQuery.DEFAULT_RETURN_FIELDS

    def test_default_return_fields_includes_source_channel(self):
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        assert "source_channel" in RecencyAggregationQuery.DEFAULT_RETURN_FIELDS

    def test_default_return_fields_includes_visibility(self):
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        assert "visibility" in RecencyAggregationQuery.DEFAULT_RETURN_FIELDS

    def test_default_return_fields_includes_stale_after(self):
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        assert "stale_after" in RecencyAggregationQuery.DEFAULT_RETURN_FIELDS

    def test_default_return_fields_all_four_attribution_present(self):
        """All four attribution fields are in DEFAULT_RETURN_FIELDS as a group."""
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        attribution_fields = {
            "source_user",
            "source_channel",
            "visibility",
            "stale_after",
        }
        present = attribution_fields.intersection(
            RecencyAggregationQuery.DEFAULT_RETURN_FIELDS
        )
        assert present == attribution_fields


# ===========================================================================
# 4. Telemetry uses platform detection
# ===========================================================================


class TestTelemetryPlatformDetection:
    """Verify RESOURCE_ATTRS uses platform module for host.arch and os.type."""

    def _get_attr_value(self, key: str) -> str:
        """Extract a string value from RESOURCE_ATTRS by key."""
        from agent_memory_server.telemetry import RESOURCE_ATTRS

        for attr in RESOURCE_ATTRS:
            if attr["key"] == key:
                return attr["value"]["stringValue"]
        raise KeyError(f"Attribute '{key}' not found in RESOURCE_ATTRS")

    def test_host_arch_matches_platform_machine(self):
        """host.arch should match platform.machine() output."""
        value = self._get_attr_value("host.arch")
        assert value == platform.machine()

    def test_os_type_matches_platform_system_lower(self):
        """os.type should match platform.system().lower() output."""
        value = self._get_attr_value("os.type")
        assert value == platform.system().lower()

    def test_host_arch_is_not_hardcoded_empty(self):
        """host.arch must not be an empty string."""
        value = self._get_attr_value("host.arch")
        assert value != ""

    def test_os_type_is_not_hardcoded_empty(self):
        """os.type must not be an empty string."""
        value = self._get_attr_value("os.type")
        assert value != ""

    def test_resource_attrs_contains_host_arch(self):
        """RESOURCE_ATTRS contains a 'host.arch' entry."""
        from agent_memory_server.telemetry import RESOURCE_ATTRS

        keys = [attr["key"] for attr in RESOURCE_ATTRS]
        assert "host.arch" in keys

    def test_resource_attrs_contains_os_type(self):
        """RESOURCE_ATTRS contains an 'os.type' entry."""
        from agent_memory_server.telemetry import RESOURCE_ATTRS

        keys = [attr["key"] for attr in RESOURCE_ATTRS]
        assert "os.type" in keys

    def test_resource_attrs_service_name(self):
        """RESOURCE_ATTRS contains the expected service.name."""
        value = self._get_attr_value("service.name")
        assert value == "memory-server"

    def test_resource_attrs_deployment_environment(self):
        """RESOURCE_ATTRS contains deployment.environment = 'homelab'."""
        value = self._get_attr_value("deployment.environment")
        assert value == "homelab"
