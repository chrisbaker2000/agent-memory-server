"""Tests for filter classes in agent_memory_server.filters."""

from datetime import UTC, datetime, timedelta
from enum import Enum
from unittest.mock import AsyncMock, patch

import pytest
from redisvl.query.filter import FilterExpression

from agent_memory_server.filters import (
    CreatedAt,
    DateTimeFilter,
    EnumFilter,
    EventDate,
    LastAccessed,
    MemoryType,
    Namespace,
    NumFilter,
    SessionId,
    TagFilter,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    MemoryRecordResult,
    MemoryRecordResults,
    SearchRequest,
)


class TestTagFilterStartswith:
    """Tests for TagFilter.startswith functionality."""

    def test_startswith_creates_prefix_filter(self):
        """startswith should create a FilterExpression with wildcard suffix."""
        filter_obj = TagFilter(field="namespace", startswith="workspace")
        result = filter_obj.to_filter()

        assert isinstance(result, FilterExpression)
        # The filter should contain the prefix with a wildcard
        filter_str = str(result)
        assert "workspace*" in filter_str
        assert "@namespace:" in filter_str

    def test_startswith_escapes_special_characters(self):
        """startswith should escape special Redis characters like colons."""
        filter_obj = TagFilter(field="namespace", startswith="workspace:abc123")
        result = filter_obj.to_filter()

        filter_str = str(result)
        # Colon should be escaped, but * should not
        assert "workspace\\:abc123*" in filter_str

    def test_startswith_with_namespace_class(self):
        """Namespace filter should support startswith."""
        ns_filter = Namespace(startswith="project:team")
        result = ns_filter.to_filter()

        filter_str = str(result)
        assert "project\\:team*" in filter_str
        assert "@namespace:" in filter_str

    def test_startswith_with_session_id_class(self):
        """SessionId filter should support startswith."""
        session_filter = SessionId(startswith="session-prefix")
        result = session_filter.to_filter()

        filter_str = str(result)
        assert "session\\-prefix*" in filter_str or "session-prefix*" in filter_str
        assert "@session_id:" in filter_str

    def test_startswith_with_user_id_class(self):
        """UserId filter should support startswith."""
        user_filter = UserId(startswith="user-")
        result = user_filter.to_filter()

        filter_str = str(result)
        assert "@user_id:" in filter_str

    def test_startswith_cannot_combine_with_eq(self):
        """startswith and eq cannot both be set."""
        with pytest.raises(ValueError, match="startswith and eq cannot both be set"):
            TagFilter(field="namespace", startswith="prefix", eq="exact")

    def test_startswith_cannot_combine_with_ne(self):
        """startswith and ne cannot both be set."""
        with pytest.raises(ValueError, match="startswith and ne cannot both be set"):
            TagFilter(field="namespace", startswith="prefix", ne="not_this")

    def test_startswith_cannot_combine_with_any(self):
        """startswith and any cannot both be set."""
        with pytest.raises(ValueError, match="startswith and any cannot both be set"):
            TagFilter(field="namespace", startswith="prefix", any=["a", "b"])

    def test_startswith_cannot_combine_with_all(self):
        """startswith and all cannot both be set."""
        with pytest.raises(ValueError, match="startswith and all cannot both be set"):
            TagFilter(field="namespace", startswith="prefix", all=["a", "b"])

    def test_startswith_empty_string_raises(self):
        """startswith with empty string should raise ValueError."""
        with pytest.raises(ValueError, match="startswith cannot be an empty string"):
            TagFilter(field="namespace", startswith="")

    def test_startswith_can_combine_with_other_filters(self):
        """startswith filter can be combined with other filters using &."""
        ns_filter = Namespace(startswith="workspace:abc")
        user_filter = UserId(eq="user123")

        combined = ns_filter.to_filter() & user_filter.to_filter()
        combined_str = str(combined)

        assert "workspace\\:abc*" in combined_str
        assert "user123" in combined_str

    def test_existing_eq_filter_still_works(self):
        """Existing eq functionality should not be affected."""
        filter_obj = TagFilter(field="namespace", eq="exact_match")
        result = filter_obj.to_filter()

        filter_str = str(result)
        assert "exact_match" in filter_str
        assert "*" not in filter_str  # No wildcard for exact match

    def test_existing_ne_filter_still_works(self):
        """Existing ne functionality should not be affected."""
        filter_obj = TagFilter(field="namespace", ne="not_this")
        result = filter_obj.to_filter()

        filter_str = str(result)
        assert "not_this" in filter_str

    def test_existing_any_filter_still_works(self):
        """Existing any functionality should not be affected."""
        filter_obj = TagFilter(field="namespace", any=["ns1", "ns2"])
        result = filter_obj.to_filter()

        filter_str = str(result)
        assert "ns1" in filter_str or "ns2" in filter_str

    def test_startswith_with_spaces_escapes_properly(self):
        """startswith with spaces should escape them properly."""
        filter_obj = TagFilter(field="namespace", startswith="my namespace")
        result = filter_obj.to_filter()

        filter_str = str(result)
        # Space should be escaped
        assert "my\\ namespace*" in filter_str

    def test_startswith_with_slash_escapes_properly(self):
        """startswith with slashes should escape them properly."""
        filter_obj = TagFilter(field="namespace", startswith="workspace/session")
        result = filter_obj.to_filter()

        filter_str = str(result)
        # Check the filter was created (exact escaping may vary)
        assert "@namespace:" in filter_str
        assert "*" in filter_str


class TestNamespacePrefixSearchIntegration:
    """Integration tests for namespace prefix filtering with search."""

    @pytest.mark.asyncio
    async def test_search_with_namespace_startswith_filter(self):
        """Test that namespace startswith filter works in search operations."""
        from agent_memory_server.long_term_memory import search_long_term_memories

        # Create mock results for hierarchical namespaces
        mock_results = MemoryRecordResults(
            memories=[
                MemoryRecordResult(
                    id="mem-1",
                    text="Memory in parent namespace",
                    dist=0.1,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    namespace="workspace:abc123",
                    memory_hash="hash1",
                ),
                MemoryRecordResult(
                    id="mem-2",
                    text="Memory in child session",
                    dist=0.2,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    namespace="workspace:abc123/session:thread-1",
                    memory_hash="hash2",
                ),
            ],
            total=2,
            next_offset=None,
        )

        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = mock_results

        with patch(
            "agent_memory_server.long_term_memory.get_memory_vector_db",
            return_value=mock_adapter,
        ):
            # Search using namespace prefix filter
            await search_long_term_memories(
                text="test query",
                namespace=Namespace(startswith="workspace:abc123"),
            )

            # Verify search was called
            mock_adapter.search_memories.assert_called_once()

            # Check the filter was passed correctly
            call_kwargs = mock_adapter.search_memories.call_args[1]
            assert "namespace" in call_kwargs
            assert call_kwargs["namespace"].startswith == "workspace:abc123"

    @pytest.mark.asyncio
    async def test_namespace_startswith_filter_expression_format(self):
        """Test that the filter expression is in correct Redis format."""
        # This tests the actual Redis query format
        ns_filter = Namespace(startswith="workspace:abc123/session")
        expr = ns_filter.to_filter()

        filter_str = str(expr)
        # Should produce: @namespace:{workspace\:abc123\/session*}
        assert "@namespace:" in filter_str
        assert "*" in filter_str
        # Colon should be escaped
        assert "\\:" in filter_str

    def test_namespace_startswith_combines_with_user_filter(self):
        """Test combining namespace prefix with user_id filter."""
        ns_filter = Namespace(startswith="workspace:abc")
        user_filter = UserId(eq="user123")

        combined = ns_filter.to_filter() & user_filter.to_filter()
        combined_str = str(combined)

        # Both filters should be present
        assert "@namespace:" in combined_str
        assert "@user_id:" in combined_str
        assert "workspace\\:abc*" in combined_str
        assert "user123" in combined_str


class TestTagFilterValidation:
    """Tests for TagFilter validation rules."""

    def test_eq_and_ne_cannot_both_be_set(self):
        """eq and ne cannot both be set."""
        with pytest.raises(ValueError, match="eq and ne cannot both be set"):
            TagFilter(field="test", eq="value1", ne="value2")

    def test_any_and_all_cannot_both_be_set(self):
        """any and all cannot both be set."""
        with pytest.raises(ValueError, match="any and all cannot both be set"):
            TagFilter(field="test", any=["a"], all=["b"])

    def test_all_cannot_be_empty_list(self):
        """all cannot be an empty list."""
        with pytest.raises(ValueError, match="all cannot be an empty list"):
            TagFilter(field="test", all=[])

    def test_any_cannot_be_empty_list(self):
        """any cannot be an empty list."""
        with pytest.raises(ValueError, match="any cannot be an empty list"):
            TagFilter(field="test", any=[])

    def test_all_filter_creates_expression(self):
        """all filter should create a valid FilterExpression."""
        filter_obj = TagFilter(field="tags", all=["tag1", "tag2"])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_no_filter_provided_raises(self):
        """No filter provided should raise ValueError."""
        filter_obj = TagFilter(field="test")
        with pytest.raises(ValueError, match="No filter provided"):
            filter_obj.to_filter()


class _SampleEnum(Enum):
    """Sample enum for testing EnumFilter."""

    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


class TestEnumFilter:
    """Tests for EnumFilter class."""

    def test_eq_filter_with_valid_value(self):
        """eq filter with valid enum value."""
        filter_obj = EnumFilter(field="status", enum_class=_SampleEnum, eq="value1")
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)
        assert "value1" in str(result)

    def test_ne_filter_with_valid_value(self):
        """ne filter with valid enum value."""
        filter_obj = EnumFilter(field="status", enum_class=_SampleEnum, ne="value1")
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_any_filter_with_valid_values(self):
        """any filter with valid enum values."""
        filter_obj = EnumFilter(
            field="status", enum_class=_SampleEnum, any=["value1", "value2"]
        )
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_all_filter_with_valid_values(self):
        """all filter with valid enum values."""
        filter_obj = EnumFilter(
            field="status", enum_class=_SampleEnum, all=["value1", "value2"]
        )
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_eq_with_invalid_value_raises(self):
        """eq with invalid enum value should raise."""
        with pytest.raises(ValueError, match="not in valid enum values"):
            EnumFilter(field="status", enum_class=_SampleEnum, eq="invalid")

    def test_ne_with_invalid_value_raises(self):
        """ne with invalid enum value should raise."""
        with pytest.raises(ValueError, match="not in valid enum values"):
            EnumFilter(field="status", enum_class=_SampleEnum, ne="invalid")

    def test_any_with_invalid_value_raises(self):
        """any with invalid enum value should raise."""
        with pytest.raises(ValueError, match="not in valid enum values"):
            EnumFilter(
                field="status", enum_class=_SampleEnum, any=["value1", "invalid"]
            )

    def test_all_with_invalid_value_raises(self):
        """all with invalid enum value should raise."""
        with pytest.raises(ValueError, match="not in valid enum values"):
            EnumFilter(field="status", enum_class=_SampleEnum, all=["invalid"])

    def test_eq_and_ne_cannot_both_be_set(self):
        """eq and ne cannot both be set."""
        with pytest.raises(ValueError, match="eq and ne cannot both be set"):
            EnumFilter(field="status", enum_class=_SampleEnum, eq="value1", ne="value2")

    def test_any_and_all_cannot_both_be_set(self):
        """any and all cannot both be set."""
        with pytest.raises(ValueError, match="any and all cannot both be set"):
            EnumFilter(
                field="status", enum_class=_SampleEnum, any=["value1"], all=["value2"]
            )

    def test_all_cannot_be_empty(self):
        """all cannot be an empty list."""
        with pytest.raises(ValueError, match="all cannot be an empty list"):
            EnumFilter(field="status", enum_class=_SampleEnum, all=[])

    def test_any_cannot_be_empty(self):
        """any cannot be an empty list."""
        with pytest.raises(ValueError, match="any cannot be an empty list"):
            EnumFilter(field="status", enum_class=_SampleEnum, any=[])

    def test_no_filter_provided_raises(self):
        """No filter provided should raise ValueError."""
        filter_obj = EnumFilter(field="status", enum_class=_SampleEnum)
        with pytest.raises(ValueError, match="No filter provided"):
            filter_obj.to_filter()


class TestNumFilter:
    """Tests for NumFilter class."""

    def test_eq_filter(self):
        """eq filter creates valid expression."""
        filter_obj = NumFilter(field="count", eq=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_ne_filter(self):
        """ne filter creates valid expression."""
        filter_obj = NumFilter(field="count", ne=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_gt_filter(self):
        """gt filter creates valid expression."""
        filter_obj = NumFilter(field="count", gt=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_lt_filter(self):
        """lt filter creates valid expression."""
        filter_obj = NumFilter(field="count", lt=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_gte_filter(self):
        """gte filter creates valid expression."""
        filter_obj = NumFilter(field="count", gte=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_lte_filter(self):
        """lte filter creates valid expression."""
        filter_obj = NumFilter(field="count", lte=10)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_between_filter(self):
        """between filter creates valid expression."""
        filter_obj = NumFilter(field="count", between=[5, 15])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_between_must_have_two_values(self):
        """between must be a list of two numbers."""
        with pytest.raises(ValueError, match="between must be a list of two numbers"):
            NumFilter(field="count", between=[5])

    def test_between_and_eq_cannot_both_be_set(self):
        """between and eq cannot both be set."""
        with pytest.raises(ValueError, match="between and eq cannot both be set"):
            NumFilter(field="count", between=[5, 15], eq=10)

    def test_between_and_ne_cannot_both_be_set(self):
        """between and ne cannot both be set."""
        with pytest.raises(ValueError, match="between and ne cannot both be set"):
            NumFilter(field="count", between=[5, 15], ne=10)

    def test_between_and_gt_cannot_both_be_set(self):
        """between and gt cannot both be set."""
        with pytest.raises(ValueError, match="between and gt cannot both be set"):
            NumFilter(field="count", between=[5, 15], gt=10)

    def test_between_and_lt_cannot_both_be_set(self):
        """between and lt cannot both be set."""
        with pytest.raises(ValueError, match="between and lt cannot both be set"):
            NumFilter(field="count", between=[5, 15], lt=10)

    def test_between_and_gte_cannot_both_be_set(self):
        """between and gte cannot both be set."""
        with pytest.raises(ValueError, match="between and gte cannot both be set"):
            NumFilter(field="count", between=[5, 15], gte=10)

    def test_no_filter_provided_raises(self):
        """No filter provided should raise ValueError."""
        filter_obj = NumFilter(field="count")
        with pytest.raises(ValueError, match="No filter provided"):
            filter_obj.to_filter()


class TestDateTimeFilter:
    """Tests for DateTimeFilter class."""

    def test_eq_filter(self):
        """eq filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", eq=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_ne_filter(self):
        """ne filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", ne=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_gt_filter(self):
        """gt filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", gt=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_lt_filter(self):
        """lt filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", lt=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_gte_filter(self):
        """gte filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", gte=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_lte_filter(self):
        """lte filter creates valid expression."""
        dt = datetime.now(UTC)
        filter_obj = DateTimeFilter(field="created_at", lte=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_between_filter(self):
        """between filter creates valid expression."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        filter_obj = DateTimeFilter(field="created_at", between=[dt1, dt2])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_between_must_have_two_values(self):
        """between must be a list of two datetimes."""
        dt = datetime.now(UTC)
        with pytest.raises(ValueError, match="between must be a list of two datetimes"):
            DateTimeFilter(field="created_at", between=[dt])

    def test_between_and_eq_cannot_both_be_set(self):
        """between and eq cannot both be set."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        with pytest.raises(ValueError, match="between and eq cannot both be set"):
            DateTimeFilter(field="created_at", between=[dt1, dt2], eq=dt1)

    def test_between_and_ne_cannot_both_be_set(self):
        """between and ne cannot both be set."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        with pytest.raises(ValueError, match="between and ne cannot both be set"):
            DateTimeFilter(field="created_at", between=[dt1, dt2], ne=dt1)

    def test_between_and_gt_cannot_both_be_set(self):
        """between and gt cannot both be set."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        with pytest.raises(ValueError, match="between and gt cannot both be set"):
            DateTimeFilter(field="created_at", between=[dt1, dt2], gt=dt1)

    def test_between_and_lt_cannot_both_be_set(self):
        """between and lt cannot both be set."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        with pytest.raises(ValueError, match="between and lt cannot both be set"):
            DateTimeFilter(field="created_at", between=[dt1, dt2], lt=dt1)

    def test_between_and_gte_cannot_both_be_set(self):
        """between and gte cannot both be set."""
        dt1 = datetime.now(UTC)
        dt2 = dt1 + timedelta(days=7)
        with pytest.raises(ValueError, match="between and gte cannot both be set"):
            DateTimeFilter(field="created_at", between=[dt1, dt2], gte=dt1)

    def test_no_filter_provided_raises(self):
        """No filter provided should raise ValueError."""
        filter_obj = DateTimeFilter(field="created_at")
        with pytest.raises(ValueError, match="No filter provided"):
            filter_obj.to_filter()


class TestMemoryType:
    """Tests for MemoryType filter class."""

    def test_memory_type_eq_filter(self):
        """MemoryType eq filter with valid value."""
        filter_obj = MemoryType(eq="message")
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)
        assert "message" in str(result)

    def test_memory_type_ne_filter(self):
        """MemoryType ne filter with valid value."""
        filter_obj = MemoryType(ne="message")
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_memory_type_any_filter(self):
        """MemoryType any filter with valid values."""
        filter_obj = MemoryType(any=["message", "semantic"])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_memory_type_invalid_value_raises(self):
        """MemoryType with invalid value should raise."""
        with pytest.raises(ValueError, match="not in valid enum values"):
            MemoryType(eq="invalid_type")


class TestSpecializedFilters:
    """Tests for specialized filter classes (CreatedAt, LastAccessed, etc.)."""

    def test_created_at_filter(self):
        """CreatedAt filter works correctly."""
        dt = datetime.now(UTC)
        filter_obj = CreatedAt(gte=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_last_accessed_filter(self):
        """LastAccessed filter works correctly."""
        dt = datetime.now(UTC)
        filter_obj = LastAccessed(lte=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_event_date_filter(self):
        """EventDate filter works correctly."""
        dt = datetime.now(UTC)
        filter_obj = EventDate(eq=dt)
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)

    def test_topics_filter(self):
        """Topics filter works correctly."""
        filter_obj = Topics(any=["topic1", "topic2"])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)


class TestTagFilterNone:
    """Tests for TagFilter `none` (negation) predicate — LAB-401.

    Before this fix, a `{none: [...]}` filter (which the JS agent-memory-client
    SDK emits) was silently dropped by the Pydantic model, leaving every
    predicate None, so `to_filter()` raised "No filter provided" → HTTP 500.
    """

    def test_none_creates_negated_tag_filter(self):
        filter_obj = TagFilter(field="topics", none=["family", "work"])
        result = filter_obj.to_filter()
        assert isinstance(result, FilterExpression)
        # RedisVL renders Tag != [list] as a NONE-of expression.
        assert str(result) == "(-@topics:{family|work})"

    def test_none_single_value(self):
        result = TagFilter(field="topics", none=["family"]).to_filter()
        assert str(result) == "(-@topics:{family})"

    def test_topics_none_filter(self):
        result = Topics(none=["family"]).to_filter()
        assert str(result) == "(-@topics:{family})"

    @pytest.mark.asyncio
    async def test_topics_none_does_not_trigger_soft_filter_fallback(self):
        """A negated filter that returns zero must NOT relax into an unfiltered
        re-search — that would surface the very records the caller excluded
        (codex F1 on the LAB-401 PR). A zero-result negation is a legitimate
        empty set, so the strict (negated) search must be the only call."""
        from unittest.mock import AsyncMock, patch

        from agent_memory_server import api
        from agent_memory_server.config import Settings

        zero_results = MemoryRecordResults(memories=[], total=0, next_offset=None)
        with (
            patch.object(api, "settings", Settings(long_term_memory=True)),
            patch.object(
                api.long_term_memory,
                "search_long_term_memories",
                AsyncMock(return_value=zero_results),
            ) as mock_search,
        ):
            result = await api.search_long_term_memory(
                SearchRequest(text="query", topics=Topics(none=["family"])),
                background_tasks=AsyncMock(),
                optimize_query=False,
                current_user=object(),
            )

        assert result.total == 0
        # Exactly one (strict) search — no relaxed fallback call.
        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["topics"].none == ["family"]

    def test_none_empty_list_raises(self):
        with pytest.raises(ValueError, match="none cannot be an empty list"):
            TagFilter(field="topics", none=[])

    def test_none_cannot_combine_with_eq(self):
        with pytest.raises(ValueError, match="none and eq cannot both be set"):
            TagFilter(field="topics", none=["a"], eq="b")

    def test_none_cannot_combine_with_any(self):
        with pytest.raises(ValueError, match="none and any cannot both be set"):
            TagFilter(field="topics", none=["a"], any=["b"])

    def test_none_cannot_combine_with_all(self):
        with pytest.raises(ValueError, match="none and all cannot both be set"):
            TagFilter(field="topics", none=["a"], all=["b"])

    def test_none_cannot_combine_with_ne(self):
        with pytest.raises(ValueError, match="none and ne cannot both be set"):
            TagFilter(field="topics", none=["a"], ne="b")

    def test_none_cannot_combine_with_startswith(self):
        with pytest.raises(ValueError, match="startswith and none cannot both be set"):
            TagFilter(field="topics", none=["a"], startswith="p")

    def test_existing_any_filter_unaffected(self):
        """Regression: the positive `any` path still works after adding `none`."""
        result = TagFilter(field="topics", any=["t1", "t2"]).to_filter()
        assert str(result) == "@topics:{t1|t2}"


class TestEnumFilterNone:
    """Tests for EnumFilter `none` (negation) predicate — LAB-401."""

    def test_none_creates_negated_filter(self):
        result = EnumFilter(
            field="status", enum_class=_SampleEnum, none=["value1", "value2"]
        ).to_filter()
        assert isinstance(result, FilterExpression)
        assert str(result) == "(-@status:{value1|value2})"

    def test_none_invalid_enum_value_raises(self):
        with pytest.raises(ValueError, match="not in valid enum values"):
            EnumFilter(field="status", enum_class=_SampleEnum, none=["bogus"])

    def test_none_empty_list_raises(self):
        with pytest.raises(ValueError, match="none cannot be an empty list"):
            EnumFilter(field="status", enum_class=_SampleEnum, none=[])

    def test_none_cannot_combine_with_any(self):
        with pytest.raises(ValueError, match="none and any cannot both be set"):
            EnumFilter(
                field="status", enum_class=_SampleEnum, none=["value1"], any=["value2"]
            )
