"""Tests for code review findings from 2026-03-18.

Covers:
  Finding #7:  _parse_list_field comma split corrupts entities containing commas
  Finding #16: Greedy regex in _parse_extraction_response_with_fallback
  Finding #38: bedrock_embedding_model_exists docstring/code disagree on credential errors
  Finding #40: nomic-embed-text prefix missing from sync embed methods
  Finding #9:  Silent exception swallow in search fallback (long_term_memory.py)
  Finding #19: Timer.__enter__ without __exit__ on exception path
  Finding #10: update_task_status silent no-op on corrupt JSON
  Finding #45: Substring topic matching false positives in enforce_topics
  Finding #37: _deduplicate_entity_variants runs before entity cap
"""

import json
import logging
import re
from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ulid import ULID

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(**overrides) -> MemoryRecord:
    """Build a MemoryRecord with sensible defaults."""
    defaults = {
        "id": str(ULID()),
        "text": "test memory",
        "memory_type": MemoryTypeEnum.SEMANTIC,
        "discrete_memory_extracted": "t",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ===========================================================================
# Finding #7: _parse_list_field comma split corrupts entities with commas
# ===========================================================================


class TestFinding07_ParseListFieldCommaSplit:
    """_parse_list_field splits on comma, corrupting entities like
    'Washington, D.C.' into ['Washington', ' D.C.'].

    The storage layer (_memory_to_data) uses ','.join() and the read layer
    uses split(','), so round-tripping through Redis loses multi-part entities.

    Fix: Use pipe '|' as the canonical delimiter everywhere (storage and parse).
    """

    def _parse(self, value):
        """Call _parse_list_field on a concrete instance."""
        from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase

        # Use concrete subclass — abstract base can't be instantiated
        instance = RedisVLMemoryVectorDatabase.__new__(RedisVLMemoryVectorDatabase)
        return instance._parse_list_field(value)

    def test_entity_with_comma_is_corrupted_by_split(self):
        """Pipe-delimited value with embedded comma is preserved as one entity."""
        # After fix, storage uses pipe delimiter, so this value comes from Redis
        # as a pipe-delimited string. An entity with a comma should survive.
        result = self._parse("Washington, D.C.|Chris Baker")
        assert len(result) == 2, (
            f"Expected 2 entities but got {len(result)}: {result}."
        )
        assert result[0] == "Washington, D.C."
        assert result[1] == "Chris Baker"

    def test_pipe_delimited_entities_preserved(self):
        """Pipe-delimited values should be correctly split."""
        result = self._parse("family|personal|home")
        assert result == ["family", "personal", "home"]

    def test_roundtrip_entity_with_comma(self):
        """Storing then reading an entity with a comma should preserve it."""
        from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase

        db = RedisVLMemoryVectorDatabase.__new__(RedisVLMemoryVectorDatabase)
        memory = _make_memory(entities=["Washington, D.C.", "Chris Baker"])
        data = db._memory_to_data(memory)

        # Verify storage uses pipe delimiter
        assert "|" in data["entities"], (
            f"Expected pipe delimiter in stored entities, got: {data['entities']}"
        )

        # Now parse back through _parse_list_field
        parsed = db._parse_list_field(data["entities"])
        assert "Washington, D.C." in parsed, (
            f"Entity 'Washington, D.C.' lost in round-trip. Got: {parsed}"
        )

    def test_empty_and_none(self):
        """Empty string and None return empty lists."""
        assert self._parse("") == []
        assert self._parse(None) == []

    def test_already_a_list(self):
        """List input is returned as-is."""
        assert self._parse(["a", "b"]) == ["a", "b"]

    def test_legacy_comma_only_values_still_parsed(self):
        """Existing Redis data with comma-only delimiters should still work
        via fallback parsing when no pipes are present."""
        # Legacy data: "family,personal" (no pipes)
        result = self._parse("family,personal")
        # After fix, if no pipe is found, fall back to comma split
        assert "family" in result
        assert "personal" in result


# ===========================================================================
# Finding #16: Greedy regex captures past JSON boundary
# ===========================================================================


class TestFinding16_GreedyRegexJsonParse:
    """The greedy regex r'"memories"\\s*:\\s*\\[(.*)\\]' captures from the first
    '[' to the LAST ']' in the entire string. If LLM appends text after JSON,
    the capture includes non-JSON content."""

    def _parse(self, content):
        from agent_memory_server.long_term_memory import (
            _parse_extraction_response_with_fallback,
        )

        return _parse_extraction_response_with_fallback(
            content, logging.getLogger("test")
        )

    def test_valid_json_parses_normally(self):
        """Well-formed JSON should parse without fallback."""
        content = '{"memories": [{"text": "hello"}]}'
        result = self._parse(content)
        assert result["memories"][0]["text"] == "hello"

    def test_trailing_text_after_json(self):
        """LLM output with text after closing brace should still parse."""
        # This is a common LLM failure mode: valid JSON followed by commentary
        content = '{"memories": [{"text": "hello"}]} I hope this helps!'
        result = self._parse(content)
        assert result["memories"][0]["text"] == "hello"

    def test_nested_brackets_in_memory_text(self):
        """Memory text containing brackets should not confuse the parser."""
        content = '{"memories": [{"text": "values are [1,2,3] in the array"}]}'
        result = self._parse(content)
        assert "[1,2,3]" in result["memories"][0]["text"]

    def test_multiple_closing_brackets_with_trailing_text(self):
        """Greedy regex should not capture past the JSON boundary when
        memory text contains brackets AND there is trailing text."""
        content = (
            '{"memories": [{"text": "array [1,2,3] here"}]}'
            " Let me know if you need more!"
        )
        result = self._parse(content)
        assert result["memories"][0]["text"] == "array [1,2,3] here"


# ===========================================================================
# Finding #40: nomic prefix missing from sync embed methods
# ===========================================================================


class TestFinding40_NomicPrefixSyncMethods:
    """Sync embed_documents() and embed_query() do not call
    _apply_nomic_prefix(), so stored vectors lack the 'search_document:'
    prefix. Async methods correctly apply it."""

    @patch("agent_memory_server.llm.embeddings.embedding")
    def test_sync_embed_documents_applies_nomic_prefix(self, mock_embedding):
        """Sync embed_documents should apply nomic 'search_document:' prefix."""
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2]}]
        )
        emb = LiteLLMEmbeddings(model="ollama/nomic-embed-text", dimensions=768)
        emb.embed_documents(["hello world"])

        # Extract the input texts from the call
        call_kwargs = mock_embedding.call_args
        input_texts = call_kwargs.kwargs.get("input", [])

        assert any("search_document:" in t for t in input_texts), (
            f"Expected 'search_document:' prefix in input texts, got: {input_texts}. "
            "Sync embed_documents does not apply nomic prefix."
        )

    @patch("agent_memory_server.llm.embeddings.embedding")
    def test_sync_embed_query_applies_nomic_prefix(self, mock_embedding):
        """Sync embed_query should apply nomic 'search_query:' prefix."""
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2]}]
        )
        emb = LiteLLMEmbeddings(model="ollama/nomic-embed-text", dimensions=768)
        emb.embed_query("hello world")

        call_kwargs = mock_embedding.call_args
        input_texts = call_kwargs.kwargs.get("input", [])

        assert any("search_query:" in t for t in input_texts), (
            f"Expected 'search_query:' prefix in input texts, got: {input_texts}. "
            "Sync embed_query does not apply nomic prefix."
        )

    @patch("agent_memory_server.llm.embeddings.embedding")
    def test_non_nomic_model_not_prefixed(self, mock_embedding):
        """Non-nomic models should NOT get any prefix."""
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2]}]
        )
        emb = LiteLLMEmbeddings(model="text-embedding-3-small", dimensions=1536)
        emb.embed_documents(["hello world"])

        call_kwargs = mock_embedding.call_args
        input_texts = call_kwargs.kwargs.get("input", [])

        assert input_texts == ["hello world"], (
            f"Non-nomic model should not be prefixed, got: {input_texts}"
        )


# ===========================================================================
# Finding #38: bedrock_embedding_model_exists docstring vs code mismatch
# ===========================================================================


class TestFinding38_BedrockDocstringCodeMismatch:
    """Docstring says 'Returns True on credential/permission errors' but
    code returns False. This blocks the model instead of allowing it through."""

    @patch("agent_memory_server._aws.utils.create_bedrock_client")
    def test_credential_error_returns_true_per_docstring(self, mock_client):
        """On ClientError (credentials), the function should return True
        to allow the actual embedding call to proceed (per docstring)."""
        from botocore.exceptions import ClientError

        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the TTL cache to ensure our mock is used
        bedrock_embedding_model_exists.cache.clear()

        mock_client.return_value.list_foundation_models.side_effect = ClientError(
            {"Error": {"Code": "UnauthorizedAccess", "Message": "No creds"}},
            "ListFoundationModels",
        )

        result = bedrock_embedding_model_exists("some-model-id")

        assert result is True, (
            "bedrock_embedding_model_exists returns False on credential errors, "
            "contradicting its docstring which says it should return True "
            "to allow the actual embedding call to proceed."
        )


# ===========================================================================
# Finding #9: Silent exception swallow in search fallback
# ===========================================================================


class TestFinding09_SilentExceptionSwallow:
    """The search fallback at long_term_memory.py:1524-1526 catches ALL
    exceptions with `except Exception: pass`, hiding connectivity errors."""

    def test_search_fallback_has_logging(self):
        """Verify the search fallback logs exceptions instead of bare 'pass'."""
        import inspect

        from agent_memory_server import long_term_memory

        source = inspect.getsource(long_term_memory.search_long_term_memories)

        # Find the problematic pattern: except Exception followed by comment + pass
        # with no logging call in between
        has_bare_except_pass = bool(
            re.search(r"except\s+Exception[^:]*:\s*\n\s*#[^\n]*\n\s*pass", source)
        )
        assert not has_bare_except_pass, (
            "search_long_term_memories has 'except Exception: pass' without logging. "
            "Connection errors, OOM, and other serious issues are silently swallowed."
        )


# ===========================================================================
# Finding #19: Timer.__enter__ without __exit__ on exception path
# ===========================================================================


class TestFinding19_TimerMissingExit:
    """search_memories() calls Timer().__enter__() manually but __exit__()
    is only called on the happy path. An exception between enter and exit
    leaves the timer dangling."""

    def test_timer_exit_is_guaranteed(self):
        """Verify search_memories guarantees __exit__ via try/finally or 'with'."""
        import inspect

        from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase

        source = inspect.getsource(RedisVLMemoryVectorDatabase.search_memories)

        has_manual_enter = "_search_timer.__enter__()" in source
        has_with_timer = "with Timer() as _search_timer" in source
        has_try_finally = "try:" in source and "finally:" in source

        # Either use `with Timer()` or `try:/finally:` to guarantee __exit__
        if has_manual_enter:
            assert has_try_finally or has_with_timer, (
                "search_memories uses manual Timer().__enter__() without a "
                "try/finally or context manager. If an exception occurs, "
                "__exit__ is never called and duration_ms is undefined."
            )


# ===========================================================================
# Finding #10: update_task_status silent no-op on corrupt JSON
# ===========================================================================


class TestFinding10_TaskUpdateSilentFailure:
    """update_task_status returns without error when task JSON is corrupt,
    making callers believe the update succeeded."""

    @pytest.mark.asyncio
    async def test_corrupt_json_returns_false(self):
        """update_task_status should return False (or raise) when task JSON
        is corrupt, so callers know the update failed."""
        from agent_memory_server.tasks import update_task_status

        mock_redis = AsyncMock()
        mock_redis.get.return_value = b"not valid json at all {{{"

        with patch(
            "agent_memory_server.tasks.get_redis_conn", return_value=mock_redis
        ):
            result = await update_task_status(
                "test-task-id",
                status="COMPLETED",
            )

            # Verify Redis.set was NOT called (update was dropped)
            assert mock_redis.set.called is False, (
                "Redis.set was not called — the update was silently dropped "
                "due to corrupt JSON, but no error was raised to the caller."
            )

            # After fix: function should return False to indicate failure
            assert result is False, (
                "update_task_status should return False when task JSON is corrupt, "
                "so callers can detect the failure."
            )


# ===========================================================================
# Finding #45: Substring topic matching false positives
# ===========================================================================


class TestFinding45_SubstringTopicFalsePositives:
    """Short controlled topics like 'ai' match as substrings inside
    unrelated words like 'email', 'maintain', 'training'.

    Fix: Use word-boundary matching instead of bare substring 'in'."""

    def test_email_not_matched_as_ai(self):
        """'email' should NOT match the 'ai' topic via substring."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["email"])
        assert "ai" not in result, (
            f"'email' incorrectly matched 'ai' topic via substring: {result}"
        )

    def test_maintain_not_matched_as_ai(self):
        """'maintain' should NOT match the 'ai' topic via substring."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["maintain"])
        assert "ai" not in result, (
            f"'maintain' incorrectly matched 'ai' topic via substring: {result}"
        )

    def test_training_not_matched_as_ai(self):
        """'training' should NOT match the 'ai' topic via substring."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["training"])
        assert "ai" not in result, (
            f"'training' incorrectly matched 'ai' topic via substring: {result}"
        )

    def test_actual_ai_topic_still_works(self):
        """The literal 'ai' topic should still be recognized."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["ai"])
        assert "ai" in result

    def test_ai_tools_matches_via_direct_or_map(self):
        """'ai tools' should map to 'ai' (it's a legitimate AI topic)."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["ai tools"])
        assert "ai" in result

    def test_home_not_matched_in_homework(self):
        """'homework' should NOT match the 'home' topic."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["homework"])
        assert "home" not in result, (
            f"'homework' incorrectly matched 'home' topic: {result}"
        )

    def test_work_not_matched_in_homework(self):
        """'homework' should NOT match 'work' topic either."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["homework"])
        assert "work" not in result, (
            f"'homework' incorrectly matched 'work' topic: {result}"
        )


# ===========================================================================
# Finding #37: _deduplicate_entity_variants runs before cap
# ===========================================================================


class TestFinding37_DeduplicateBeforeCap:
    """_deduplicate_entity_variants has O(n*m) complexity and runs before
    the MAX_ENTITY_COUNT cap is applied. A huge LLM-returned entity list
    would cause quadratic behavior.

    This is a performance regression test — passes today (fast enough)
    but documents the ordering concern."""

    def test_large_entity_list_performance(self):
        """With a large entity list, dedup should still complete quickly."""
        import time

        from agent_memory_server.extraction import clean_entities

        # Generate 500 entities (mix of single and multi-word)
        entities = [f"entity_{i}" for i in range(250)]
        entities += [f"entity_{i} extended" for i in range(250)]

        start = time.monotonic()
        result = clean_entities(entities)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Should complete in under 100ms even with 500 entities
        assert elapsed_ms < 100, (
            f"clean_entities took {elapsed_ms:.1f}ms for 500 entities."
        )
        # Result should be capped at MAX_ENTITY_COUNT (30)
        assert len(result) <= 30

    def test_cap_applied_after_dedup(self):
        """Verify the cap is applied (entities beyond 30 are dropped)."""
        from agent_memory_server.extraction import clean_entities

        entities = [f"unique_entity_{i}" for i in range(50)]
        result = clean_entities(entities)
        assert len(result) == 30
