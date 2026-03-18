"""Tests for bugs identified in the March 18, 2026 code review.

Memory server fork findings:
  #3  (CRITICAL) — Telemetry flush holds lock during HTTP call
  #10 (MEDIUM)   — Extraction marks messages as processed on LLM failure
  #22 (MEDIUM)   — enforce_topics() non-deterministic set iteration
  #34 (LOW)      — Regex truncation on ] in memory text
  #16 (MEDIUM)   — get_working_memory() swallows all exceptions
  #15 (MEDIUM)   — count_memories() fetches all records for a count
  #21 (MEDIUM)   — PATCH can't clear optional fields to null
"""

import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Finding #3 — Telemetry flush holds lock during HTTP call
# ============================================================================


class TestFinding03_TelemetryLockDuringHTTP:
    """telemetry.py:188-231 — _flush_locked() does httpx.Client().post()
    while holding _buffer_lock. If SigNoz is slow (5s timeout), all threads
    calling record_metric() block.

    After fix: buffer is drained under lock, HTTP happens outside lock.
    """

    def test_flush_does_not_block_record_metric_during_http(self):
        """record_metric() should not block while flush HTTP is in-flight."""
        from agent_memory_server import telemetry

        # Save originals and restore after test
        orig_buffer = telemetry._metric_buffer
        orig_enabled = telemetry.TELEMETRY_ENABLED
        orig_endpoint = telemetry.OTLP_ENDPOINT

        try:
            telemetry._metric_buffer = []
            telemetry.TELEMETRY_ENABLED = True
            telemetry.OTLP_ENDPOINT = "http://localhost:99999/v1/metrics"

            # Seed buffer to trigger overflow flush
            for i in range(telemetry.MAX_BUFFER_SIZE - 1):
                telemetry._metric_buffer.append({"name": f"test.{i}"})

            # Track when record_metric starts and finishes
            record_metric_duration_ms = None

            # Mock httpx to simulate a slow endpoint
            slow_post = MagicMock()

            def slow_http_post(*args, **kwargs):
                time.sleep(0.5)  # 500ms simulated latency
                resp = MagicMock()
                resp.status_code = 200
                return resp

            slow_post.post = slow_http_post
            slow_post.__enter__ = lambda s: s
            slow_post.__exit__ = lambda s, *a: None

            with patch("httpx.Client", return_value=slow_post):
                # Trigger a flush by adding the MAX_BUFFER_SIZE-th metric
                # This will call _flush_locked() inside record_metric()

                # Spawn a thread that does the overflow-triggering record
                def overflow_record():
                    telemetry.record_metric("test.overflow", 1.0)

                overflow_thread = threading.Thread(target=overflow_record)
                overflow_thread.start()

                # Give the overflow thread a moment to start the flush
                time.sleep(0.05)

                # Now try record_metric from the main thread — should NOT block
                start = time.monotonic()
                telemetry.record_metric("test.concurrent", 42.0)
                record_metric_duration_ms = (time.monotonic() - start) * 1000

                overflow_thread.join(timeout=2.0)

            # After the fix, record_metric should complete quickly (<200ms)
            # even while flush HTTP is in-flight (500ms)
            assert record_metric_duration_ms < 300, (
                f"record_metric() took {record_metric_duration_ms:.0f}ms — "
                f"it's blocked by the flush HTTP call (expected <300ms)"
            )

        finally:
            telemetry._metric_buffer = orig_buffer
            telemetry.TELEMETRY_ENABLED = orig_enabled
            telemetry.OTLP_ENDPOINT = orig_endpoint

    def test_flush_creates_new_httpx_client_each_time(self):
        """Finding #33 (LOW): Each flush creates a new httpx.Client.
        After fix, a persistent client should be reused."""
        from agent_memory_server import telemetry

        orig_buffer = telemetry._metric_buffer
        orig_enabled = telemetry.TELEMETRY_ENABLED

        try:
            telemetry.TELEMETRY_ENABLED = True
            client_instances = []

            original_client_cls = None
            try:
                import httpx
                original_client_cls = httpx.Client
            except ImportError:
                pytest.skip("httpx not installed")

            def tracking_client(*args, **kwargs):
                instance = MagicMock()
                resp = MagicMock()
                resp.status_code = 200
                instance.post.return_value = resp
                instance.__enter__ = lambda s: s
                instance.__exit__ = lambda s, *a: None
                client_instances.append(instance)
                return instance

            with patch("httpx.Client", side_effect=tracking_client):
                # Flush twice
                telemetry._metric_buffer = [{"name": "test.1"}]
                telemetry.flush()
                telemetry._metric_buffer = [{"name": "test.2"}]
                telemetry.flush()

            # Current behavior: 2 separate Client instances (wasteful)
            # This test documents the current behavior
            assert len(client_instances) == 2, (
                f"Expected 2 Client instances (current behavior), got {len(client_instances)}"
            )

        finally:
            telemetry._metric_buffer = orig_buffer
            telemetry.TELEMETRY_ENABLED = orig_enabled


# ============================================================================
# Finding #22 — enforce_topics() non-deterministic set iteration
# ============================================================================


class TestFinding22_EnforceTopicsNonDeterminism:
    """extraction.py:553-571 — CONTROLLED_TOPICS is a set.
    Topic 'home_security' could match 'home' or 'security' depending on
    set iteration order, which varies across Python restarts.

    The substring match loop iterates a set, so order is not guaranteed.
    """

    def test_ambiguous_topic_matches_consistently(self):
        """A topic containing multiple controlled-topic substrings should
        produce deterministic results regardless of set iteration order."""
        from agent_memory_server.extraction import enforce_topics, CONTROLLED_TOPICS

        # Find two controlled topics where one is a substring of a compound
        # word — this is the class of inputs that's non-deterministic
        # "home_security" contains both "home" and "security" (if they exist)
        has_home = "home" in CONTROLLED_TOPICS
        has_security = "security" in CONTROLLED_TOPICS

        if not has_home or not has_security:
            pytest.skip("Need both 'home' and 'security' in CONTROLLED_TOPICS")

        # Run many times — if non-deterministic, results will vary
        results = set()
        for _ in range(50):
            output = enforce_topics(["home_security"])
            if output:
                results.add(output[0])

        # After fix: should always produce the same result
        assert len(results) <= 1, (
            f"enforce_topics(['home_security']) produced multiple different "
            f"results across runs: {results} — non-deterministic!"
        )

    def test_exact_match_takes_precedence_over_substring(self):
        """If a topic exactly matches a controlled topic, it should be used
        directly without falling through to substring matching."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["home"])
        assert result == ["home"]

        result = enforce_topics(["security"])
        assert result == ["security"]

    def test_enforce_topics_deduplicates(self):
        """Duplicate inputs should be deduplicated in output."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["home", "home", "HOME"])
        assert result == ["home"]

    def test_enforce_topics_drops_unknown(self):
        """Topics not in vocabulary are silently dropped."""
        from agent_memory_server.extraction import enforce_topics

        result = enforce_topics(["zzz_not_a_real_topic_xyz"])
        assert result == []

    def test_enforce_topics_drops_long_descriptions(self):
        """Topics >30 chars with spaces are dropped (likely sentences)."""
        from agent_memory_server.extraction import enforce_topics

        long_topic = "this is a very long topic description that is clearly a sentence"
        assert len(long_topic) > 30 and " " in long_topic
        result = enforce_topics([long_topic])
        assert result == []


# ============================================================================
# Finding #34 — Regex truncation on ] in memory text
# ============================================================================


class TestFinding34_RegexTruncationOnBrackets:
    """long_term_memory.py:89 — The regex r'"memories"\\s*:\\s*\\[(.*?)\\]'
    with non-greedy .*? matches the FIRST ']'. If a memory text contains
    a literal ']', the regex truncates prematurely.
    """

    def test_regex_truncates_on_bracket_in_text(self):
        """Demonstrate the bug: memory text with ] causes premature truncation."""
        # This is malformed JSON the LLM might return (missing closing })
        content = '{"memories": [{"text": "array [1,2,3] was processed", "topics": ["tech"]}]}'

        # This is valid JSON — standard parser should handle it
        parsed = json.loads(content)
        assert parsed["memories"][0]["text"] == "array [1,2,3] was processed"

        # But the fallback regex (used when JSON is malformed) truncates
        # Simulate what happens with slightly malformed JSON
        malformed = '{"memories": [{"text": "array [1,2,3] was processed", "topics": ["tech"]}]'
        # Missing the final }

        # The regex from long_term_memory.py:89
        memories_match = re.search(r'"memories"\s*:\s*\[(.*?)\]', malformed, re.DOTALL)

        if memories_match:
            extracted = memories_match.group(1)
            # BUG: non-greedy .*? stops at the first ] — which is after "3"
            # instead of at the actual end of the array
            assert "]" not in extracted or "was processed" not in extracted, (
                "If this fails, the regex actually captured correctly (bug may be fixed)"
            )

    def test_greedy_regex_captures_full_array(self):
        """A greedy regex with balanced bracket counting would capture correctly."""
        malformed = '{"memories": [{"text": "array [1,2,3] was processed", "topics": ["tech"]}]'

        # Greedy approach: match from first [ to last ]
        memories_match = re.search(r'"memories"\s*:\s*\[(.*)\]', malformed, re.DOTALL)
        if memories_match:
            extracted = memories_match.group(1)
            assert "was processed" in extracted, (
                "Greedy regex should capture the full array content"
            )

    def test_valid_json_parses_without_regex_fallback(self):
        """Valid JSON with brackets in text should parse normally."""
        content = json.dumps({
            "memories": [
                {"text": "Python list [1, 2, 3] is a sequence type"},
                {"text": "Regex uses [a-z] for character classes"},
            ]
        })

        # Standard parser handles this fine
        parsed = json.loads(content)
        assert len(parsed["memories"]) == 2
        assert "[1, 2, 3]" in parsed["memories"][0]["text"]


# ============================================================================
# Finding #16 — get_working_memory() swallows all exceptions
# ============================================================================


class TestFinding16_WorkingMemorySwallowsExceptions:
    """working_memory.py:455-457 — The outer try/except catches Exception
    and returns None. Redis timeouts, JSON decode errors, etc. are silently
    treated as 'session not found' (404) instead of server error (503).
    """

    @pytest.mark.asyncio
    async def test_redis_timeout_returns_none_instead_of_raising(self):
        """A Redis timeout should raise, not silently return None."""
        from agent_memory_server.working_memory import get_working_memory

        mock_redis = AsyncMock()
        # Simulate a Redis timeout
        mock_redis.json.return_value.get.side_effect = ConnectionError(
            "Connection timed out"
        )

        with patch(
            "agent_memory_server.working_memory.get_redis_conn",
            return_value=mock_redis,
        ):
            result = await get_working_memory("test-session")

        # BUG: Returns None (interpreted as 404) instead of raising (503)
        # This test documents the current broken behavior
        assert result is None, (
            "Current behavior: ConnectionError is swallowed, returns None"
        )


# ============================================================================
# Finding #21 — PATCH can't clear optional fields to null
# ============================================================================


class TestFinding21_PatchCantClearFields:
    """api.py:907 — {k: v for k, v in updates.model_dump().items() if v is not None}
    filters out null values, making it impossible to clear optional fields.
    """

    def test_none_filtered_from_patch_dict(self):
        """Setting a field to None in a PATCH should clear it, but doesn't."""
        # Simulate what the API endpoint does
        updates = {
            "text": "Updated text",
            "stale_after": None,  # Intent: clear this field
            "source_user": "chris",
        }

        # BUG: The None-filtering removes "stale_after"
        filtered = {k: v for k, v in updates.items() if v is not None}

        assert "stale_after" not in filtered, (
            "Current behavior: None values are filtered, can't clear fields"
        )
        assert "text" in filtered
        assert "source_user" in filtered

    def test_explicit_sentinel_would_distinguish_missing_from_null(self):
        """Using UNSET sentinel would let us distinguish 'not provided'
        from 'explicitly set to null'."""
        _UNSET = object()

        # With sentinel pattern
        updates = {
            "text": "Updated text",
            "stale_after": None,  # Explicitly null → should clear
            "visibility": _UNSET,  # Not provided → should not change
        }

        # Filter out only UNSET, keep None
        filtered = {k: v for k, v in updates.items() if v is not _UNSET}

        assert "stale_after" in filtered  # Kept (explicit null)
        assert filtered["stale_after"] is None
        assert "visibility" not in filtered  # Filtered (not provided)
        assert "text" in filtered


# ============================================================================
# Finding #10 — Extraction marks messages as processed on LLM failure
# ============================================================================


class TestFinding10_ExtractionMarksProcessedOnFailure:
    """extraction.py:828-836 — When extraction fails, the memory is still
    marked as discrete_memory_extracted='t'. This means temporary LLM outages
    permanently lose those messages from extraction.
    """

    def test_failed_extraction_marks_as_processed(self):
        """Demonstrate: extraction failure still sets extracted='t'."""
        # Simulate the extraction error handling logic from extraction.py:828-836
        class MockMemory:
            def __init__(self, id, text, extracted="f"):
                self.id = id
                self.text = text
                self.discrete_memory_extracted = extracted

            def model_copy(self, update=None):
                copy = MockMemory(self.id, self.text, self.discrete_memory_extracted)
                if update:
                    for k, v in update.items():
                        setattr(copy, k, v)
                return copy

        memory = MockMemory("mem-1", "Grant loves basketball")
        all_updated_memories = []

        # Simulate the except block from extraction.py:831-836
        try:
            raise RuntimeError("LLM temporarily unavailable")
        except Exception:
            # BUG: Still marks as processed — permanently lost from extraction
            updated_memory = memory.model_copy(
                update={"discrete_memory_extracted": "t"}
            )
            all_updated_memories.append(updated_memory)

        assert all_updated_memories[0].discrete_memory_extracted == "t", (
            "BUG confirmed: failed extraction permanently marks message as extracted"
        )

    def test_correct_behavior_would_leave_unextracted(self):
        """Correct behavior: failed extraction should leave extracted='f'
        so it can be retried, with a retry counter to prevent infinite loops."""

        class MockMemory:
            def __init__(self, id, text, extracted="f", extract_attempts=0):
                self.id = id
                self.text = text
                self.discrete_memory_extracted = extracted
                self.extract_attempts = extract_attempts

            def model_copy(self, update=None):
                copy = MockMemory(
                    self.id, self.text,
                    self.discrete_memory_extracted,
                    self.extract_attempts,
                )
                if update:
                    for k, v in update.items():
                        setattr(copy, k, v)
                return copy

        MAX_EXTRACT_ATTEMPTS = 3
        memory = MockMemory("mem-1", "Grant loves basketball", extract_attempts=0)
        all_updated_memories = []

        # Correct pattern: increment retry counter, don't mark as extracted
        try:
            raise RuntimeError("LLM temporarily unavailable")
        except Exception:
            attempts = memory.extract_attempts + 1
            if attempts >= MAX_EXTRACT_ATTEMPTS:
                # Give up after max retries
                updated = memory.model_copy(update={
                    "discrete_memory_extracted": "t",
                    "extract_attempts": attempts,
                })
            else:
                # Leave as unextracted for retry
                updated = memory.model_copy(update={
                    "extract_attempts": attempts,
                })
            all_updated_memories.append(updated)

        # First failure: should NOT be marked as extracted
        assert all_updated_memories[0].discrete_memory_extracted == "f"
        assert all_updated_memories[0].extract_attempts == 1
