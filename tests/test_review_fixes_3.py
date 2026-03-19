"""Tests for telemetry.py findings from code review (2026-03-19).

Covers:
  Finding #41: httpx.Client recreated every flush
  Finding #42: start() TOCTOU race on _running flag
  Finding #74: stop() no thread join
"""

import inspect
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Finding #41: httpx.Client recreated every flush
# ===========================================================================


class TestFinding41_HttpxClientReuse:
    """_send_metrics creates a new httpx.Client on every flush, paying TLS
    handshake and connection setup costs each time.

    Fix: Create client in start(), reuse in _send_metrics, close in stop()."""

    def test_send_metrics_reuses_persistent_client(self):
        """_send_metrics should use _client (persistent) as primary path,
        not create a new httpx.Client every call."""
        from agent_memory_server.telemetry import _send_metrics

        source = inspect.getsource(_send_metrics)
        # After fix: should reference _client for reuse
        assert "_client" in source, (
            "_send_metrics does not reference _client for reuse."
        )
        # Should NOT have `with httpx.Client(...) as client:` (the old pattern)
        assert "with httpx.Client(" not in source, (
            "_send_metrics still uses `with httpx.Client()` context manager "
            "which creates and destroys a client on every call."
        )

    def test_module_has_persistent_client_variable(self):
        """Module should have a _client variable for reuse."""
        from agent_memory_server import telemetry

        assert hasattr(telemetry, "_client"), (
            "telemetry module missing _client variable for httpx.Client reuse."
        )

    def test_start_creates_client(self):
        """start() should create the httpx.Client."""
        from agent_memory_server import telemetry

        # Save original state
        orig_running = telemetry._running
        orig_thread = telemetry._flush_thread
        orig_client = getattr(telemetry, "_client", None)

        try:
            telemetry._running = False
            telemetry._flush_thread = None
            telemetry._client = None

            with patch.object(telemetry, "TELEMETRY_ENABLED", True):
                telemetry.start()
                assert telemetry._client is not None, (
                    "start() did not create _client."
                )
                telemetry.stop()
        finally:
            telemetry._running = orig_running
            telemetry._flush_thread = orig_thread
            telemetry._client = orig_client

    def test_stop_closes_client(self):
        """stop() should close the httpx.Client."""
        from agent_memory_server import telemetry

        mock_client = MagicMock()
        orig_client = getattr(telemetry, "_client", None)
        orig_running = telemetry._running

        try:
            telemetry._client = mock_client
            telemetry._running = True
            telemetry.stop()
            mock_client.close.assert_called_once()
        finally:
            telemetry._client = orig_client
            telemetry._running = orig_running


# ===========================================================================
# Finding #42: start() TOCTOU race
# ===========================================================================


class TestFinding42_StartToctouRace:
    """start() checks _running then sets it without holding _buffer_lock.
    Two concurrent start() calls could both see _running=False and create
    two flush threads.

    Fix: Guard check-and-set with _buffer_lock."""

    def test_start_uses_lock(self):
        """start() should acquire _buffer_lock before checking _running."""
        from agent_memory_server.telemetry import start

        source = inspect.getsource(start)
        # After fix, should use _buffer_lock to guard the check-and-set
        assert "_buffer_lock" in source, (
            "start() does not use _buffer_lock to guard the _running "
            "check-and-set. Two concurrent start() calls could create "
            "duplicate flush threads."
        )


# ===========================================================================
# Finding #74: stop() no thread join
# ===========================================================================


class TestFinding74_StopNoThreadJoin:
    """stop() sets _running=False and flushes, but never joins the flush
    thread. The thread may still be mid-flush when the process exits."""

    def test_stop_joins_thread(self):
        """stop() source should call _flush_thread.join()."""
        from agent_memory_server.telemetry import stop

        source = inspect.getsource(stop)
        assert "join(" in source, (
            "stop() does not call _flush_thread.join(). "
            "The flush thread may outlive the stop() call."
        )
