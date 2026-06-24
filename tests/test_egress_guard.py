"""Unit tests for the read/egress-side bulk-exfiltration volume guard (C5,
utils/egress_guard.py) + its wiring in search_long_term_memories.

The pure functions (config validation, window key, boundary TTL, classify) need
no Redis. record_and_check is exercised with a lightweight fake async Redis
pipeline (mirrors the AsyncMock pattern in tests/test_content_security.py).

Run: uv run pytest tests/test_egress_guard.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

import agent_memory_server.long_term_memory as ltm
from agent_memory_server.utils.egress_guard import (
    EgressGuardConfig,
    EgressVerdict,
    _seconds_to_boundary,
    _window_key,
    classify_egress,
    config_from_settings,
    record_and_check,
)


def _cfg(enabled=True, window_seconds=60, max_records=100) -> EgressGuardConfig:
    return EgressGuardConfig(
        enabled=enabled, window_seconds=window_seconds, max_records=max_records
    )


class _FakePipeline:
    """Minimal async-execute Redis pipeline double. Records calls; returns the
    configured HINCRBY total as the first execute() result."""

    def __init__(self, hincrby_total: int):
        self._total = hincrby_total
        self.calls: list[tuple] = []

    def hincrby(self, key, field, amount):
        self.calls.append(("hincrby", key, field, amount))
        return self

    def expire(self, key, ttl):
        self.calls.append(("expire", key, ttl))
        return self

    async def execute(self):
        return [self._total, True]


def _fake_redis(hincrby_total: int) -> MagicMock:
    pipe = _FakePipeline(hincrby_total)
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    redis._pipe = pipe  # exposed for assertions
    return redis


# --- config_from_settings (fail-safe flooring) -----------------------------


def test_config_from_settings_floors_invalid_values():
    with (
        patch.object(ltm.settings, "memory_egress_guard_enabled", True),
        patch.object(ltm.settings, "memory_egress_guard_window_seconds", 0),
        patch.object(ltm.settings, "memory_egress_guard_max_records", 0),
    ):
        cfg = config_from_settings()
    assert cfg.enabled is True
    assert cfg.window_seconds == 1  # floored from 0 (no divide-by-zero window key)
    assert cfg.max_records == 1  # floored from 0


def test_config_from_settings_passthrough():
    with (
        patch.object(ltm.settings, "memory_egress_guard_enabled", False),
        patch.object(ltm.settings, "memory_egress_guard_window_seconds", 30),
        patch.object(ltm.settings, "memory_egress_guard_max_records", 500),
    ):
        cfg = config_from_settings()
    assert cfg == EgressGuardConfig(enabled=False, window_seconds=30, max_records=500)


# --- _window_key / _seconds_to_boundary (pure) -----------------------------


def test_window_key_is_global_and_bucketed():
    # Fixed buckets align to multiples of window_seconds: bucket 16 = [960,1020).
    # Same bucket within a window; different bucket across the boundary.
    assert _window_key(960.0, 60) == _window_key(1019.9, 60)
    assert _window_key(1019.9, 60) != _window_key(1020.0, 60)
    assert _window_key(1000.0, 60).startswith("memory_egress:window:")
    # No per-consumer component — one global counter. 1000 // 60 == 16.
    assert _window_key(1000.0, 60) == "memory_egress:window:16"


def test_seconds_to_boundary_positive_with_grace():
    # At the exact boundary the TTL is window+1 (never EXPIRE 0 on a live key).
    assert _seconds_to_boundary(1200.0, 60) == 61
    # Mid-window: remaining + 1s grace.
    assert _seconds_to_boundary(1230.0, 60) == 31
    assert _seconds_to_boundary(1259.0, 60) == 2


# --- classify_egress (pure verdict) ----------------------------------------


def test_classify_under_threshold_is_ok():
    v = classify_egress(window_total=50, this_request=10, config=_cfg(max_records=100))
    assert v.outcome == "ok"
    assert v.reason is None
    assert v.window_total == 50


def test_classify_at_threshold_is_ok_strict_greater_than():
    # Strictly greater-than: equal to the cap does NOT flag.
    v = classify_egress(window_total=100, this_request=1, config=_cfg(max_records=100))
    assert v.outcome == "ok"


def test_classify_over_threshold_flags_with_reason():
    v = classify_egress(window_total=101, this_request=1, config=_cfg(max_records=100))
    assert v.outcome == "flagged"
    assert v.reason is not None
    assert "101" in v.reason and "100" in v.reason


# --- record_and_check (async, fake redis) ----------------------------------


@pytest.mark.asyncio
async def test_record_and_check_disabled_is_noop():
    redis = _fake_redis(99999)
    v = await record_and_check(redis, 500, config=_cfg(enabled=False), now=1000.0)
    assert v.outcome == "ok"
    assert v.window_total == 0
    redis.pipeline.assert_not_called()  # no Redis round-trip when disabled


@pytest.mark.asyncio
async def test_record_and_check_zero_count_is_noop():
    redis = _fake_redis(99999)
    v = await record_and_check(redis, 0, config=_cfg(), now=1000.0)
    assert v.outcome == "ok"
    redis.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_record_and_check_increments_and_sets_ttl():
    redis = _fake_redis(hincrby_total=40)
    v = await record_and_check(
        redis, 40, config=_cfg(window_seconds=60, max_records=100), now=1000.0
    )
    assert v.outcome == "ok"
    assert v.window_total == 40
    assert v.this_request == 40
    calls = redis._pipe.calls
    assert calls[0] == ("hincrby", "memory_egress:window:16", "records", 40)
    assert calls[1][0] == "expire"
    assert calls[1][2] == _seconds_to_boundary(1000.0, 60)


@pytest.mark.asyncio
async def test_record_and_check_flags_when_window_exceeds_cap():
    # The post-increment window total (from HINCRBY) is what's compared, so a
    # single oversized pull can trip the cap on its own request.
    redis = _fake_redis(hincrby_total=2500)
    v = await record_and_check(redis, 2500, config=_cfg(max_records=2000), now=1000.0)
    assert v.outcome == "flagged"
    assert v.window_total == 2500


@pytest.mark.asyncio
async def test_record_and_check_fails_open_on_redis_error():
    redis = MagicMock()
    bad_pipe = MagicMock()
    bad_pipe.hincrby = MagicMock(return_value=bad_pipe)
    bad_pipe.expire = MagicMock(return_value=bad_pipe)
    bad_pipe.execute = AsyncMock(side_effect=RedisError("boom"))
    redis.pipeline = MagicMock(return_value=bad_pipe)
    v = await record_and_check(redis, 5000, config=_cfg(max_records=10), now=1000.0)
    # Fail-open: a Redis error returns ok, never flagged/blocked.
    assert v.outcome == "ok"
    assert v.window_total == 0


# --- wiring: _observe_recall_egress never breaks recall --------------------


@pytest.mark.asyncio
async def test_observe_recall_egress_flagged_emits_telemetry():
    with (
        patch.object(
            ltm, "egress_config_from_settings", return_value=_cfg(max_records=10)
        ),
        patch.object(ltm, "get_redis_conn", AsyncMock(return_value=_fake_redis(50))),
        patch.object(
            ltm,
            "egress_record_and_check",
            AsyncMock(
                return_value=EgressVerdict(
                    outcome="flagged", window_total=50, this_request=50, reason="over"
                )
            ),
        ),
        patch.object(ltm, "record_counter") as counter,
    ):
        await ltm._observe_recall_egress(50)
    counter.assert_called_once()
    assert counter.call_args.args[0] == "memory_server.egress_guard.flagged"


@pytest.mark.asyncio
async def test_observe_recall_egress_swallows_errors():
    # Even a get_redis_conn explosion must not propagate out of recall.
    with (
        patch.object(ltm, "egress_config_from_settings", return_value=_cfg()),
        patch.object(
            ltm, "get_redis_conn", AsyncMock(side_effect=RuntimeError("redis down"))
        ),
    ):
        # Must not raise.
        await ltm._observe_recall_egress(5)


@pytest.mark.asyncio
async def test_observe_recall_egress_disabled_skips_redis():
    with (
        patch.object(
            ltm, "egress_config_from_settings", return_value=_cfg(enabled=False)
        ),
        patch.object(ltm, "get_redis_conn", AsyncMock()) as conn,
    ):
        await ltm._observe_recall_egress(9999)
    conn.assert_not_called()


# --- exempt: trusted internal full-corpus caller bypasses the window (LAB-388)


@pytest.mark.asyncio
async def test_observe_recall_egress_exempt_caller_does_not_trip_guard():
    """A trusted internal full-corpus enumeration (curator backup,
    bypass_recall_filters=True → exempt=True) must NOT touch the shared window:
    no config read, no Redis round-trip, no flag/telemetry — even for a record
    count that would massively exceed the cap. This is the LAB-388 fix: the
    nightly backup drains ~10k+ records and was poisoning the global counter."""
    with (
        patch.object(ltm, "egress_config_from_settings") as cfg,
        patch.object(ltm, "get_redis_conn", AsyncMock()) as conn,
        patch.object(ltm, "egress_record_and_check", AsyncMock()) as check,
        patch.object(ltm, "record_counter") as counter,
    ):
        await ltm._observe_recall_egress(10_396, exempt=True)
    # Short-circuits before any work — the trusted caller never enters the window.
    cfg.assert_not_called()
    conn.assert_not_called()
    check.assert_not_called()
    counter.assert_not_called()


@pytest.mark.asyncio
async def test_observe_recall_egress_nonexempt_burst_still_flags():
    """The exemption is keyed strictly on the trusted-caller flag, so the
    real-exfil detection floor is unchanged: a NON-exempt caller (a regular
    recall, or a compromised plugin offset-walking the corpus WITHOUT the
    bypass flag) whose burst exceeds the cap STILL flags. Asserts the default
    exempt=False path is unaffected by the new parameter."""
    with (
        patch.object(
            ltm, "egress_config_from_settings", return_value=_cfg(max_records=2000)
        ),
        patch.object(
            ltm, "get_redis_conn", AsyncMock(return_value=_fake_redis(10_396))
        ),
        patch.object(
            ltm,
            "egress_record_and_check",
            AsyncMock(
                return_value=EgressVerdict(
                    outcome="flagged",
                    window_total=10_396,
                    this_request=10_396,
                    reason="over",
                )
            ),
        ),
        patch.object(ltm, "record_counter") as counter,
    ):
        # exempt defaults to False — the non-exempt drain still trips the guard.
        await ltm._observe_recall_egress(10_396)
    counter.assert_called_once()
    assert counter.call_args.args[0] == "memory_server.egress_guard.flagged"
