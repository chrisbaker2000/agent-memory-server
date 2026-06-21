"""Read/egress-side bulk-exfiltration volume guard (C5).

The read-side counterpart to the write-time content-security pass
(:mod:`agent_memory_server.utils.content_security`). Port of wfr-memory-commons
``utils/egress_dlp.py``, reduced to the single-tenant homelab:

- **Single GLOBAL fixed-window counter** — the upstream per-consumer keying
  (``quota:{consumer}:egress:...``) is dropped; the homelab memory server is
  single-tenant, so there is one window counter for the whole server.
- **Counts total records returned** — the upstream ``data_classification``
  sensitive sub-limit is dropped (homelab memory records carry no classification
  band). One dimension: how many records recall handed back.
- **Detect-only** — flags + logs + emits telemetry; it never blocks a recall
  (no ``429``). Enforcement is a deliberate later step once a baseline is
  observed, exactly as the C1/relevance-gate rollouts were staged.
- **Fail-open** — any Redis error logs a warning and returns an ``ok`` verdict.
  This is defense-in-depth, not the access gate (visibility remains the gate); a
  Redis blip must never turn into a recall outage.

Why a cumulative window and not a per-request cap: per-request ``limit`` is
already small/bounded, so the exfiltration vector is *cross-request cumulative
volume* — a compromised plugin or a runaway loop walking ``offset`` / repeatedly
pulling to drain the corpus. A request-frequency limiter does not cap *records
returned* (N searches/min x limit each stays under any request-rate bucket while
draining thousands of records). This module closes that dimension.

The window is a hard FIXED window (``now // window_seconds``), not sliding, so a
burst straddling a bucket boundary can return up to ~2x the threshold across two
adjacent windows before either flags. Tune the threshold with that ~2x boundary
burst in mind. Everything pure here is unit-tested in tests/test_egress_guard.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import structlog
from redis.exceptions import RedisError

from agent_memory_server.config import settings


logger = structlog.get_logger(__name__)

# Redis key prefix for the global fixed-window egress counter. A hash so a future
# port can re-introduce sub-dimension fields (e.g. a sensitivity band) without a
# keyspace change — for now only the ``records`` field is used.
_KEY_PREFIX = "memory_egress:window"
_RECORDS_FIELD = "records"


@dataclass(frozen=True)
class EgressGuardConfig:
    """Resolved, validated egress-guard configuration.

    :ivar enabled: master switch. When False the guard is a no-op (verdict ``ok``).
    :ivar window_seconds: fixed-window width. >= 1 (validated).
    :ivar max_records: soft threshold — a window total strictly greater than this
        flags. >= 1 (validated).
    """

    enabled: bool
    window_seconds: int
    max_records: int


@dataclass(frozen=True)
class EgressVerdict:
    """Outcome of an egress-volume check.

    :ivar outcome: ``ok`` (under threshold, disabled, or fail-open) or ``flagged``
        (window total crossed the soft threshold — detect-only, not blocked).
    :ivar window_total: post-increment record count in the current window.
    :ivar this_request: records contributed by this request.
    :ivar reason: which threshold tripped, for the log/telemetry; ``None`` if ok.
    """

    outcome: Literal["ok", "flagged"]
    window_total: int
    this_request: int
    reason: str | None = None


def config_from_settings() -> EgressGuardConfig:
    """Build a validated config from global settings, FAIL-SAFE.

    Coerces ``window_seconds`` and ``max_records`` to >= 1 so a misconfigured
    value (0 / negative) can never produce a divide-by-zero window key or a
    threshold that flags on the first record. Validation only floors values;
    it never raises (the caller is on the recall hot path).
    """
    window_seconds = max(1, int(settings.memory_egress_guard_window_seconds))
    max_records = max(1, int(settings.memory_egress_guard_max_records))
    return EgressGuardConfig(
        enabled=bool(settings.memory_egress_guard_enabled),
        window_seconds=window_seconds,
        max_records=max_records,
    )


def _window_key(now: float, window_seconds: int) -> str:
    """Fixed-window counter key for the given instant. Global (no consumer)."""
    bucket = int(now // window_seconds)
    return f"{_KEY_PREFIX}:{bucket}"


def _seconds_to_boundary(now: float, window_seconds: int) -> int:
    """Seconds until the current fixed-window bucket ends (+1s grace).

    Used as the key TTL so each bucket self-expires shortly after it closes — no
    sweeper needed. The +1 grace guarantees a positive EXPIRE even at the exact
    boundary (``EXPIRE key 0`` would delete the key we just incremented).
    """
    elapsed = now % window_seconds
    return int(window_seconds - elapsed) + 1


def classify_egress(
    window_total: int, this_request: int, config: EgressGuardConfig
) -> EgressVerdict:
    """Pure verdict: flag when the post-increment window total exceeds the soft
    threshold. Detect-only — there is no ``blocked`` outcome in this rollout.
    """
    if window_total > config.max_records:
        return EgressVerdict(
            outcome="flagged",
            window_total=window_total,
            this_request=this_request,
            reason=(
                f"recall egress volume {window_total} records in "
                f"{config.window_seconds}s window exceeds soft cap {config.max_records}"
            ),
        )
    return EgressVerdict(
        outcome="ok", window_total=window_total, this_request=this_request
    )


async def record_and_check(
    redis,
    record_count: int,
    *,
    config: EgressGuardConfig,
    now: float | None = None,
) -> EgressVerdict:
    """Add this request's record count to the current fixed window and classify.

    HINCRBY + EXPIRE in one pipeline (the increment is included in the totals, so
    a single oversized pull can trip the threshold on its own request). FAIL-OPEN:
    any ``RedisError`` logs a warning and returns ``ok`` — the guard is
    defense-in-depth, never the gate.

    A disabled guard or a non-positive ``record_count`` is a no-op ``ok`` (no
    Redis round-trip).
    """
    if not config.enabled or record_count <= 0:
        return EgressVerdict(
            outcome="ok", window_total=0, this_request=max(0, record_count)
        )

    ts = time.time() if now is None else now
    key = _window_key(ts, config.window_seconds)
    try:
        pipe = redis.pipeline()
        pipe.hincrby(key, _RECORDS_FIELD, record_count)
        pipe.expire(key, _seconds_to_boundary(ts, config.window_seconds))
        results = await pipe.execute()
        window_total = int(results[0])
    except RedisError as exc:
        # Fail-open: a Redis blip must not break recall. Defense-in-depth only.
        logger.warning(
            "egress guard fail-open (redis error)",
            error=str(exc),
            record_count=record_count,
        )
        return EgressVerdict(outcome="ok", window_total=0, this_request=record_count)

    return classify_egress(window_total, record_count, config)
