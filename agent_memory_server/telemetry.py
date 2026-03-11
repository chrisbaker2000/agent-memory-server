"""
Lightweight OTLP telemetry for the memory server.

Emits metrics and traces to SigNoz via OTLP HTTP (localhost:4318).
Uses direct HTTP POST — no OpenTelemetry SDK dependency required.

Instrumented operations:
  - Embedding generation (aembed_documents, aembed_query)
  - Memory search (search_memories — vector, text, hybrid)
  - Memory store (add_memories)
  - Background task tracking (pending asyncio tasks)

All metrics use the "memory_server." prefix to distinguish from the
external metrics-collector's "memory." prefix metrics.

Usage:
    from agent_memory_server.telemetry import record_metric, record_histogram

    record_metric("memory_server.search.count", 1, {"search_type": "hybrid"})
    record_histogram("memory_server.search.duration_ms", 42.5, {"search_type": "hybrid"})
"""

import logging
import os
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OTLP_ENDPOINT = os.environ.get(
    "OTLP_METRICS_ENDPOINT", "http://localhost:4318/v1/metrics"
)
SERVICE_NAME = "memory-server"
HOSTNAME = socket.gethostname()
FLUSH_INTERVAL = 30  # seconds
MAX_BUFFER_SIZE = 200

# Enable/disable via env var (default: enabled)
TELEMETRY_ENABLED = os.environ.get("MEMORY_SERVER_TELEMETRY", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Resource attributes (consistent with monitoring_lib)
# ---------------------------------------------------------------------------

RESOURCE_ATTRS = [
    {"key": "service.name", "value": {"stringValue": SERVICE_NAME}},
    {"key": "host.name", "value": {"stringValue": HOSTNAME}},
    {"key": "host.arch", "value": {"stringValue": "arm64"}},
    {"key": "os.type", "value": {"stringValue": "darwin"}},
    {"key": "deployment.environment", "value": {"stringValue": "homelab"}},
]

# ---------------------------------------------------------------------------
# Metric buffer and flush thread
# ---------------------------------------------------------------------------

_metric_buffer: list[dict] = []
_buffer_lock = threading.Lock()
_flush_thread: threading.Thread | None = None
_running = False


def _now_ns() -> str:
    """Current time as Unix nanoseconds string."""
    return str(int(time.time() * 1_000_000_000))


def _attrs(attributes: dict[str, Any] | None) -> list[dict]:
    """Convert a dict to OTLP attribute array."""
    if not attributes:
        return []
    result = []
    for k, v in attributes.items():
        if isinstance(v, bool):
            result.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, (int, float)):
            result.append({"key": k, "value": {"doubleValue": v}})
        else:
            result.append({"key": k, "value": {"stringValue": str(v)}})
    return result


def record_metric(
    name: str,
    value: float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a gauge metric data point.

    Buffered and flushed every FLUSH_INTERVAL seconds.
    """
    if not TELEMETRY_ENABLED:
        return

    ts = _now_ns()
    metric = {
        "name": name,
        "unit": "{count}",
        "gauge": {
            "dataPoints": [
                {
                    "asDouble": value,
                    "timeUnixNano": ts,
                    "attributes": _attrs(attributes),
                }
            ],
        },
    }

    with _buffer_lock:
        _metric_buffer.append(metric)
        if len(_metric_buffer) >= MAX_BUFFER_SIZE:
            _flush_locked()


def record_histogram(
    name: str,
    value: float,
    attributes: dict[str, Any] | None = None,
    unit: str = "ms",
) -> None:
    """Record a histogram-style metric as a gauge data point.

    Since we're using lightweight OTLP HTTP without the SDK,
    we emit gauge data points that SigNoz can aggregate into
    histograms via ClickHouse queries (avg, p50, p95, max).
    """
    if not TELEMETRY_ENABLED:
        return

    ts = _now_ns()
    metric = {
        "name": name,
        "unit": unit,
        "gauge": {
            "dataPoints": [
                {
                    "asDouble": value,
                    "timeUnixNano": ts,
                    "attributes": _attrs(attributes),
                }
            ],
        },
    }

    with _buffer_lock:
        _metric_buffer.append(metric)


def record_counter(
    name: str,
    value: float = 1.0,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a counter increment as a sum data point."""
    if not TELEMETRY_ENABLED:
        return

    ts = _now_ns()
    metric = {
        "name": name,
        "unit": "{count}",
        "sum": {
            "dataPoints": [
                {
                    "asDouble": value,
                    "timeUnixNano": ts,
                    "startTimeUnixNano": ts,
                    "attributes": _attrs(attributes),
                }
            ],
            "aggregationTemporality": 1,  # DELTA
            "isMonotonic": True,
        },
    }

    with _buffer_lock:
        _metric_buffer.append(metric)


def _flush_locked() -> None:
    """Flush buffered metrics to OTLP endpoint. Must hold _buffer_lock."""
    if not _metric_buffer:
        return

    metrics = list(_metric_buffer)
    _metric_buffer.clear()

    # Release lock before HTTP call
    payload = {
        "resourceMetrics": [
            {
                "resource": {"attributes": RESOURCE_ATTRS},
                "scopeMetrics": [
                    {
                        "scope": {
                            "name": "memory-server-telemetry",
                            "version": "1.0.0",
                        },
                        "metrics": metrics,
                    }
                ],
            }
        ],
    }

    try:
        import requests

        resp = requests.post(
            OTLP_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code not in (200, 202):
            logger.debug(
                "Telemetry flush error: HTTP %s - %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logger.debug("Telemetry flush error: %s", e)


def flush() -> None:
    """Manually flush buffered metrics."""
    with _buffer_lock:
        _flush_locked()


def _flush_loop() -> None:
    """Background thread that periodically flushes metrics."""
    global _running
    while _running:
        time.sleep(FLUSH_INTERVAL)
        with _buffer_lock:
            _flush_locked()


def start() -> None:
    """Start the background flush thread."""
    global _flush_thread, _running
    if not TELEMETRY_ENABLED:
        return
    if _running:
        return
    _running = True
    _flush_thread = threading.Thread(target=_flush_loop, daemon=True, name="telemetry-flush")
    _flush_thread.start()
    logger.info("Memory server telemetry started (endpoint: %s)", OTLP_ENDPOINT)


def stop() -> None:
    """Stop the background flush thread and flush remaining metrics."""
    global _running
    _running = False
    flush()
    logger.info("Memory server telemetry stopped")


# ---------------------------------------------------------------------------
# Instrumentation decorators / context managers
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager for timing operations."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.duration_ms = (time.monotonic() - self.start_time) * 1000
