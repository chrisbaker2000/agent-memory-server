"""SummaryView refresh reuse + internal egress exemption (2026-10-10).

openclaw-redis-agent-memory triggers `POST /v1/summary-views/{id}/run` after
every saved turn: ~400 runs in two days, each fetching ~5.3K memories and
re-running the summarization model over an unchanged set, and each fetch
flagged the recall egress guard. These tests lock:
  - unchanged partitions reuse the stored summary (no model call);
  - any input change (add, delete, edit, updated_at) recomputes;
  - a failed model call never replaces an earlier good summary;
  - the refresh's internal fetch is egress-exempt, while a plain recall is not.

No Redis required: storage, fetch and the model call are monkeypatched.

Run: uv run pytest tests/test_summary_view_refresh_reuse.py -v
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
from agent_memory_server import summary_views as sv
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    SummaryView,
    SummaryViewPartitionResult,
)


T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def _mem(i: int, text: str | None = None, updated: datetime = T0) -> MemoryRecord:
    return MemoryRecord(
        id=f"m{i}", text=text or f"memory {i}", namespace="claw", updated_at=updated
    )


VIEW = SummaryView(
    id="view-1",
    name="baker_family_summary",
    source="long_term",
    group_by=["namespace"],
    filters={"namespace": "claw"},
    time_window_days=90,
    continuous=False,
    prompt=None,
    model_name=None,
)


class TestPartitionFingerprint:
    def test_order_independent_and_stable(self):
        a = [_mem(1), _mem(2), _mem(3)]
        assert sv._partition_fingerprint(a) == sv._partition_fingerprint(
            list(reversed(a))
        )

    @pytest.mark.parametrize(
        "changed",
        [
            [_mem(1), _mem(2)],  # deletion
            [_mem(1), _mem(2), _mem(3), _mem(4)],  # addition
            [_mem(1), _mem(2), _mem(3, text="edited")],  # text edit
            [_mem(1), _mem(2), _mem(3, updated=T0 + timedelta(seconds=1))],  # touch
        ],
    )
    def test_any_input_change_changes_it(self, changed):
        base = [_mem(1), _mem(2), _mem(3)]
        assert sv._partition_fingerprint(base) != sv._partition_fingerprint(changed)

    def test_empty_partition_has_a_fingerprint(self):
        assert len(sv._partition_fingerprint([])) == 64


@pytest.fixture
def harness(monkeypatch):
    """Wire refresh_summary_view to in-memory storage and a counting model."""
    store: dict[str, SummaryViewPartitionResult] = {}
    state = {"memories": [_mem(1), _mem(2), _mem(3)], "model_calls": 0, "fail": False}

    async def fake_get_view(view_id):
        return VIEW

    async def fake_fetch(view, extra_group=None, page_size=1000, overall_limit=None):
        return list(state["memories"])

    async def fake_list(view_id, group_filter=None):
        return list(store.values())

    async def fake_save(result):
        store[sv.encode_partition_key(result.group)] = result

    async def fake_summarize(view, group, memories):
        state["model_calls"] += 1
        return SummaryViewPartitionResult(
            view_id=view.id,
            group=group,
            summary=sv.SUMMARY_FAILED_TEXT
            if state["fail"]
            else f"summary #{state['model_calls']}",
            memory_count=len(memories),
            source_fingerprint=sv._partition_fingerprint(memories),
        )

    monkeypatch.setattr(sv, "get_summary_view", fake_get_view)
    monkeypatch.setattr(sv, "_fetch_long_term_memories_for_view", fake_fetch)
    monkeypatch.setattr(sv, "list_partition_results", fake_list)
    monkeypatch.setattr(sv, "save_partition_result", fake_save)
    monkeypatch.setattr(sv, "summarize_partition_long_term", fake_summarize)
    return store, state


def _summary(store):
    (only,) = store.values()
    return only.summary


@pytest.mark.asyncio
class TestRefreshReuse:
    async def test_unchanged_partition_skips_the_model(self, harness):
        store, state = harness
        await sv.refresh_summary_view("view-1")
        await sv.refresh_summary_view("view-1")
        await sv.refresh_summary_view("view-1")
        assert state["model_calls"] == 1
        assert _summary(store) == "summary #1"

    async def test_changed_partition_recomputes(self, harness):
        store, state = harness
        await sv.refresh_summary_view("view-1")
        state["memories"].append(_mem(4))
        await sv.refresh_summary_view("view-1")
        assert state["model_calls"] == 2
        assert _summary(store) == "summary #2"
        assert next(iter(store.values())).memory_count == 4

    async def test_failed_model_call_keeps_the_previous_good_summary(self, harness):
        store, state = harness
        await sv.refresh_summary_view("view-1")
        state["memories"].append(_mem(4))
        state["fail"] = True
        await sv.refresh_summary_view("view-1")
        assert _summary(store) == "summary #1"  # not replaced by the fallback text
        # The stale fingerprint stays, so the next refresh retries the model.
        state["fail"] = False
        await sv.refresh_summary_view("view-1")
        assert state["model_calls"] == 3
        assert _summary(store) == "summary #3"

    async def test_failure_summary_is_never_reused(self, harness):
        store, state = harness
        state["fail"] = True
        await sv.refresh_summary_view("view-1")  # no previous → fallback stored
        assert _summary(store) == sv.SUMMARY_FAILED_TEXT
        state["fail"] = False
        await sv.refresh_summary_view("view-1")  # same inputs, but must retry
        assert state["model_calls"] == 2
        assert _summary(store) == "summary #2"

    async def test_results_stored_before_this_change_are_recomputed_once(self, harness):
        store, state = harness
        legacy = SummaryViewPartitionResult(
            view_id="view-1", group={"namespace": "claw"}, summary="old", memory_count=3
        )
        store[sv.encode_partition_key(legacy.group)] = legacy  # no fingerprint
        await sv.refresh_summary_view("view-1")
        await sv.refresh_summary_view("view-1")
        assert state["model_calls"] == 1


@pytest.mark.asyncio
async def test_view_fetch_is_egress_exempt(monkeypatch):
    seen: list[dict] = []

    async def fake_search(**kwargs):
        seen.append(kwargs)
        return MemoryRecordResults(total=0, memories=[], next_offset=None)

    monkeypatch.setattr(sv.long_term_memory, "search_long_term_memories", fake_search)
    await sv._fetch_long_term_memories_for_view(VIEW)
    assert seen and all(call["egress_exempt"] is True for call in seen)


class _ListingDB:
    async def list_memories(self, **kw):
        rows = [MemoryRecordResult(**_mem(i).model_dump(), dist=0.0) for i in (1, 2)]
        return MemoryRecordResults(total=2, next_offset=None, memories=rows)

    async def search_memories(self, query, **kw):
        return await self.list_memories()


@pytest.mark.asyncio
@pytest.mark.parametrize(("egress_exempt", "expected"), [(True, True), (False, False)])
async def test_search_egress_exempt_reaches_the_guard(egress_exempt, expected):
    observed = AsyncMock()
    with (
        patch.object(ltm, "get_memory_vector_db", AsyncMock(return_value=_ListingDB())),
        patch.object(ltm, "_observe_recall_egress", observed),
    ):
        await ltm.search_long_term_memories(
            text="", limit=10, egress_exempt=egress_exempt
        )
    observed.assert_awaited_once()
    assert observed.await_args.kwargs["exempt"] is expected
