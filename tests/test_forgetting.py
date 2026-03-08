from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from agent_memory_server.long_term_memory import (
    _parse_stale_after,
    select_ids_for_forgetting,
)
from agent_memory_server.models import MemoryRecordResult, MemoryTypeEnum
from agent_memory_server.utils.recency import (
    rerank_with_recency,
    score_recency,
)


def make_result(
    id: str,
    text: str,
    dist: float,
    created_days_ago: int,
    accessed_days_ago: int,
    user_id: str | None = "u1",
    namespace: str | None = "ns1",
    stale_after: datetime | None = None,
    pinned: bool = False,
):
    now = datetime.now(UTC)
    return MemoryRecordResult(
        id=id,
        text=text,
        dist=dist,
        created_at=now - timedelta(days=created_days_ago),
        updated_at=now - timedelta(days=created_days_ago),
        last_accessed=now - timedelta(days=accessed_days_ago),
        user_id=user_id,
        session_id=None,
        namespace=namespace,
        topics=[],
        entities=[],
        memory_hash="",
        memory_type=MemoryTypeEnum.SEMANTIC,
        persisted_at=None,
        extracted_from=[],
        event_date=None,
        stale_after=stale_after,
        pinned=pinned,
    )


def default_params():
    return {
        "semantic_weight": 0.8,
        "recency_weight": 0.2,
        "freshness_weight": 0.6,
        "novelty_weight": 0.4,
        "half_life_last_access_days": 7.0,
        "half_life_created_days": 30.0,
    }


def test_score_recency_monotonicity_with_age():
    params = default_params()
    now = datetime.now(UTC)

    newer = make_result("a", "new", dist=0.5, created_days_ago=1, accessed_days_ago=1)
    older = make_result("b", "old", dist=0.5, created_days_ago=60, accessed_days_ago=60)

    r_new = score_recency(newer, now=now, params=params)
    r_old = score_recency(older, now=now, params=params)

    assert 0.0 <= r_new <= 1.0
    assert 0.0 <= r_old <= 1.0
    assert r_new > r_old


def test_rerank_with_recency_prefers_recent_when_similarity_close():
    params = default_params()
    now = datetime.now(UTC)

    # More similar but old
    old_more_sim = make_result(
        "old", "old", dist=0.05, created_days_ago=45, accessed_days_ago=45
    )
    # Less similar but fresh
    fresh_less_sim = make_result(
        "fresh", "fresh", dist=0.25, created_days_ago=0, accessed_days_ago=0
    )

    ranked = rerank_with_recency([old_more_sim, fresh_less_sim], now=now, params=params)

    # With the default modest recency weight, freshness should win when similarity is close
    assert ranked[0].id == "fresh"
    assert ranked[1].id == "old"


def test_rerank_with_recency_respects_semantic_weight_when_gap_large():
    # If semantic similarity difference is large, it should dominate
    params = default_params()
    params["semantic_weight"] = 0.9
    params["recency_weight"] = 0.1
    now = datetime.now(UTC)

    much_more_similar_old = make_result(
        "old", "old", dist=0.01, created_days_ago=90, accessed_days_ago=90
    )
    weak_similar_fresh = make_result(
        "fresh", "fresh", dist=0.6, created_days_ago=0, accessed_days_ago=0
    )

    ranked = rerank_with_recency(
        [weak_similar_fresh, much_more_similar_old], now=now, params=params
    )
    assert ranked[0].id == "old"


def test_select_ids_for_forgetting_ttl_and_inactivity():
    now = datetime.now(UTC)
    recent = make_result(
        "keep1", "recent", dist=0.3, created_days_ago=5, accessed_days_ago=2
    )
    old_but_active = make_result(
        "keep2", "old-but-active", dist=0.3, created_days_ago=60, accessed_days_ago=1
    )
    old_and_inactive = make_result(
        "del1", "old-inactive", dist=0.3, created_days_ago=60, accessed_days_ago=45
    )
    very_old = make_result(
        "del2", "very-old", dist=0.3, created_days_ago=400, accessed_days_ago=5
    )

    policy = {
        "max_age_days": 365 / 12,  # ~30 days
        "max_inactive_days": 30,
        "budget": None,  # no budget cap in this test
        "memory_type_allowlist": None,
    }

    to_delete = select_ids_for_forgetting(
        [recent, old_but_active, old_and_inactive, very_old],
        policy=policy,
        now=now,
        pinned_ids=set(),
    )
    # Both TTL and inactivity should catch different items
    assert set(to_delete) == {"del1", "del2"}


def test_select_ids_for_forgetting_budget_keeps_top_by_recency():
    now = datetime.now(UTC)

    # Create 5 results, with varying ages
    r1 = make_result("m1", "t", dist=0.3, created_days_ago=1, accessed_days_ago=1)
    r2 = make_result("m2", "t", dist=0.3, created_days_ago=5, accessed_days_ago=5)
    r3 = make_result("m3", "t", dist=0.3, created_days_ago=10, accessed_days_ago=10)
    r4 = make_result("m4", "t", dist=0.3, created_days_ago=20, accessed_days_ago=20)
    r5 = make_result("m5", "t", dist=0.3, created_days_ago=40, accessed_days_ago=40)

    policy = {
        "max_age_days": None,
        "max_inactive_days": None,
        "budget": 2,  # keep only 2 most recent by recency score, delete the rest
        "memory_type_allowlist": None,
    }

    to_delete = select_ids_for_forgetting(
        [r1, r2, r3, r4, r5], policy=policy, now=now, pinned_ids=set()
    )

    # Expect 3 deletions: the 3 least recent are deleted
    assert len(to_delete) == 3
    # The two most recent should be kept (m1, m2), so they should NOT be in delete set
    assert "m1" not in to_delete and "m2" not in to_delete


def test_select_ids_for_forgetting_respects_pinned_ids():
    now = datetime.now(UTC)
    r1 = make_result("m1", "t", dist=0.4, created_days_ago=1, accessed_days_ago=1)
    r2 = make_result("m2", "t", dist=0.4, created_days_ago=2, accessed_days_ago=2)
    r3 = make_result("m3", "t", dist=0.4, created_days_ago=30, accessed_days_ago=30)

    policy = {
        "max_age_days": None,
        "max_inactive_days": None,
        "budget": 1,
        "memory_type_allowlist": None,
    }

    to_delete = select_ids_for_forgetting(
        [r1, r2, r3], policy=policy, now=now, pinned_ids={"m1"}
    )

    # We must keep m1 regardless of budget; so m2/m3 compete for deletion, m3 is older and should be deleted
    assert "m1" not in to_delete
    assert "m3" in to_delete


# ---------------------------------------------------------------------------
# _parse_stale_after helper
# ---------------------------------------------------------------------------


def test_parse_stale_after_none():
    assert _parse_stale_after(None) is None


def test_parse_stale_after_datetime_aware():
    dt = datetime(2026, 6, 1, tzinfo=UTC)
    assert _parse_stale_after(dt) == dt


def test_parse_stale_after_datetime_naive():
    dt = datetime(2026, 6, 1)
    result = _parse_stale_after(dt)
    assert result is not None
    assert result.tzinfo is UTC
    assert result.year == 2026


def test_parse_stale_after_timestamp():
    ts = datetime(2026, 6, 1, tzinfo=UTC).timestamp()
    result = _parse_stale_after(ts)
    assert result is not None
    assert abs((result - datetime(2026, 6, 1, tzinfo=UTC)).total_seconds()) < 1


def test_parse_stale_after_iso_string():
    iso = "2026-06-01T00:00:00+00:00"
    result = _parse_stale_after(iso)
    assert result is not None
    assert result == datetime(2026, 6, 1, tzinfo=UTC)


def test_parse_stale_after_iso_string_naive():
    iso = "2026-06-01T00:00:00"
    result = _parse_stale_after(iso)
    assert result is not None
    assert result.tzinfo is UTC


def test_parse_stale_after_invalid_string():
    assert _parse_stale_after("not-a-date") is None


def test_parse_stale_after_unsupported_type():
    assert _parse_stale_after([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# stale_after policy in select_ids_for_forgetting
# ---------------------------------------------------------------------------

NO_TTL_POLICY = {
    "max_age_days": None,
    "max_inactive_days": None,
    "budget": None,
    "memory_type_allowlist": None,
}


def test_stale_after_deletes_expired_memory():
    """Memories past their stale_after datetime should be deleted."""
    now = datetime.now(UTC)
    stale = make_result(
        "stale1", "old event", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now - timedelta(hours=1),
    )
    fresh = make_result(
        "fresh1", "current", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now + timedelta(days=30),
    )
    no_stale = make_result(
        "none1", "no expiry", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
    )

    to_delete = select_ids_for_forgetting(
        [stale, fresh, no_stale], policy=NO_TTL_POLICY, now=now, pinned_ids=set()
    )
    assert set(to_delete) == {"stale1"}


def test_stale_after_respects_pinned_ids():
    """Pinned memories should never be deleted even if stale_after has passed."""
    now = datetime.now(UTC)
    stale_pinned = make_result(
        "pinned1", "important", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now - timedelta(days=1),
    )

    to_delete = select_ids_for_forgetting(
        [stale_pinned], policy=NO_TTL_POLICY, now=now, pinned_ids={"pinned1"}
    )
    assert "pinned1" not in to_delete


def test_stale_after_respects_pinned_field():
    """Memories with pinned=True on the record itself are exempt."""
    now = datetime.now(UTC)
    stale_pinned = make_result(
        "pinned2", "important", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now - timedelta(days=1),
        pinned=True,
    )

    to_delete = select_ids_for_forgetting(
        [stale_pinned], policy=NO_TTL_POLICY, now=now, pinned_ids=set()
    )
    assert "pinned2" not in to_delete


def test_stale_after_alongside_ttl():
    """stale_after is additive — both stale and TTL-expired items are caught."""
    now = datetime.now(UTC)
    stale_only = make_result(
        "stale1", "stale", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now - timedelta(hours=1),
    )
    ttl_only = make_result(
        "ttl1", "old", dist=0.3,
        created_days_ago=100, accessed_days_ago=100,
    )
    safe = make_result(
        "safe1", "fine", dist=0.3,
        created_days_ago=1, accessed_days_ago=1,
    )

    policy = {
        "max_age_days": 30,
        "max_inactive_days": None,
        "budget": None,
        "memory_type_allowlist": None,
    }

    to_delete = select_ids_for_forgetting(
        [stale_only, ttl_only, safe], policy=policy, now=now, pinned_ids=set()
    )
    assert set(to_delete) == {"stale1", "ttl1"}


def test_stale_after_disabled_via_config():
    """When stale_after_cleanup_enabled is False, stale memories are NOT deleted."""
    now = datetime.now(UTC)
    stale = make_result(
        "stale1", "stale", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now - timedelta(hours=1),
    )

    with patch(
        "agent_memory_server.long_term_memory.settings"
    ) as mock_settings:
        mock_settings.stale_after_cleanup_enabled = False
        to_delete = select_ids_for_forgetting(
            [stale], policy=NO_TTL_POLICY, now=now, pinned_ids=set()
        )
    assert "stale1" not in to_delete


def test_stale_after_exact_boundary():
    """When now == stale_after, the memory should be deleted (>= semantics)."""
    now = datetime.now(UTC)
    boundary = make_result(
        "edge1", "edge", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now,
    )

    to_delete = select_ids_for_forgetting(
        [boundary], policy=NO_TTL_POLICY, now=now, pinned_ids=set()
    )
    assert "edge1" in to_delete


def test_stale_after_future_not_deleted():
    """Memories with stale_after in the future should NOT be deleted."""
    now = datetime.now(UTC)
    future = make_result(
        "future1", "future", dist=0.3,
        created_days_ago=5, accessed_days_ago=1,
        stale_after=now + timedelta(days=7),
    )

    to_delete = select_ids_for_forgetting(
        [future], policy=NO_TTL_POLICY, now=now, pinned_ids=set()
    )
    assert to_delete == []
