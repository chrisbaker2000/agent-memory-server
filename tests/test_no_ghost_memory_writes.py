"""Background writes must never resurrect a deleted memory as a partial "ghost" hash.

HSET / HINCRBY on a missing key CREATES it. Three writers touched memory keys after the
fact, unconditionally:

- extract_memory_structure: a background task that writes topics/entities seconds after
  the store. A memory deleted in between came back as {topics, entities}, with no
  text / source_user / visibility (reproduced live 2026-10-09 by the memory smoke test
  and E2E section 26, which store a memory, delete it, and leave the ghost behind).
- update_last_accessed: a missing memory's HGET is None and read as "never accessed",
  so it got HSET last_accessed + HINCRBY access_count (the April Finding04 ghost).
- deduplicate_by_hash: HSET last_accessed on a search hit that could already be gone.

All three now go through one atomic Lua helper that writes only while the hash still has
its `text`. fakeredis runs the Lua (lupa), so these tests are hermetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import fakeredis
import pytest

from agent_memory_server.long_term_memory import (
    extract_memory_structure,
    update_last_accessed,
    update_memory_if_present,
)
from agent_memory_server.models import MemoryRecord
from agent_memory_server.utils.keys import Keys


@pytest.fixture()
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=False)


async def _live_memory(redis, mid: str) -> str:
    key = Keys.memory_key(mid)
    await redis.hset(
        key,
        mapping={
            "text": "Chris likes tea",
            "source_user": "Chris Baker",
            "visibility": "everyone",
        },
    )
    return key


# --- the helper -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_updates_a_live_memory(redis):
    key = await _live_memory(redis, "live-1")
    assert (
        await update_memory_if_present(
            redis, key, {"topics": "food", "entities": "Chris"}
        )
        is True
    )
    stored = await redis.hgetall(key)
    assert stored[b"topics"] == b"food" and stored[b"entities"] == b"Chris"
    assert stored[b"text"] == b"Chris likes tea"


@pytest.mark.asyncio
async def test_helper_never_creates_a_missing_memory(redis):
    key = Keys.memory_key("deleted-1")
    assert (
        await update_memory_if_present(
            redis, key, {"topics": "food"}, incr_field="access_count"
        )
        is False
    )
    assert await redis.exists(key) == 0


@pytest.mark.asyncio
async def test_helper_ignores_a_hash_without_text(redis):
    # A pre-existing ghost (no text) is not "live": the helper must not keep feeding it.
    key = Keys.memory_key("ghost-1")
    await redis.hset(key, mapping={"topics": "x"})
    assert await update_memory_if_present(redis, key, {"topics": "y"}) is False
    assert (await redis.hgetall(key)) == {b"topics": b"x"}


@pytest.mark.asyncio
async def test_helper_increments_when_asked(redis):
    key = await _live_memory(redis, "live-2")
    await update_memory_if_present(
        redis, key, {"last_accessed": "1"}, incr_field="access_count"
    )
    await update_memory_if_present(
        redis, key, {"last_accessed": "2"}, incr_field="access_count"
    )
    stored = await redis.hgetall(key)
    assert stored[b"access_count"] == b"2" and stored[b"last_accessed"] == b"2"


# --- the three writers ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_memory_structure_does_not_resurrect_a_deleted_memory(redis):
    memory = MemoryRecord(
        id="deleted-2", text="Chris likes tea"
    )  # stored, then deleted
    with (
        patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            AsyncMock(return_value=redis),
        ),
        patch(
            "agent_memory_server.long_term_memory.handle_extraction",
            AsyncMock(return_value=(["food"], ["Chris Baker"])),
        ),
    ):
        await extract_memory_structure(memory)
    assert await redis.exists(Keys.memory_key("deleted-2")) == 0


@pytest.mark.asyncio
async def test_extract_memory_structure_writes_a_live_memory(redis):
    key = await _live_memory(redis, "live-3")
    memory = MemoryRecord(id="live-3", text="Chris likes tea")
    with (
        patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            AsyncMock(return_value=redis),
        ),
        patch(
            "agent_memory_server.long_term_memory.handle_extraction",
            AsyncMock(return_value=(["food"], ["Chris Baker"])),
        ),
        patch(
            "agent_memory_server.long_term_memory.enforce_topics",
            side_effect=lambda t: t,
        ),
    ):
        await extract_memory_structure(memory)
    stored = await redis.hgetall(key)
    assert stored[b"topics"] == b"food"
    assert stored[b"entities"] == b"Chris Baker"


@pytest.mark.asyncio
async def test_update_last_accessed_skips_missing_and_counts_only_live(redis):
    live = await _live_memory(redis, "live-4")
    missing = Keys.memory_key("deleted-3")
    updated = await update_last_accessed(
        ["live-4", "deleted-3"], redis_client=redis, min_interval_seconds=0
    )
    assert updated == 1, "only the live memory is updated (and counted)"
    assert await redis.exists(missing) == 0, (
        "a missing memory must not be created as a ghost"
    )
    stored = await redis.hgetall(live)
    assert stored[b"access_count"] == b"1" and b"last_accessed" in stored
