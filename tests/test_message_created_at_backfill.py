"""LAB-391: read-time created_at backfill for legacy working-memory messages.

The MemoryMessage validator warns ("created_at will become required") and recency
scoring degrades when a stored message lacks created_at. `backfill_message_created_at`
fills a default (the enclosing working-memory record's timestamp) on reconstruction so
re-reading legacy data is quiet and deterministically ordered, without clobbering
messages that already carry a real created_at.
"""

from datetime import UTC, datetime

from agent_memory_server.models import MemoryMessage
from agent_memory_server.working_memory import backfill_message_created_at


DEFAULT = "2026-01-01T00:00:00+00:00"


def test_backfills_when_created_at_absent():
    out = backfill_message_created_at({"role": "user", "content": "hi"}, DEFAULT)
    assert out["created_at"] == DEFAULT
    # original keys preserved
    assert out["role"] == "user"
    assert out["content"] == "hi"


def test_backfills_when_created_at_is_none():
    out = backfill_message_created_at(
        {"role": "assistant", "content": "yo", "created_at": None}, DEFAULT
    )
    assert out["created_at"] == DEFAULT


def test_preserves_existing_created_at():
    real = "2026-06-23T12:00:00+00:00"
    out = backfill_message_created_at(
        {"role": "user", "content": "hi", "created_at": real}, DEFAULT
    )
    assert out["created_at"] == real


def test_does_not_mutate_input_dict():
    src = {"role": "user", "content": "hi"}
    backfill_message_created_at(src, DEFAULT)
    assert "created_at" not in src  # returns a new dict, original untouched


def test_passes_through_non_dict_unchanged():
    assert backfill_message_created_at(None, DEFAULT) is None
    assert backfill_message_created_at("not-a-dict", DEFAULT) == "not-a-dict"


def test_backfilled_message_constructs_without_warning(caplog):
    """A legacy message dict, once backfilled, instantiates MemoryMessage cleanly."""
    backfilled = backfill_message_created_at(
        {"id": "m1", "role": "user", "content": "legacy"},
        datetime.now(UTC).isoformat(),
    )
    with caplog.at_level("WARNING"):
        msg = MemoryMessage(**backfilled)
    assert msg.created_at is not None
    assert not any(
        "created_at" in rec.message and "required" in rec.message
        for rec in caplog.records
    )
