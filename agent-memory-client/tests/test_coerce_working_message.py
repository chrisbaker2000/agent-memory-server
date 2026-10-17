"""LAB-538: working-memory message coercion keeps the caller's created_at."""

from datetime import UTC, datetime

import pytest

from agent_memory_client.client import coerce_working_message
from agent_memory_client.models import MemoryMessage


def test_dict_keeps_explicit_created_at():
    msg = coerce_working_message(
        {
            "role": "user",
            "content": "Grant's game",
            "created_at": "2026-10-16T20:40:06+00:00",
        }
    )
    assert msg.created_at == datetime(2026, 10, 16, 20, 40, 6, tzinfo=UTC)


def test_memory_message_is_returned_unchanged():
    original = MemoryMessage(
        role="assistant",
        content="ok",
        created_at=datetime(2026, 10, 16, 20, 41, tzinfo=UTC),
    )
    assert coerce_working_message(original) is original


def test_dict_requires_role_and_content():
    with pytest.raises(ValueError, match="role"):
        coerce_working_message({"content": "hi"})


def test_non_dict_is_rejected():
    with pytest.raises(ValueError, match="dictionaries"):
        coerce_working_message("nope")  # type: ignore[arg-type]
