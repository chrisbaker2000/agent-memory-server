"""LAB-533: off-enum ``memory_type`` from the extraction LLM is coerced, never fatal.

Observed live: ``memory_type='epistemic'`` raised a pydantic ValidationError on
``MemoryRecord`` and aborted the whole session-thread extraction. These tests lock:

* the alias/unknown/missing coercion policy of ``coerce_memory_type`` (incl. the
  WARNING carrying the original value + the ``memory_server.extraction.
  memory_type_coerced`` counter),
* the LIVE trailing-edge path end to end — ``run_delayed_extraction`` →
  ``extract_memories_from_session_thread`` → ``MemoryRecord`` →
  ``index_long_term_memories`` — so a coerced record actually reaches the persist
  call and one bad record never blocks its siblings,
* the shared extraction-prompt enum constraint in all strategies.

Pure + mock-based — no Redis/network/LLM. Samples are anonymized.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent_memory_server.long_term_memory as ltm
import agent_memory_server.utils.memory_type as mt
from agent_memory_server.memory_strategies import (
    CustomMemoryStrategy,
    DiscreteMemoryStrategy,
    SummaryMemoryStrategy,
    UserPreferencesMemoryStrategy,
    _subject_attribution_preamble,
)
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum


METRIC = "memory_server.extraction.memory_type_coerced"


@pytest.fixture
def spy(monkeypatch):
    """Capture the coercion module's WARNING logs and counter increments."""
    fake_logger = MagicMock()
    counters: list[tuple[str, dict]] = []
    monkeypatch.setattr(mt, "logger", fake_logger)
    monkeypatch.setattr(
        mt,
        "record_counter",
        lambda name, **kw: counters.append((name, kw.get("attributes", {}))),
    )
    return SimpleNamespace(logger=fake_logger, counters=counters)


def _warnings(spy) -> list[str]:
    return [c.args[0] for c in spy.logger.warning.call_args_list]


# --- coerce_memory_type policy ------------------------------------------------


def test_metric_name_is_the_documented_one():
    assert mt.COERCION_METRIC == METRIC


def test_epistemic_coerces_to_episodic_with_warning_and_counter(spy):
    assert mt.coerce_memory_type("epistemic", site="t") == "episodic"
    # Case/whitespace-insensitive alias match.
    assert mt.coerce_memory_type("  Epistemic ", site="t") == "episodic"
    warnings = _warnings(spy)
    assert len(warnings) == 2
    assert "'epistemic'" in warnings[0] and "'episodic'" in warnings[0]
    assert "'  Epistemic '" in warnings[1]  # the ORIGINAL value, not normalized
    assert (
        spy.counters
        == [
            (
                METRIC,
                {
                    "reason": "alias",
                    "coerced_to": "episodic",
                    "site": "t",
                    "original": "epistemic",
                },
            )
        ]
        * 2
    )


def test_unknown_value_defaults_to_semantic_with_warning(spy):
    assert mt.coerce_memory_type("bogus-type", site="t") == "semantic"
    warnings = _warnings(spy)
    assert len(warnings) == 1 and "'bogus-type'" in warnings[0]
    # Unknown values are bucketed — the raw LLM string is NOT a metric attribute
    # (bounded cardinality).
    assert spy.counters == [
        (METRIC, {"reason": "unknown", "coerced_to": "semantic", "site": "t"})
    ]


def test_unknown_value_ignores_call_site_default(spy):
    # The call-site default governs ONLY a missing type; an emitted-but-unknown
    # value always falls back to "semantic" (LAB-533).
    assert mt.coerce_memory_type("bogus", default="episodic") == "semantic"


def test_message_is_reserved_and_coerces_to_semantic(spy):
    assert mt.coerce_memory_type("message") == "semantic"
    assert spy.counters[0][1]["reason"] == "reserved"


@pytest.mark.parametrize("value", [123, ["semantic"], {"type": "semantic"}, 1.5])
def test_non_string_coerces_to_semantic_without_raising(spy, value):
    assert mt.coerce_memory_type(value) == "semantic"
    assert spy.counters[0][1]["reason"] == "invalid_type"
    assert len(_warnings(spy)) == 1


@pytest.mark.parametrize(
    ("value", "default"),
    [(None, "episodic"), (None, "semantic"), ("", "episodic"), ("  ", "semantic")],
)
def test_missing_or_empty_uses_call_site_default_silently(spy, value, default):
    assert mt.coerce_memory_type(value, default=default) == default
    assert spy.counters == []
    spy.logger.warning.assert_not_called()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("episodic", "episodic"), ("semantic", "semantic"), ("SEMANTIC ", "semantic")],
)
def test_valid_values_pass_through_silently(spy, value, expected):
    assert mt.coerce_memory_type(value) == expected
    assert spy.counters == []
    spy.logger.warning.assert_not_called()


def test_invalid_default_is_a_programming_error():
    with pytest.raises(ValueError):
        mt.coerce_memory_type("episodic", default="message")


def test_aliases_only_map_to_valid_extractable_types():
    assert set(mt._MEMORY_TYPE_ALIASES.values()) <= mt.VALID_MEMORY_TYPES
    # Keys are normalized and never shadow a valid value.
    for key in mt._MEMORY_TYPE_ALIASES:
        assert key == key.strip().lower()
        assert key not in {e.value for e in MemoryTypeEnum}


def test_every_coerced_value_is_accepted_by_memory_record():
    for value in ["epistemic", "bogus", "message", 7, None, ""]:
        MemoryRecord(id="x", text="t", memory_type=mt.coerce_memory_type(value))


# --- the LIVE session-thread path, end to end (extract -> construct -> persist)


def _wm():
    return SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                content="User A is shopping for a home in Atlanta",
                discrete_memory_extracted="f",
            )
        ]
    )


class _LLMStrategy:
    """Stands in for DiscreteMemoryStrategy: returns raw LLM JSON dicts."""

    def __init__(self, out):
        self._out = out

    async def extract_memories(self, text, source_user_name=None):
        return self._out


async def _run_delayed(llm_output):
    """Drive run_delayed_extraction with everything external mocked; return the
    MemoryRecords handed to index_long_term_memories (the persist call)."""
    persisted: list[MemoryRecord] = []

    async def _capture_index(memories, **kwargs):
        persisted.extend(memories)

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.delete = AsyncMock(return_value=1)
    wm = _wm()

    with (
        patch(
            "agent_memory_server.utils.redis.get_redis_conn",
            AsyncMock(return_value=fake_redis),
        ),
        patch.object(
            ltm, "should_extract_session_thread", AsyncMock(return_value=True)
        ),
        patch.object(ltm, "set_extraction_debounce", AsyncMock()),
        patch(
            "agent_memory_server.working_memory.get_working_memory",
            AsyncMock(return_value=wm),
        ),
        patch("agent_memory_server.working_memory.set_working_memory", AsyncMock()),
        patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=_LLMStrategy(llm_output),
        ),
        patch.object(ltm, "index_long_term_memories", _capture_index),
    ):
        count = await ltm.run_delayed_extraction(
            session_id="agent:main:test:direct:user-a",
            source_user="user-a",
            scheduled_timestamp=None,
        )
    return count, persisted


@pytest.mark.asyncio
async def test_live_path_persists_epistemic_record_as_episodic(spy):
    """Acceptance: replaying the failing shape stores the record as 'episodic'
    and logs a WARNING — no exception, nothing dropped."""
    count, persisted = await _run_delayed(
        [{"text": "User A toured a house in Atlanta in May 2026", "type": "epistemic"}]
    )
    assert count == 1
    assert len(persisted) == 1
    assert persisted[0].memory_type == "episodic"
    assert any("'epistemic'" in w for w in _warnings(spy))
    assert (
        METRIC,
        {
            "reason": "alias",
            "coerced_to": "episodic",
            "site": "session_thread",
            "original": "epistemic",
        },
    ) in spy.counters


@pytest.mark.asyncio
async def test_live_path_unknown_type_persists_as_semantic(spy):
    count, persisted = await _run_delayed(
        [{"text": "User A prefers single-story homes", "type": "TOTALLY-NEW-TYPE"}]
    )
    assert count == 1
    assert persisted[0].memory_type == "semantic"
    assert any("'TOTALLY-NEW-TYPE'" in w for w in _warnings(spy))


@pytest.mark.asyncio
async def test_live_path_one_invalid_record_does_not_block_siblings(spy):
    """A batch with an unconstructable record (no text), non-dict debris, and an
    off-enum type still persists every constructable sibling."""
    count, persisted = await _run_delayed(
        [
            {"text": "User A's budget is $500k", "type": "semantic"},
            {"type": "semantic"},  # missing required text -> skipped
            "not a memory object",  # LLM debris -> skipped
            {"text": "User A met an agent on 2026-05-02", "type": "epistemic"},
            {"text": "User A wants a yard", "type": "episodic"},
        ]
    )
    assert count == 3
    assert [r.text for r in persisted] == [
        "User A's budget is $500k",
        "User A met an agent on 2026-05-02",
        "User A wants a yard",
    ]
    assert [r.memory_type for r in persisted] == ["semantic", "episodic", "episodic"]


# --- extraction prompts carry the exact enum constraint ------------------------

_CONSTRAINT_MARKERS = ('"type" field MUST be exactly one of "episodic" or "semantic"',)


def _assert_constraint(text: str):
    for marker in _CONSTRAINT_MARKERS:
        assert marker in text


def test_constraint_lists_exactly_the_extractable_enum_values():
    c = mt.MEMORY_TYPE_PROMPT_CONSTRAINT
    _assert_constraint(c)
    assert '"epistemic"' in c  # explicitly forbidden by name
    # Brace-free so it is safe as a str.format() value and in the custom preamble.
    assert "{" not in c and "}" not in c


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    [
        DiscreteMemoryStrategy(),
        SummaryMemoryStrategy(),
        UserPreferencesMemoryStrategy(),
    ],
)
async def test_builtin_strategies_send_the_constraint_to_the_llm(strategy):
    """Assert on the prompt ACTUALLY SENT to the LLM (not the raw template), so a
    missing format kwarg or dropped placeholder is caught."""
    sent: list[str] = []

    async def _fake_completion(model, messages, response_format):
        sent.append(messages[0]["content"])
        return SimpleNamespace(content='{"memories": []}')

    with patch(
        "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
        _fake_completion,
    ):
        assert await strategy.extract_memories("hello", source_user_name="User A") == []
    assert len(sent) == 1
    _assert_constraint(sent[0])
    assert "{memory_type_constraint}" not in sent[0]


def test_custom_strategy_preamble_carries_the_constraint():
    _assert_constraint(_subject_attribution_preamble("User A", "(no roster)"))


@pytest.mark.asyncio
async def test_custom_strategy_coerces_instead_of_dropping(spy):
    """The custom strategy's security validator rejects any off-enum type; LAB-533
    normalizes first so a near-miss enum value no longer drops the memory."""
    strategy = CustomMemoryStrategy(
        custom_prompt="Extract memories from: {message}. Return JSON."
    )

    async def _fake_completion(model, messages, response_format):
        return SimpleNamespace(
            content='{"memories": [{"type": "epistemic", "text": "User A likes porches"}]}'
        )

    with patch(
        "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
        _fake_completion,
    ):
        out = await strategy.extract_memories("hi", source_user_name="User A")
    assert out == [{"type": "episodic", "text": "User A likes porches"}]
    assert spy.counters[0][1]["site"] == "custom_strategy"
