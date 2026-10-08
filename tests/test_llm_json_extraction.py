"""Regression locks for fenced-JSON LLM output and extraction-failure handling.

2026-10-08: after the generation model moved to Claude Haiku 5.5, every
session-thread extraction raised JSONDecodeError. LiteLLM maps a schema-less
``response_format={"type": "json_object"}`` to nothing on Anthropic, so the
model returned ```json … ``` fenced output and the bare ``json.loads`` failed.
``extract_memories_from_session_thread`` then swallowed the error and returned
``[]``, which ``run_delayed_extraction`` treated as "nothing to remember" and
marked every message extracted — permanently dropping the thread's facts.
"""

import json
import pathlib
import re
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import RetryError

from agent_memory_server.llm.types import ChatCompletionResponse
from agent_memory_server.long_term_memory import (
    MemoryExtractionError,
    extract_memories_from_session_thread,
    run_delayed_extraction,
)
from agent_memory_server.memory_strategies import (
    DiscreteMemoryStrategy,
    _require_memories_list,
)
from agent_memory_server.models import MemoryMessage, WorkingMemory
from agent_memory_server.utils.llm_json import parse_llm_json_object
from agent_memory_server.working_memory import get_working_memory, set_working_memory


PAYLOAD = {
    "memories": [
        {
            "type": "semantic",
            "kind": "fact",
            "text": "Chris Baker runs the OpenClaw gateway on a Mac mini.",
            "topics": ["OpenClaw infrastructure"],
            "entities": ["Chris Baker", "OpenClaw"],
        }
    ]
}
BODY = json.dumps(PAYLOAD, indent=2)


def _response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        content=content,
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        model="claude-haiku-5-5",
    )


class TestParseLlmJsonObject:
    @pytest.mark.parametrize(
        "content",
        [
            BODY,
            f"```json\n{BODY}\n```",  # the exact shape logged live from Haiku 5.5
            f"```JSON\n{BODY}\n```",
            f"```\n{BODY}\n```",
            f"  \n```json\n{BODY}\n```\n\n",
            f"```json\r\n{BODY}\r\n```",
            f"Here are the memories:\n{BODY}\nLet me know if you need more.",
            f"Here are the memories:\n```json\n{BODY}\n```",
        ],
    )
    def test_accepts_bare_fenced_and_prose_wrapped(self, content):
        assert parse_llm_json_object(content) == PAYLOAD

    def test_fence_inside_string_value_is_preserved(self):
        value = {"text": "use ```json fences``` in docs"}
        assert parse_llm_json_object(json.dumps(value)) == value

    @pytest.mark.parametrize(
        "content",
        [
            "",
            None,
            "   ",
            "no json here",
            '```json\n{"memories": [{"text": "trunc',  # max_tokens truncation
            "[1, 2, 3]",  # top-level array is not an object
            '"just a string"',
        ],
    )
    def test_rejects_with_json_decode_error(self, content):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json_object(content)


class TestDiscreteStrategyParsesFencedOutput:
    @pytest.mark.asyncio
    async def test_fenced_response_yields_memories_on_first_attempt(self):
        mock = AsyncMock(return_value=_response(f"```json\n{BODY}\n```"))
        with patch(
            "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
            mock,
        ):
            memories = await DiscreteMemoryStrategy().extract_memories(
                "[Chris Baker]: the gateway runs on the mac mini"
            )
        assert memories == PAYLOAD["memories"]
        assert mock.await_count == 1  # no retry burned on a parse failure

    @pytest.mark.asyncio
    async def test_off_schema_object_is_retried_not_read_as_empty(self):
        # codex F1: a parseable object without "memories" must not become [].
        mock = AsyncMock(
            side_effect=[
                _response('```json\n{"error": "rate limited"}\n```'),
                _response(BODY),
            ]
        )
        with patch(
            "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
            mock,
        ):
            memories = await DiscreteMemoryStrategy().extract_memories("x")
        assert memories == PAYLOAD["memories"]
        assert mock.await_count == 2

    @pytest.mark.asyncio
    async def test_persistent_off_schema_output_raises(self):
        mock = AsyncMock(return_value=_response('{"error": "rate limited"}'))
        with (
            patch(
                "agent_memory_server.memory_strategies.LLMClient.create_chat_completion",
                mock,
            ),
            pytest.raises(RetryError),
        ):
            await DiscreteMemoryStrategy().extract_memories("x")
        assert mock.await_count == 3


class TestRequireMemoriesList:
    def test_explicit_empty_list_is_a_legitimate_empty_result(self):
        assert _require_memories_list({"memories": []}) == []

    @pytest.mark.parametrize(
        "data",
        [{}, {"error": "rate limited"}, {"memories": None}, {"memories": "x"}],
    )
    def test_missing_or_non_list_raises(self, data):
        with pytest.raises(ValueError, match="no 'memories' list"):
            _require_memories_list(data)


@pytest.mark.asyncio
class TestExtractionFailureDoesNotDropMessages:
    SESSION = "test-extraction-failure-keeps-unextracted"
    USER = "test-user"
    NS = "test"

    async def _seed(self, redis):
        await set_working_memory(
            WorkingMemory(
                session_id=self.SESSION,
                user_id=self.USER,
                namespace=self.NS,
                messages=[
                    MemoryMessage(
                        id="msg-1",
                        role="user",
                        content="The gateway runs on the mac mini.",
                        discrete_memory_extracted="f",
                    )
                ],
                memories=[],
            ),
            redis_client=redis,
        )

    async def test_thread_extraction_raises_typed_error(self, async_redis_client):
        await self._seed(async_redis_client)
        with (
            patch(
                "agent_memory_server.memory_strategies.DiscreteMemoryStrategy.extract_memories",
                AsyncMock(side_effect=json.JSONDecodeError("bad", "x", 0)),
            ),
            pytest.raises(MemoryExtractionError) as exc,
        ):
            await extract_memories_from_session_thread(
                session_id=self.SESSION, namespace=self.NS, user_id=self.USER
            )
        assert isinstance(exc.value.__cause__, json.JSONDecodeError)

    async def _run_failing(self, redis, retry_attempt=0):
        """Run one failing delayed extraction; return (count, schedule mock, counter mock)."""
        await self._seed(redis)
        with (
            patch(
                "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                AsyncMock(side_effect=MemoryExtractionError("llm down")),
            ),
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index,
            patch(
                "agent_memory_server.long_term_memory.schedule_trailing_extraction",
                AsyncMock(),
            ) as mock_schedule,
            patch(
                "agent_memory_server.long_term_memory.record_counter"
            ) as mock_counter,
        ):
            count = await run_delayed_extraction(
                session_id=self.SESSION,
                namespace=self.NS,
                user_id=self.USER,
                scheduled_timestamp=None,
                retry_attempt=retry_attempt,
            )
        mock_index.assert_not_called()
        return count, mock_schedule, mock_counter

    async def _flags(self, redis):
        wm = await get_working_memory(
            session_id=self.SESSION,
            user_id=self.USER,
            namespace=self.NS,
            redis_client=redis,
        )
        assert wm is not None
        return [m.discrete_memory_extracted for m in wm.messages]

    @pytest.fixture(autouse=True)
    def _extraction_settings(self):
        from agent_memory_server.config import settings

        saved = (
            settings.enable_discrete_memory_extraction,
            settings.extraction_failure_max_retries,
            settings.extraction_failure_retry_base_seconds,
        )
        settings.enable_discrete_memory_extraction = True
        settings.extraction_failure_max_retries = 3
        settings.extraction_failure_retry_base_seconds = 300
        yield
        (
            settings.enable_discrete_memory_extraction,
            settings.extraction_failure_max_retries,
            settings.extraction_failure_retry_base_seconds,
        ) = saved

    async def test_delayed_extraction_leaves_messages_unextracted(
        self, async_redis_client, mock_memory_vector_db
    ):
        count, _, _ = await self._run_failing(async_redis_client)
        assert count == 0
        assert await self._flags(async_redis_client) == ["f"]

    @pytest.mark.parametrize("attempt,delay", [(0, 300), (1, 600), (2, 1200)])
    async def test_failure_schedules_backed_off_retry(
        self, async_redis_client, mock_memory_vector_db, attempt, delay
    ):
        # codex round 2 F1: a quiet session must be retried without a new message.
        _, mock_schedule, mock_counter = await self._run_failing(
            async_redis_client, retry_attempt=attempt
        )
        mock_schedule.assert_awaited_once()
        kwargs = mock_schedule.await_args.kwargs
        assert kwargs["session_id"] == self.SESSION
        assert kwargs["namespace"] == self.NS
        assert kwargs["user_id"] == self.USER
        assert kwargs["delay_seconds"] == delay
        assert kwargs["retry_attempt"] == attempt + 1
        mock_counter.assert_called_once_with(
            "memory_server.extraction.failed",
            attributes={"outcome": "retry_scheduled"},
        )

    async def test_retries_exhausted_stops_and_reports(
        self, async_redis_client, mock_memory_vector_db
    ):
        count, mock_schedule, mock_counter = await self._run_failing(
            async_redis_client, retry_attempt=3
        )
        assert count == 0
        mock_schedule.assert_not_awaited()
        mock_counter.assert_called_once_with(
            "memory_server.extraction.failed", attributes={"outcome": "exhausted"}
        )
        assert await self._flags(async_redis_client) == ["f"]


@pytest.mark.asyncio
async def test_schedule_passes_delay_and_retry_attempt_to_asyncio_task(
    async_redis_client,
):
    """The retry delay/attempt must reach run_delayed_extraction (no-docket path)."""
    import asyncio

    from agent_memory_server.config import settings
    from agent_memory_server.long_term_memory import schedule_trailing_extraction

    saved = settings.use_docket
    settings.use_docket = False
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await real_sleep(0)

    try:
        with (
            patch("asyncio.sleep", fake_sleep),
            patch(
                "agent_memory_server.long_term_memory.run_delayed_extraction",
                AsyncMock(return_value=0),
            ) as mock_run,
        ):
            await schedule_trailing_extraction(
                session_id="s",
                namespace="n",
                user_id="u",
                redis=async_redis_client,
                delay_seconds=600,
                retry_attempt=2,
            )
            for _ in range(5):
                if mock_run.await_count:
                    break
                await real_sleep(0.01)
        assert sleeps[0] == 600
        mock_run.assert_awaited_once()
        assert mock_run.await_args.kwargs["retry_attempt"] == 2
        ttl = await async_redis_client.ttl("extraction_pending:s")
        assert 600 < ttl <= 1200
    finally:
        settings.use_docket = saved


@pytest.mark.asyncio
async def test_strategy_aware_failure_leaves_parent_unextracted():
    """codex round 3 F1: extract_memories_with_strategy must not mark a parent
    "t" when its strategy raises — that permanently dropped its facts."""
    from unittest.mock import MagicMock

    import agent_memory_server.extraction as ext
    from agent_memory_server.models import MemoryRecord

    class _FailingStrategy:
        async def extract_memories(self, text, source_user_name=None):
            raise ValueError("LLM JSON response has no 'memories' list")

    fake_db = MagicMock()
    fake_db.update_memories = AsyncMock(return_value=1)
    index = AsyncMock()
    parent = MemoryRecord(
        id="msg-fail",
        text="a conversation",
        memory_type="message",
        extraction_strategy="discrete",
        extraction_strategy_config={},
        discrete_memory_extracted="f",
    )
    with (
        patch(
            "agent_memory_server.memory_vector_db_factory.get_memory_vector_db",
            AsyncMock(return_value=fake_db),
        ),
        patch(
            "agent_memory_server.memory_strategies.get_memory_strategy",
            return_value=_FailingStrategy(),
        ),
        patch("agent_memory_server.long_term_memory.index_long_term_memories", index),
        patch("agent_memory_server.extraction.record_counter") as counter,
    ):
        await ext.extract_memories_with_strategy(memories=[parent], deduplicate=False)

    fake_db.update_memories.assert_not_called()
    index.assert_not_called()
    counter.assert_called_once_with(
        "memory_server.extraction.failed",
        attributes={"outcome": "left_unextracted", "site": "strategy"},
    )


def test_no_bare_json_loads_of_llm_content():
    """Every LLM-output parse must go through parse_llm_json_object."""
    src = pathlib.Path(__file__).resolve().parent.parent / "agent_memory_server"
    bare = re.compile(r"json\.loads\(\s*\w+\.content\s*\)")
    offenders = [
        f"{p.relative_to(src.parent)}:{n}"
        for p in src.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if bare.search(line)
    ]
    assert offenders == []
