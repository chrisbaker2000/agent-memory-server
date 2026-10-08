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

from agent_memory_server.llm.types import ChatCompletionResponse
from agent_memory_server.long_term_memory import (
    MemoryExtractionError,
    extract_memories_from_session_thread,
    run_delayed_extraction,
)
from agent_memory_server.memory_strategies import DiscreteMemoryStrategy
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

    async def test_delayed_extraction_leaves_messages_unextracted(
        self, async_redis_client, mock_memory_vector_db
    ):
        from agent_memory_server.config import settings

        original = settings.enable_discrete_memory_extraction
        settings.enable_discrete_memory_extraction = True
        try:
            await self._seed(async_redis_client)
            with (
                patch(
                    "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
                    AsyncMock(side_effect=MemoryExtractionError("llm down")),
                ),
                patch(
                    "agent_memory_server.long_term_memory.index_long_term_memories"
                ) as mock_index,
            ):
                count = await run_delayed_extraction(
                    session_id=self.SESSION,
                    namespace=self.NS,
                    user_id=self.USER,
                    scheduled_timestamp=None,
                )

            assert count == 0
            mock_index.assert_not_called()
            wm = await get_working_memory(
                session_id=self.SESSION,
                user_id=self.USER,
                namespace=self.NS,
                redis_client=async_redis_client,
            )
            assert wm is not None
            assert [m.discrete_memory_extracted for m in wm.messages] == ["f"]
        finally:
            settings.enable_discrete_memory_extraction = original


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
