"""Tolerant parsing of JSON objects returned by chat-completion LLMs.

Why: every extraction call site requests ``response_format={"type": "json_object"}``.
OpenAI enforces that natively, but LiteLLM maps a schema-less ``json_object`` to
*nothing* on Anthropic (no forced tool, no ``output_format`` — see
``AnthropicConfig.map_response_format_to_anthropic_tool``), so Claude models are
free to wrap the object in a markdown fence (```json … ```) or a sentence of
prose. Claude Haiku 5.5 does this routinely; a bare ``json.loads`` then raised
``JSONDecodeError`` on every session-thread extraction (2026-10-08).
"""

from __future__ import annotations

import json
import re
from typing import Any


# Opening fence with an optional language tag (```json, ```JSON, ```), and the
# closing fence. Anchored to the whole (stripped) payload so a fence *inside* a
# JSON string value is never touched.
_FENCED = re.compile(
    r"\A```[A-Za-z0-9_-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?```\Z", re.DOTALL
)


def parse_llm_json_object(content: str | None) -> dict[str, Any]:
    """Parse a JSON object from raw LLM output.

    Accepts, in order of preference:
      1. a bare JSON object;
      2. a JSON object wrapped in a single markdown code fence;
      3. the first JSON object embedded in surrounding prose.

    Args:
        content: The model's text output (``ChatCompletionResponse.content``).

    Returns:
        The decoded top-level JSON object.

    Raises:
        json.JSONDecodeError: if no JSON object can be decoded, or the decoded
            top-level value is not an object (e.g. a bare array). Callers rely on
            this exact type to log the payload and let tenacity retry.
    """
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("LLM returned empty content", text, 0)

    fenced = _FENCED.match(text)
    if fenced:
        text = fenced.group("body").strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        # raw_decode stops at the end of the first complete value, so trailing
        # prose ("Let me know if…") is ignored; a truncated object still raises.
        value, _end = json.JSONDecoder().raw_decode(text, start)

    if not isinstance(value, dict):
        raise json.JSONDecodeError(
            f"expected a JSON object, got {type(value).__name__}", text, 0
        )
    return value
