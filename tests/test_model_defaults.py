"""Fork model-default regression locks (LAB-379; current-generation policy 2026-10-08).

The fork's in-code Settings defaults drifted historically onto retired models
(gpt-5 / gpt-5-mini, later gpt-4o-mini). The live homelab deployment overrides
them via run-local.sh (GENERATION/FAST/SLOW_MODEL = anthropic/claude-haiku-5-5),
so drift is invisible — until an env export fails and the fallback reaches a
retired model. These tests lock the in-code fallbacks to the deployed values.

Settings is a pydantic BaseSettings that reads matching env vars, so these tests
assert on the class-level field defaults (``Settings.model_fields[...].default``)
rather than an instantiated Settings — that pins the *in-code* fallback
independent of whatever the test runner's environment happens to export.

Run: uv run pytest tests/test_model_defaults.py -v
"""

import re

import pytest

from agent_memory_server.config import MODEL_CONFIGS, Settings
from agent_memory_server.llm.client import LLMClient


DEPLOYED = "anthropic/claude-haiku-5-5"
LLM_FIELDS = ("generation_model", "fast_model", "slow_model")
# Retired for this deployment: 3.x/4.x Claude and the gpt-4o / gpt-5 OpenAI chat models.
RETIRED = re.compile(r"claude-(?:3|(?:opus|sonnet|haiku)-4)|gpt-4o|gpt-5")


@pytest.mark.parametrize("field", LLM_FIELDS)
def test_llm_model_defaults_match_deployment(field):
    assert Settings.model_fields[field].default == DEPLOYED


@pytest.mark.parametrize("field", LLM_FIELDS)
def test_no_llm_default_references_a_retired_model(field):
    default = Settings.model_fields[field].default
    assert not RETIRED.search(default), f"{field} default {default!r} is a retired model"


@pytest.mark.parametrize("model_id", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_claude_5_5_family_has_model_configs(model_id):
    cfg = MODEL_CONFIGS[model_id]
    assert cfg.name == model_id
    assert cfg.max_tokens >= 1_000_000


def test_unknown_model_falls_back_to_haiku_5_5():
    cfg = LLMClient.get_model_config("definitely-not-a-model")
    assert cfg.name == "claude-haiku-5-5"
