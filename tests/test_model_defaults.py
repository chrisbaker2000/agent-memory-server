"""Fork model-default regression locks (LAB-379).

The fork's in-code Settings defaults drifted historically onto retired OpenAI
models (gpt-5 / gpt-5-mini). The live homelab deployment overrides them via
run-local.sh (GENERATION_MODEL / FAST_MODEL = openai/gpt-4o-mini), so the drift
was invisible — until an env export failed and the fallback reached a model that
no longer exists. These tests lock the in-code fallbacks to the deployed values
so the default can never silently regress to gpt-5*.

Settings is a pydantic BaseSettings that reads matching env vars, so these tests
assert on the class-level field defaults (``Settings.model_fields[...].default``)
rather than an instantiated Settings — that pins the *in-code* fallback
independent of whatever the test runner's environment happens to export.

Run: uv run pytest tests/test_model_defaults.py -v
"""

from agent_memory_server.config import Settings


def test_generation_model_default_is_gpt_4o_mini():
    # LAB-379: aligned to deployed GENERATION_MODEL (was gpt-5 historically).
    assert Settings.model_fields["generation_model"].default == "openai/gpt-4o-mini"


def test_fast_model_default_is_gpt_4o_mini():
    # LAB-379: aligned to deployed FAST_MODEL (was gpt-5-mini historically).
    assert Settings.model_fields["fast_model"].default == "openai/gpt-4o-mini"


def test_no_model_default_references_retired_gpt5():
    # Guard against re-introducing a gpt-5* default on any of the LLM model knobs.
    for field in ("generation_model", "fast_model", "slow_model"):
        default = Settings.model_fields[field].default
        assert "gpt-5" not in default, (
            f"{field} default {default!r} references retired gpt-5*"
        )


def test_slow_model_default_left_as_anthropic():
    # slow_model is the heavier complex-query model and is intentionally NOT
    # repointed to gpt-4o-mini (LAB-379 scope). It stays an Anthropic model;
    # the deployed SLOW_MODEL (Haiku) overrides this via run-local.sh.
    assert Settings.model_fields["slow_model"].default.startswith("anthropic/")
