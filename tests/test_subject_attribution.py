"""Regression tests for memory subject-attribution (root-cause fix 2026-06-20).

Symptom: facts about a relative ("my son Christian is a lightweight rower")
were stored as "Chris Baker is a lightweight rower" because the extraction
prompts centered every fact on the SPEAKER ({user_name}) with no rule to
attribute third-party facts to the named person.

These lock in the prompt structure (roster injection + subject-attribution
rule). LLM behavior itself is validated live, but a prompt that loses the rule
would regress silently — so we assert the rule text + the {family_context}
placeholder are present and that every prompt still formats with all kwargs.

Run: uv run pytest tests/test_subject_attribution.py -v
"""

from datetime import datetime

from agent_memory_server.memory_strategies import (
    DiscreteMemoryStrategy,
    SummaryMemoryStrategy,
    UserPreferencesMemoryStrategy,
    _load_family_context,
)

_DT = datetime(2026, 6, 20).strftime("%A, %B %d, %Y")


# --- family roster ---------------------------------------------------------


def test_family_context_lists_children_and_spouse():
    roster = _load_family_context()
    # Depends on ~/.openclaw/family.json being present on this host.
    assert "Christian Baker" in roster
    assert "Lindsey Baker" in roster
    assert "Lindalee Baker" in roster
    assert "speaker's child" in roster
    assert "speaker's spouse" in roster


def test_family_context_marks_speaker_distinctly():
    roster = _load_family_context()
    assert "Chris Baker — the application user / primary speaker" in roster


# --- prompt structure: roster placeholder present --------------------------


def test_all_prompts_inject_family_context():
    assert "{family_context}" in DiscreteMemoryStrategy.EXTRACTION_PROMPT
    assert "{family_context}" in SummaryMemoryStrategy.SUMMARY_PROMPT
    assert "{family_context}" in UserPreferencesMemoryStrategy.PREFERENCES_PROMPT


def test_all_prompts_distinguish_speaker_from_subject():
    for tmpl in (
        DiscreteMemoryStrategy.EXTRACTION_PROMPT,
        SummaryMemoryStrategy.SUMMARY_PROMPT,
        UserPreferencesMemoryStrategy.PREFERENCES_PROMPT,
    ):
        assert "KNOWN PEOPLE" in tmpl
        assert "NOT the speaker" in tmpl


def test_discrete_prompt_has_subject_attribution_rule_and_counterexample():
    p = DiscreteMemoryStrategy.EXTRACTION_PROMPT
    assert "SUBJECT ATTRIBUTION" in p
    # The concrete counter-example that pins the exact failure mode.
    assert "Christian Baker is a lightweight rower" in p
    assert 'do not turn "my son is a lightweight rower"' in p


# --- prompts still format with all required kwargs -------------------------


def test_discrete_prompt_formats():
    out = DiscreteMemoryStrategy.EXTRACTION_PROMPT.format(
        message="my son Christian is a lightweight rower",
        top_k_topics=5,
        current_datetime=_DT,
        user_name="Chris Baker",
        family_context=_load_family_context(),
    )
    assert "Christian Baker is a lightweight rower" in out
    assert "Chris Baker — the application user" in out


def test_summary_prompt_formats():
    out = SummaryMemoryStrategy.SUMMARY_PROMPT.format(
        message="x",
        current_datetime=_DT,
        user_name="Chris Baker",
        family_context=_load_family_context(),
        max_length=500,
    )
    assert "KNOWN PEOPLE" in out


def test_preferences_prompt_formats():
    out = UserPreferencesMemoryStrategy.PREFERENCES_PROMPT.format(
        message="x",
        top_k_topics=5,
        current_datetime=_DT,
        user_name="Chris Baker",
        family_context=_load_family_context(),
    )
    assert "KNOWN PEOPLE" in out
