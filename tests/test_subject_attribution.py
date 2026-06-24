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

import asyncio
from datetime import datetime
from types import SimpleNamespace

from agent_memory_server.memory_strategies import (
    CustomMemoryStrategy,
    DiscreteMemoryStrategy,
    SummaryMemoryStrategy,
    UserPreferencesMemoryStrategy,
    _load_family_context,
    _subject_attribution_preamble,
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


# --- A10b: fail-loud when the family roster is unavailable/empty ---------------
# A missing/corrupt roster silently degrades extraction to speaker-centric
# attribution (the 2026-06-20 mis-attribution class). _load_family_context now
# WARNs + emits memory_server.family_roster.unavailable so it is observable.


def test_family_context_fails_loud_when_roster_missing(tmp_path, monkeypatch):
    import agent_memory_server.memory_strategies as ms

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ms, "record_counter", lambda name, **kw: calls.append((name, kw))
    )
    monkeypatch.setattr(
        ms.os.path, "expanduser", lambda _p: str(tmp_path / "nope.json")
    )
    monkeypatch.setattr(ms, "_FAMILY_CONTEXT_CACHE", None)

    assert ms._load_family_context() == ""
    assert any(
        name == "memory_server.family_roster.unavailable"
        and kw.get("attributes", {}).get("reason") == "load_error"
        for name, kw in calls
    )


def test_family_context_fails_loud_when_roster_empty(tmp_path, monkeypatch):
    import json as _json

    import agent_memory_server.memory_strategies as ms

    roster_file = tmp_path / "family.json"
    roster_file.write_text(_json.dumps({"users": {}}))
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ms, "record_counter", lambda name, **kw: calls.append((name, kw))
    )
    monkeypatch.setattr(ms.os.path, "expanduser", lambda _p: str(roster_file))
    monkeypatch.setattr(ms, "_FAMILY_CONTEXT_CACHE", None)

    assert ms._load_family_context() == ""
    assert any(
        name == "memory_server.family_roster.unavailable"
        and kw.get("attributes", {}).get("reason") == "empty"
        for name, kw in calls
    )


# --- LAB-396: the CUSTOM strategy (the live OpenClaw config path) must carry the
# roster + subject-attribution rule, which it previously bypassed entirely. ----


def test_subject_attribution_preamble_is_self_contained():
    """The preamble must be fully resolved (no leftover template placeholders) so
    it can be prepended after the secure formatter without re-formatting."""
    block = _subject_attribution_preamble("Chris Baker", _load_family_context())
    assert "{" not in block and "}" not in block
    assert "KNOWN PEOPLE" in block
    assert "SUBJECT ATTRIBUTION" in block
    assert "Christian Baker is a lightweight rower" in block
    assert "Chris Baker" in block  # the speaker is named


def _capture_custom_prompt(monkeypatch, custom_prompt, *, source_user_name):
    """Run CustomMemoryStrategy with a mocked LLM, return the prompt actually sent."""
    import agent_memory_server.memory_strategies as ms

    captured: dict[str, str] = {}

    async def _fake_completion(*, model, messages, response_format):  # noqa: ANN001
        captured["prompt"] = messages[0]["content"]
        return SimpleNamespace(content='{"memories": []}')

    monkeypatch.setattr(
        ms.LLMClient,
        "create_chat_completion",
        classmethod(lambda cls, **kw: _fake_completion(**kw)),
    )
    strategy = CustomMemoryStrategy(custom_prompt=custom_prompt)
    asyncio.run(
        strategy.extract_memories(
            "my son Christian is a lightweight rower",
            source_user_name=source_user_name,
        )
    )
    return captured["prompt"]


def test_custom_strategy_injects_roster_and_rule(monkeypatch):
    sent = _capture_custom_prompt(
        monkeypatch,
        "Extract facts from this conversation.\n\nMessage:\n{message}\n\n"
        "Return a JSON object with a memories list.",
        source_user_name="Chris Baker",
    )
    # Roster + rule + counter-example are all present on the custom path now.
    assert "KNOWN PEOPLE" in sent
    assert "SUBJECT ATTRIBUTION" in sent
    assert "Christian Baker is a lightweight rower" in sent
    assert "Chris Baker" in sent
    # The user's own custom prompt is preserved and follows the preamble.
    assert "Extract facts from this conversation." in sent
    assert sent.index("SUBJECT ATTRIBUTION") < sent.index(
        "Extract facts from this conversation."
    )


def test_custom_strategy_grounds_third_party_subject(monkeypatch):
    """A 'my son Christian…' utterance reaches the LLM together with the roster +
    counter-example, so a compliant extractor attributes it to Christian, not the
    speaker. (LLM behavior is validated live; here we lock the prompt contract.)"""
    sent = _capture_custom_prompt(
        monkeypatch,
        "Extract memories.\n\nMessage:\n{message}",
        source_user_name="Chris Baker",
    )
    assert "my son Christian is a lightweight rower" in sent  # the utterance
    assert "Christian Baker is a lightweight rower" in sent  # the attribution target
    # The rule explicitly forbids copying the speaker's name onto the son's fact.
    assert (
        "NEVER copy the speaker's name onto a fact that is about someone else" in sent
    )


def test_custom_strategy_preserves_fail_loud_when_roster_missing(tmp_path, monkeypatch):
    import agent_memory_server.memory_strategies as ms

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ms, "record_counter", lambda name, **kw: calls.append((name, kw))
    )
    monkeypatch.setattr(
        ms.os.path, "expanduser", lambda _p: str(tmp_path / "nope.json")
    )
    monkeypatch.setattr(ms, "_FAMILY_CONTEXT_CACHE", None)

    sent = _capture_custom_prompt(
        monkeypatch,
        "Extract memories.\n\nMessage:\n{message}",
        source_user_name="Chris Baker",
    )
    # Extraction still proceeds (degraded), and the degradation is observable.
    assert "(no roster available)" in sent
    assert "SUBJECT ATTRIBUTION" in sent
    assert any(name == "memory_server.family_roster.unavailable" for name, _ in calls)
