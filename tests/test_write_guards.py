"""LAB-54 — behavioral tests for the write-time memory guards.

`tests/memory/write-guards.sh` only source-greps these two load-bearing guards
(a log string, a regex of the validation, the pattern identifier). A refactor
that broke the verb allowlist or the event_date coercion while preserving the
comment/identifier/log would pass that grep, and a benign reword would falsely
fail it. These tests feed bad input through the actual guard functions and assert
the guard ACTS — the behavioral contract the grep cannot encode.

The guards are extracted as pure helpers (`_is_user_as_person`,
`_coerce_event_date`) from `index_long_term_memories`, mirroring the existing
`_is_noise_content` / `_normalize_source_user` testable-helper pattern.
"""

from datetime import datetime

import pytest

from agent_memory_server.long_term_memory import (
    _coerce_event_date,
    _is_user_as_person,
)


class TestUserAsPersonRejection:
    """`_is_user_as_person` → True means index_long_term_memories drops the memory."""

    @pytest.mark.parametrize(
        "text",
        [
            "User prefers dark mode",
            "User is a software engineer",
            "User has three children",
            "User asked about mortgage rates",
            # The 2026-04-14 production regression: "User learned …" bypassed an
            # earlier guard. Lock it so a future allowlist trim can't reopen it.
            "User learned on April 5, 2026 about the new tax law",
            "User wants to refinance",
            "User owns a 2019 Subaru",
        ],
    )
    def test_user_verb_at_start_is_rejected(self, text):
        assert _is_user_as_person(text) is True, f"should reject: {text!r}"

    def test_user_verb_mid_sentence_is_rejected(self):
        # "On March 14, User asked …" — the mid-sentence person reference that
        # documentably bypassed the original start-anchored regex.
        assert _is_user_as_person("On March 14, User asked about taxes") is True
        assert _is_user_as_person("Earlier today. User mentioned a deadline") is True

    def test_leading_tag_envelope_is_stripped_before_matching(self):
        # The "[topic]" envelope must not let a "User <verb>" memory through.
        assert _is_user_as_person("[preference] User likes oat milk") is True
        assert _is_user_as_person("[fact] User works at a bank") is True

    @pytest.mark.parametrize(
        "text",
        [
            # Real person name — the correct, attributed form. Must pass.
            "Chris prefers dark mode",
            "Christian asked about rowing practice",
            # Lowercase "user" as a common noun, not a person reference.
            "The user interface is clean and fast",
            "Reset the user password after the migration",
            # "User" followed by a non-verb noun — not a "User <verb>" claim.
            "User Guide for the espresso machine is in the drawer",
            # "User" embedded in a larger word — not at a token boundary.
            "Superuser access was granted to the deploy account",
            "",
        ],
    )
    def test_benign_text_is_not_rejected(self, text):
        assert _is_user_as_person(text) is False, f"should NOT reject: {text!r}"


class TestEventDateCoercion:
    """`_coerce_event_date` normalizes the NUMERIC event_date so Redis Search
    never silently orphans the record on a stringly-typed value."""

    def test_datetime_passes_through_unchanged(self):
        dt = datetime(2026, 2, 22, 13, 45, 0)
        assert _coerce_event_date(dt) is dt

    def test_none_stays_none(self):
        assert _coerce_event_date(None) is None

    def test_iso_date_string_is_converted(self):
        result = _coerce_event_date("2026-02-22")
        assert isinstance(result, datetime)
        assert (result.year, result.month, result.day) == (2026, 2, 22)

    def test_iso_datetime_string_is_converted(self):
        result = _coerce_event_date("2026-02-22T13:45:00")
        assert isinstance(result, datetime)
        assert (result.hour, result.minute) == (13, 45)

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-date",
            "2026-13-99",  # parses syntactically? month 13 is invalid → ValueError
            "yesterday",
            "",
            12345,  # int — str(12345) is not ISO
            [],  # unparseable type
        ],
    )
    def test_unparseable_values_become_none(self, value):
        # Never raise, never store a junk value — normalize to None so the write
        # funnel can index the record cleanly (or drop the bad date).
        assert _coerce_event_date(value) is None
