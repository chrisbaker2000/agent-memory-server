"""Family registry: test isolation, one configured path, and session attribution.

Root causes fixed 2026-10-09:

1. Live state in tests. Every registry reader opened ``~/.openclaw/family.json``
   directly (two of three ignored ``settings.family_json_path``), so results
   depended on the host. On the homelab host the API test saw a full name where
   it expected the short id; elsewhere the same tests passed. ``tests/conftest.py``
   now points ``FAMILY_JSON_PATH`` at the synthetic ``tests/fixtures/family.json``
   and all readers go through ``family_registry_path()``.

2. Owner fallback matched every session. ``resolve_user_from_session_id`` meant to
   attribute only the local ``agent:main:main`` and hook sessions to the owner,
   but tested ``"main" in parts``, and every key starts with ``agent:main:``.
   An unknown DM peer, or a group session, was attributed to the owner. Now only
   the session segment after ``agent:<id>:`` is inspected, and the owner is the
   registry's ``adminUser``.
"""

import json
import logging

import pytest

import agent_memory_server.long_term_memory as ltm
import agent_memory_server.memory_strategies as ms
from agent_memory_server.config import family_registry_path, settings
from agent_memory_server.extraction import (
    _load_family_registry,
    resolve_user_display_name,
    resolve_user_from_session_id,
)


# --- isolation ---------------------------------------------------------------


def test_tests_use_the_synthetic_registry_not_the_live_one():
    path = family_registry_path()
    assert path.endswith("tests/fixtures/family.json")
    assert ".openclaw" not in path


def test_synthetic_registry_is_loaded():
    assert resolve_user_display_name("alex") == "Alex Example"
    assert resolve_user_display_name("casey") == "Casey Other"


# --- session → source_user ---------------------------------------------------


@pytest.mark.parametrize(
    ("session_id", "expected"),
    [
        ("agent:main:discord:direct:100000000000000002", "sam"),
        ("agent:main:slack:dm:UTESTADMIN1", "alex"),
        ("agent:main:whatsapp:direct:+15550100001", "alex"),
        ("Agent:Main:Discord:Direct:100000000000000003", "riley"),
    ],
)
def test_known_dm_peer_resolves_to_its_user(session_id, expected):
    assert resolve_user_from_session_id(session_id) == expected


@pytest.mark.parametrize(
    "session_id",
    [
        # Unknown DM peer: was attributed to the owner via the "main" agent id.
        "agent:main:discord:direct:999999999",
        "agent:main:slack:dm:UNKNOWNPEER",
        # Group conversation: many speakers, no single user.
        "agent:main:discord:group:12345",
        "agent:main:whatsapp:group:120363000000000000",
        # A non-main agent with an unknown peer.
        "agent:helper:discord:direct:999",
    ],
)
def test_unattributable_sessions_resolve_to_none(session_id):
    assert resolve_user_from_session_id(session_id) is None


@pytest.mark.parametrize(
    "session_id",
    [
        "agent:main:main",  # local gateway session
        "agent:main:hook:email:",  # system hook session
        "agent:main:discord:channel:123456",  # owner-run channel
    ],
)
def test_owner_sessions_resolve_to_the_registry_admin(session_id):
    assert resolve_user_from_session_id(session_id) == "alex"


def test_owner_fallback_is_none_without_an_admin(monkeypatch):
    import agent_memory_server.extraction as ex

    monkeypatch.setattr(ex, "_family_admin_user", None)
    assert resolve_user_from_session_id("agent:main:main") is None


@pytest.mark.parametrize("session_id", [None, ""])
def test_empty_session_resolves_to_none(session_id):
    assert resolve_user_from_session_id(session_id) is None


# --- every reader honours settings.family_json_path -------------------------


@pytest.fixture
def other_roster(tmp_path, monkeypatch):
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps(
            {
                "users": {
                    "quinn": {
                        "displayName": "Quinn",
                        "lastName": "Sample",
                        "role": "admin",
                    },
                    "drew": {
                        "displayName": "Drew",
                        "lastName": "Sample",
                        "role": "child",
                    },
                },
                "adminUser": "quinn",
            }
        )
    )
    monkeypatch.setattr(settings, "family_json_path", str(roster))
    monkeypatch.setattr(ms, "_FAMILY_CONTEXT_CACHE", None)
    monkeypatch.setattr(ltm, "_FAMILY_NAME_MAP", None)
    return roster


def test_identity_loader_reads_the_configured_path(other_roster):
    names, _identities, admin = _load_family_registry()
    assert names == {"quinn": "Quinn Sample", "drew": "Drew Sample"}
    assert admin == "quinn"


def test_roster_context_reads_the_configured_path(other_roster):
    context = ms._load_family_context()
    assert "Quinn Sample — the application user / primary speaker" in context
    # The surname comes from the registry, not a hardcoded one.
    assert "Drew Sample — the speaker's child" in context


def test_source_user_normalization_reads_the_configured_path(other_roster):
    assert ltm._normalize_source_user("quinn") == "Quinn Sample"
    assert ltm._normalize_source_user("drew_sample") == "Drew Sample"


def test_source_user_normalization_warns_when_registry_unreadable(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(settings, "family_json_path", str(tmp_path / "missing.json"))
    monkeypatch.setattr(ltm, "_FAMILY_NAME_MAP", None)
    with caplog.at_level(logging.WARNING, logger=ltm.logger.name):
        assert ltm._normalize_source_user("someone") == "someone"
    assert any("family registry" in r.getMessage() for r in caplog.records)
