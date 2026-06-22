"""LAB-64: the REST API entrypoint must default to a loopback bind.

The memory server runs unauthenticated in the homelab; the `python -m
agent_memory_server.main` / `agent-memory api` entrypoint previously hardcoded
host="0.0.0.0", exposing the whole memory corpus on the LAN. It now binds
settings.host (default 127.0.0.1), overridable via the HOST env var.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import agent_memory_server.main as main_mod
from agent_memory_server.config import Settings


def test_settings_host_defaults_to_loopback():
    # A freshly constructed Settings (no HOST env override) binds loopback.
    assert Settings().host == "127.0.0.1"


def test_settings_host_overridable_via_env(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert Settings().host == "0.0.0.0"


def test_main_entrypoint_binds_settings_host_not_wildcard():
    # The __main__ uvicorn.run call must use settings.host, never a hardcoded 0.0.0.0.
    src = inspect.getsource(main_mod)
    main_block = src[src.index('if __name__ == "__main__":') :]
    assert "host=settings.host" in main_block
    assert 'host="0.0.0.0"' not in main_block


def test_no_wildcard_bind_literal_in_main_module():
    # Defense-in-depth: the module file carries no stray 0.0.0.0 REST bind.
    text = Path(main_mod.__file__).read_text()
    assert text.count('host="0.0.0.0"') == 0


def test_cli_api_host_defaults_to_loopback():
    # The `agent-memory api` CLI command must also default to loopback (LAB-64), not 0.0.0.0.
    from agent_memory_server.cli import api

    host_opt = next(p for p in api.params if p.name == "host")
    assert host_opt.default == "127.0.0.1"
