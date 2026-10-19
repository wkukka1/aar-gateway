"""Every gateway test gets its own log folder and home folder, so none of
them reads or writes the real ~/.aar-gateway (preferences, overrides,
request log) or ~/.claude (settings, agents)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_gateway_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    # a port nothing listens on: a test must never reach (or stop) the
    # gateway this machine is really running on 8787
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        monkeypatch.setenv("AAR_GATEWAY_PORT", str(sock.getsockname()[1]))
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    # FastAPI's TestClient calls itself "testserver"; the real gateway only
    # answers to localhost names (server._local_callers_only)
    monkeypatch.setenv("AAR_GATEWAY_ALLOWED_HOSTS", "testserver")
