"""The gateway can be asked to stop cleanly (Windows has no SIGTERM: stop used to be TerminateProcess)."""

from __future__ import annotations

import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aar_gateway import cli, server


def test_shutdown_asks_the_running_server_to_exit(monkeypatch):
    fake = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(server.app.state, "uvicorn", fake, raising=False)
    with TestClient(server.app) as client:
        assert client.post("/api/shutdown").json() == {"status": "stopping"}
    assert fake.should_exit is True


def test_shutdown_without_a_server_handle_is_a_503(monkeypatch):
    monkeypatch.setattr(server.app.state, "uvicorn", None, raising=False)
    with TestClient(server.app) as client:
        assert client.post("/api/shutdown").status_code == 503


def test_a_web_page_cannot_shut_the_gateway_down(monkeypatch):
    fake = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(server.app.state, "uvicorn", fake, raising=False)
    with TestClient(server.app) as client:
        assert client.post("/api/shutdown", headers={"origin": "https://evil.example"}).status_code == 403
    assert fake.should_exit is False


def test_stop_on_windows_asks_politely_before_it_kills(monkeypatch, capsys):
    monkeypatch.setattr(cli, "WINDOWS", True)
    monkeypatch.setattr(cli, "_running_pid", lambda: 4242)
    asked = []
    monkeypatch.setattr(cli, "_request_shutdown", lambda: asked.append(1) or True)
    monkeypatch.setattr(cli, "_wait_exit", lambda pid, seconds: True)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)

    def no_kill(*a):
        raise AssertionError("TerminateProcess used although the gateway agreed to stop")
    monkeypatch.setattr(os, "kill", no_kill)
    assert cli._stop() is True and asked == [1]


def test_stop_on_windows_still_kills_a_gateway_that_ignores_the_request(monkeypatch):
    monkeypatch.setattr(cli, "WINDOWS", True)
    monkeypatch.setattr(cli, "_running_pid", lambda: 4242)
    monkeypatch.setattr(cli, "_request_shutdown", lambda: False)
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(cli, "_wait_exit", lambda pid, seconds: True)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    assert cli._stop() is True and killed == [4242]
