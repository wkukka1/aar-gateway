"""The gateway serves this machine's Claude Code, CLI and dashboard only:
a web page can't drive it (cross-site requests, DNS rebinding), and a router
that fails to load leaves Claude Code working, unrouted."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import interceptor, routing, server


@pytest.fixture
def local(monkeypatch):
    monkeypatch.delenv("AAR_GATEWAY_ALLOWED_HOSTS", raising=False)  # the real rule: localhost names only
    with TestClient(server.app, base_url="http://127.0.0.1:8787") as client:
        yield client


def test_a_web_page_cannot_change_settings_or_send_requests(local, monkeypatch):
    sent = []
    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: sent.append(req) or httpx.Response(200, json={}))))
    evil = {"origin": "https://evil.example", "content-type": "text/plain"}
    try:
        r = local.post("/api/preferences", content=json.dumps({"mode": "off"}), headers=evil)
        assert r.status_code == 403 and interceptor.preferences()["mode"] == "auto"
        r = local.post("/v1/messages", content=json.dumps({"model": "x", "messages": []}), headers=evil)
        assert r.status_code == 403 and sent == []
    finally:
        interceptor.set_upstream_client(None)


def test_dns_rebinding_hosts_are_refused_and_local_names_accepted(local):
    assert local.get("/health", headers={"host": "attacker.example:8787"}).status_code == 403
    for host in ("127.0.0.1:8787", "localhost:8787", "[::1]:8787", "localhost"):
        assert local.get("/health", headers={"host": host}).status_code == 200, host


def test_the_dashboard_itself_and_claude_code_get_through(local):
    same_origin = {"origin": "http://127.0.0.1:8787"}
    assert local.post("/api/preferences", json={"preset": "balanced"}, headers=same_origin).status_code == 200
    assert local.get("/stats").status_code == 200  # Claude Code and the CLI send no Origin


def test_extra_hosts_can_be_allowed(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_ALLOWED_HOSTS", "box.lan, 10.0.0.5")
    with TestClient(server.app, base_url="http://box.lan:8787") as client:
        assert client.get("/health").status_code == 200


def test_a_router_that_fails_to_load_leaves_the_gateway_up_and_says_why(monkeypatch):
    def broken(*args, **kwargs):
        raise FileNotFoundError("data/processed/llmrouterbench/responses.parquet")
    monkeypatch.setattr(routing, "build_picker", broken)
    with TestClient(server.app) as client:
        health = client.get("/health").json()["router"]
        assert health["enabled"] is False and "responses.parquet" in health["error"]
        prefs = client.get("/api/preferences").json()
        assert prefs["router_enabled"] is False and "FileNotFoundError" in prefs["router_error"]
