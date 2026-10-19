"""The phone view: read-only, on one address, and nothing else answers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aar_gateway.view import build_app

TS = "100.64.12.7"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("AAR_GATEWAY_ALLOWED_HOSTS", "testserver")  # the test client's own host
    return TestClient(build_app(TS), base_url=f"http://{TS}:8788")


def test_the_dashboard_and_its_read_only_data_answer(client):
    assert client.get("/dashboard").status_code == 200
    assert client.get("/stats").json()["totals"]["requests"] == 0
    assert client.get("/health").json()["view"] is True
    assert "mode" in client.get("/api/preferences").json()


def test_nothing_can_change_anything(client):
    assert client.post("/api/preferences", json={"mode": "off"}).status_code == 405
    assert client.post("/v1/messages", json={}).status_code == 405
    assert client.post("/api/subagent-spawn", json={}).status_code == 405


def test_other_paths_are_refused(client):
    assert client.get("/api/events").status_code == 405
    assert client.get("/nope").status_code == 405


def test_another_host_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    evil = TestClient(build_app(TS), base_url="http://evil.example:8788")
    assert evil.get("/stats").status_code == 403


def test_the_view_never_writes_the_gateway_pid_file(client, tmp_path):
    client.get("/stats")
    assert not (tmp_path / "gateway.pid").exists()
