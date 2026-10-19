"""The server's own endpoints and its Claude Code launcher -- the routing
itself is covered by test_interceptor*.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aar_gateway import cli, interceptor, server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.delenv("AAR_GATEWAY_CONFIG", raising=False)
    with TestClient(server.app) as c:
        yield c


def test_health_reports_the_router(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["router"]["enabled"] is False  # no config: defaults have the router off
    assert "claude-opus-5" in body["router"]["models"]


def test_stats_on_an_empty_log(client):
    totals = client.get("/stats").json()["totals"]
    assert totals["requests"] == 0 and totals["cost_usd"] == 0.0


def test_statusline_before_any_turn(client):
    assert "waiting" in client.get("/api/statusline", params={"session_id": "nope"}).json()["line"]


def test_claude_is_launched_with_tool_search_and_a_status_line(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))  # no user settings -> no existing status line
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENABLE_TOOL_SEARCH", raising=False)
    args, env = cli._claude_command("http://127.0.0.1:8787", ["-p", "hi"])
    assert env["ENABLE_TOOL_SEARCH"] == "true"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert args[0] == "claude" and args[-2:] == ["-p", "hi"]
    settings = json.loads(args[args.index("--settings") + 1])
    assert settings["statusLine"]["command"].endswith("-m aar_gateway.cli statusline")


def test_an_existing_status_line_is_left_alone():
    settings = Path.home() / ".claude" / "settings.json"  # a temp folder (conftest)
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"statusLine": {"type": "command"}}))
    args, _ = cli._claude_command("http://x", [])
    assert "statusLine" not in json.loads(args[args.index("--settings") + 1])


def test_a_users_own_tool_search_setting_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "false")
    assert cli._claude_command("http://x", [])[1]["ENABLE_TOOL_SEARCH"] == "false"


def test_the_log_rotates(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 5)
    interceptor._append_record({"a": 1})
    interceptor._append_record({"b": 2})
    interceptor.join_archivers(10)
    assert len(list(tmp_path.glob("requests-*.jsonl.gz"))) == 1  # dated and gzipped, never overwritten
    assert (tmp_path / "requests.jsonl").read_text().strip() == '{"b": 2}'
    assert interceptor.read_records() == [{"a": 1}, {"b": 2}]  # the rotated one still counts
