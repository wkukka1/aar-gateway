"""`aar-gateway history`: see, export, switch off and delete the on-device history."""

from __future__ import annotations

import json

import pytest

from aar_gateway import cli, history

REC = {"ts": "2026-10-18T10:00:00+00:00", "status": 200, "routed_model": "claude-haiku-5-5",
       "requested_model": "claude-opus-5-5", "subagent": False, "cost_usd": 0.01}


def _keep(n):
    for i in range(n):
        history.record(REC, f"message number {i} " + "x" * 200, "/work")


def test_on_off_and_status(capsys):
    assert cli.main(["history", "off"]) == 0
    assert history.enabled() is False
    assert "off" in capsys.readouterr().out
    assert cli.main(["history", "on"]) == 0 and history.enabled() is True
    _keep(3)
    cli.main(["history", "status"])
    out = capsys.readouterr().out
    assert "on" in out and "3 message" in out and str(history.history_path()) in out


def test_show_prints_the_newest_first_and_truncates_long_prompts(capsys):
    _keep(5)
    cli.main(["history", "show", "2"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert "message number 4" in lines[0] and "message number 3" in lines[1] and len(lines) == 2
    assert all(len(line) < 400 for line in lines)


def test_export_writes_every_row_to_one_file(tmp_path, capsys):
    _keep(4)
    dest = tmp_path / "all.jsonl"
    assert cli.main(["history", "export", str(dest)]) == 0
    assert len(dest.read_text(encoding="utf-8").splitlines()) == 4
    assert "4" in capsys.readouterr().out
    assert json.loads(dest.read_text(encoding="utf-8").splitlines()[0])["prompt"].startswith("message number 0")


def test_delete_asks_first_and_only_a_yes_deletes(monkeypatch, capsys):
    _keep(3)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    cli.main(["history", "delete"])
    assert len(history.read()) == 3
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    cli.main(["history", "delete"])
    assert history.read() == []


def test_delete_without_a_terminal_needs_yes_flag(monkeypatch):
    _keep(2)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit):
        cli.main(["history", "delete"])
    assert len(history.read()) == 2
    assert cli.main(["history", "delete", "--yes"]) == 0 and history.read() == []


def test_status_reports_the_history_size(monkeypatch, capsys):
    _keep(2)
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: {"status": "ok", "pid": 1, "router": {}})
    cli._status(as_json=True)
    info = json.loads(capsys.readouterr().out)
    assert info["history"]["enabled"] is True and info["history"]["files"] == 1 and info["history"]["bytes"] > 0
