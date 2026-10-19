"""Every restart is a stop followed by a start, in that order."""

from __future__ import annotations

from aar_gateway import cli


def test_restart_stops_then_starts(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "_stop", lambda: calls.append("stop") or True)
    monkeypatch.setattr(cli, "_ensure_running", lambda url, **kw: calls.append(("start", kw.get("quiet"))))
    assert cli.main(["restart"]) == 0
    assert calls == ["stop", ("start", True)] and "restarted" in capsys.readouterr().out


def test_the_helper_is_what_the_commands_use(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_stop", lambda: calls.append("stop") or True)
    monkeypatch.setattr(cli, "_ensure_running", lambda url, **kw: calls.append("start"))
    cli._restart()
    assert calls == ["stop", "start"]
