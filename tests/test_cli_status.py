"""`aar-gateway status` reports the files the gateway really uses and survives a missing pid file."""

from __future__ import annotations

import json

from aar_gateway import cli, config


def _health(monkeypatch):
    def get(url, timeout=1.0):
        if url.endswith("/health"):
            return {"status": "ok", "pid": 4242, "router": {}}
        return {"totals": {}}
    monkeypatch.setattr(cli, "_get", get)


def test_status_names_the_request_log_the_gateway_writes(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "custom-name.jsonl"))
    _health(monkeypatch)
    cli._status(as_json=True)
    assert json.loads(capsys.readouterr().out)["requests_log"] == str(config.log_path())


def test_status_of_a_gateway_that_has_no_pid_file_does_not_crash(monkeypatch, capsys):
    _health(monkeypatch)
    assert not cli._pid_file().exists()
    assert cli._status(as_json=True) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["pid"] == 4242 and "up_since" not in info
