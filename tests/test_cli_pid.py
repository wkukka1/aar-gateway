"""`aar-gateway stop` must only ever signal a process that is the gateway."""

from __future__ import annotations

import subprocess
import sys

import pytest

from aar_gateway import cli


@pytest.fixture
def unrelated_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    proc.kill()
    proc.wait()


@pytest.fixture(params=[True, False], ids=["windows", "posix"])
def os_flavour(request, monkeypatch):
    monkeypatch.setattr(cli, "WINDOWS", request.param)


def test_a_stale_pid_file_naming_a_reused_pid_is_not_a_running_gateway(unrelated_process, os_flavour):
    cli._pid_file().parent.mkdir(parents=True, exist_ok=True)
    cli._pid_file().write_text(str(unrelated_process.pid))  # the gateway died; its pid was reused
    assert cli._running_pid() is None
    assert unrelated_process.poll() is None  # probing must not have touched it


def test_stop_leaves_an_unrelated_process_alone(unrelated_process, os_flavour, capsys):
    cli._pid_file().parent.mkdir(parents=True, exist_ok=True)
    cli._pid_file().write_text(str(unrelated_process.pid))
    assert cli._stop() is False
    assert unrelated_process.poll() is None
    assert "not running" in capsys.readouterr().out


def test_the_pid_the_gateway_reports_wins_over_the_file(monkeypatch):
    cli._pid_file().parent.mkdir(parents=True, exist_ok=True)
    cli._pid_file().write_text("111")
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: {"pid": 222})
    assert cli._running_pid() == 222
