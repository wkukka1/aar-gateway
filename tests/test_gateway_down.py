"""Gateway down, in every way it can be down. Whatever happens, a Claude Code
hook must exit 0 quickly, a session must reach Anthropic, and the user must
be told once. Real processes and sockets, no mocks of the failure itself."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from aar_gateway import cli

SRC = str(Path(cli.__file__).resolve().parents[1])  # the src/ folder that holds aar_gateway


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(tmp: Path, port: int, settings: Path, config: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "AAR_GATEWAY_CONFIG"}
    return {**env, "HOME": str(tmp / "home"), "PYTHONPATH": SRC, "AAR_GATEWAY_PORT": str(port),
            "AAR_GATEWAY_LOG": str(tmp / "gw" / "requests.jsonl"),
            "AAR_GATEWAY_CLAUDE_SETTINGS": str(settings), "AAR_GATEWAY_SETTINGS_SETTLE": "0",
            "AAR_GATEWAY_CONFIG": str(config)}


def _hook(env: dict, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "aar_gateway.cli", "check", *args], env=env,
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _settings(tmp: Path, port: int) -> Path:
    path = tmp / "settings.json"
    path.write_text(json.dumps({"model": "opus", "env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}}))
    return path


@pytest.fixture
def down(tmp_path):
    port = _port()
    settings = _settings(tmp_path, port)
    broken = tmp_path / "broken.yaml"
    broken.write_text("interceptor: [not, a, mapping\n")  # a start attempt fails at once
    return tmp_path, port, settings, _env(tmp_path, port, settings, broken)


def test_nothing_listening_falls_back_and_says_so_once(down):
    tmp, port, settings, env = down
    first = _hook(env, "--session")
    assert first.returncode == 0 and "talking to Anthropic directly" in json.loads(first.stdout)["systemMessage"]
    assert json.loads(settings.read_text())["env"].get("ANTHROPIC_BASE_URL") is None
    second = _hook(env)
    assert second.returncode == 0 and second.stdout.strip() == ""  # still down: no repeat alert


def test_a_hung_listener_is_treated_as_down_without_blocking_the_message(down):
    """Something accepts the connection and never answers: the old code waited
    on it for every message."""
    tmp, port, settings, env = down
    server = socket.socket()
    server.bind(("127.0.0.1", port))
    server.listen(50)  # accepts into the backlog; nothing ever reads or replies
    try:
        started = time.monotonic()
        result = _hook(env)
        took = time.monotonic() - started
        assert result.returncode == 0
        assert took < 40, f"hook held the message for {took:.0f}s"
        assert (tmp / "gw" / "bypassed").exists()
    finally:
        server.close()


def test_a_corrupt_settings_file_never_fails_the_hook(down):
    """SystemExit used to escape the hook's guard: exit 1, which Claude Code can
    treat as blocking the user's message."""
    tmp, port, settings, env = down
    settings.write_text("{not json")
    result = _hook(env)
    assert result.returncode == 0, result.stderr


def test_a_missing_settings_file_never_fails_the_hook(down):
    tmp, port, settings, env = down
    settings.unlink()
    assert _hook(env).returncode == 0


def test_recovery_turns_routing_back_on_once(down):
    tmp, port, settings, env = down
    _hook(env)
    assert (tmp / "gw" / "bypassed").exists()
    import http.server

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        back = _hook(env)
        assert "routing is back on" in json.loads(back.stdout)["systemMessage"]
        assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{port}"
        assert not (tmp / "gw" / "bypassed").exists()
    finally:
        server.shutdown()


def test_a_storm_of_hooks_with_the_gateway_down_all_finish_cleanly(down):
    tmp, port, settings, env = down
    procs = [subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "check"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(25)]
    outs = [p.communicate(timeout=180) for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert json.loads(settings.read_text()).get("env", {}).get("ANTHROPIC_BASE_URL") is None
    assert sum("talking to Anthropic" in out for out, _ in outs) <= 25
