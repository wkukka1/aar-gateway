"""Many writers at once: Claude Code's settings (every session's hooks change
them), the dashboard's preferences, and the hook's switch to Anthropic and
back. Real processes and threads, no mocks of the race itself."""

from __future__ import annotations

import http.server
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
from aar_gateway.config import atomic_write_text

SRC = str(Path(cli.__file__).resolve().parents[1])  # the src/ folder that holds aar_gateway


def _env(tmp_path: Path, settings: Path, port: int, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "AAR_GATEWAY_CONFIG"}
    return {**env, "HOME": str(tmp_path / "home"), "PYTHONPATH": SRC, "AAR_GATEWAY_PORT": str(port),
            "AAR_GATEWAY_LOG": str(tmp_path / "gw" / "requests.jsonl"),
            "AAR_GATEWAY_CLAUDE_SETTINGS": str(settings), "AAR_GATEWAY_SETTINGS_SETTLE": "0", **extra}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_settings_changes_from_many_processes_are_never_lost(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus", "env": {}}))
    adder = ("import sys\nfrom pathlib import Path\nfrom aar_gateway import cli\n"
             "p = Path(sys.argv[1])\n"
             "with cli._settings_lock(p):\n"
             "    s = cli._read_claude_settings(p)\n"
             "    s['env'][sys.argv[2]] = '1'\n"
             "    cli._write_claude_settings(p, s)\n")
    env = {**os.environ, "PYTHONPATH": SRC}
    procs = [subprocess.Popen([sys.executable, "-c", adder, str(settings), f"KEY_{i}"], env=env) for i in range(20)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    final = json.loads(settings.read_text())
    assert final["model"] == "opus" and sorted(final["env"]) == sorted(f"KEY_{i}" for i in range(20))
    assert not list(tmp_path.glob("*.tmp"))


def test_a_reader_never_sees_half_a_file_while_writers_replace_it(tmp_path):
    path = tmp_path / "prefs.json"
    atomic_write_text(path, json.dumps({"n": -1, "pad": "x" * 20000}))
    stop, bad = threading.Event(), []

    def read():
        while not stop.is_set():
            try:
                json.loads(path.read_text())
            except ValueError as exc:
                bad.append(exc)
            except FileNotFoundError:
                bad.append("missing")
            except PermissionError:
                pass  # Windows, mid-replace: busy for a moment, never partial
            time.sleep(0.001)  # a reader like Claude Code, not a tight loop holding the file open

    reader = threading.Thread(target=read)
    reader.start()
    writers = [threading.Thread(target=lambda i=i: [atomic_write_text(path, json.dumps({"n": i, "pad": "x" * 20000}))
                                                    for _ in range(20)]) for i in range(50)]
    for w in writers:
        w.start()
    for w in writers:
        w.join()
    stop.set()
    reader.join()
    assert bad == [] and json.loads(path.read_text())["n"] in range(50)


def _hold_in_a_process(path: Path) -> subprocess.Popen:
    code = "\n".join([
        "import sys, time",
        "from pathlib import Path",
        "from aar_gateway import cli",
        "with cli._settings_lock(Path(sys.argv[1])):",
        "    print('held', flush=True)",
        "    time.sleep(60)",
    ])
    proc = subprocess.Popen([sys.executable, "-c", code, str(path)], env={**os.environ, "PYTHONPATH": SRC},
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_a_lock_held_by_a_process_that_died_is_free_at_once(tmp_path):
    """The OS drops a dead process's lock: no stale-file timeout to wait out."""
    settings = tmp_path / "settings.json"
    holder = _hold_in_a_process(settings)
    holder.kill()
    holder.wait()
    t0 = time.monotonic()
    with cli._settings_lock(settings, wait=5):
        pass
    assert time.monotonic() - t0 < 3


def test_a_lock_held_by_a_live_process_is_never_taken_over(tmp_path):
    settings = tmp_path / "settings.json"
    holder = _hold_in_a_process(settings)
    try:
        with pytest.raises(SystemExit, match="held by another"):
            with cli._settings_lock(settings, wait=1):
                raise AssertionError("took a lock a live process holds")
    finally:
        holder.kill()
        holder.wait()


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 -- http.server's name
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, *a):
        pass


def _consistent(settings: Path, marker: Path, url: str) -> tuple[bool, bool]:
    """(the address is in the settings, the bypass marker exists): always opposite."""
    has_url = json.loads(settings.read_text())["env"].get("ANTHROPIC_BASE_URL") == url
    return has_url, marker.exists()


def test_many_hooks_at_once_switch_to_anthropic_and_back_consistently(tmp_path):
    """Thirty sessions' hooks firing together, the gateway down and then up:
    the address and the bypass marker must always agree, or routing could
    stay off with nothing left to bring it back."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus", "env": {"ANTHROPIC_BASE_URL": url, "FOO": "1"},
                                    "statusLine": {"type": "command", "command": "x"}}))
    marker = tmp_path / "gw" / "bypassed"
    broken = tmp_path / "broken.yaml"
    broken.write_text("interceptor: [not, a, mapping\n")  # a start attempt fails at once
    env = _env(tmp_path, settings, port, AAR_GATEWAY_CONFIG=str(broken))

    def fire(n):
        procs = [subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "check"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]
        assert all(p.wait(timeout=120) == 0 for p in procs)

    fire(30)  # the gateway is down
    assert _consistent(settings, marker, url) == (False, True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        fire(30)  # it answers again
        assert _consistent(settings, marker, url) == (True, False)
    finally:
        server.shutdown()
    final = json.loads(settings.read_text())
    assert final["env"]["FOO"] == "1" and final["model"] == "opus" and final["statusLine"]["command"] == "x"
