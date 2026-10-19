"""The aar-gateway command line: process management, setup checks,
start-at-login files, and the Windows-specific paths."""

from __future__ import annotations

import plistlib
import sys

import pytest

from aar_gateway import cli


@pytest.fixture
def nothing_running(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_PORT", "59999")  # nothing listens here
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: None)


def test_status_when_stopped_says_how_to_start(nothing_running, capsys):
    assert cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert "stopped" in out and "aar-gateway start" in out


def test_status_when_running_shows_router_and_savings(monkeypatch, capsys):
    replies = {"/health": {"router": {"enabled": True, "kind": "knn"}},
               "/stats": {"totals": {"requests": 12, "cost_usd": 1.0, "cost_if_requested_usd": 4.0}}}
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: next(
        (v for k, v in replies.items() if k in url), None))
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "running" in out and "router         on" in out and "75% less plan usage" in out


def test_config_defaults_to_the_home_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AAR_GATEWAY_CONFIG", raising=False)
    assert cli.main(["config"]) == 1 and "none" in capsys.readouterr().out
    (tmp_path / "config.yaml").write_text("interceptor: {}\n")  # AAR_GATEWAY_LOG is under tmp_path
    assert cli.main(["config"]) == 0 and capsys.readouterr().out.strip() == str(tmp_path / "config.yaml")
    monkeypatch.setenv("AAR_GATEWAY_CONFIG", str(tmp_path / "other.yaml"))
    cli.main(["config"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "other.yaml")


def test_logs_shows_the_last_lines(tmp_path, capsys):
    (tmp_path / "gateway.log").write_text("".join(f"line {i}\n" for i in range(10)))
    assert cli.main(["logs", "-n", "2"]) == 0
    assert capsys.readouterr().out == "line 8\nline 9\n"


def test_logs_from_windows_print_one_line_ending(tmp_path, capsys):
    (tmp_path / "gateway.log").write_bytes(b"a\r\nb\r\n")
    assert cli.main(["logs", "-n", "1"]) == 0
    assert capsys.readouterr().out == "b\n"


def test_doctor_flags_a_missing_claude_and_a_bad_config(nothing_running, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: None)
    bad = tmp_path / "config.yaml"
    bad.write_text("interceptor:\n  models:\n    m: {lane: carrier-pigeon}\n")
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ Claude Code installed" in out and "✗ config" in out and "carrier-pigeon" in out
    assert "fix:" in out


def test_launchd_and_systemd_restart_on_crash_but_not_after_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_CONFIG", str(tmp_path / "config.yaml"))
    plist = plistlib.loads(cli.launchd_plist().encode())
    assert plist["ProgramArguments"][1:] == ["-m", "aar_gateway.cli", "serve"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"]["AAR_GATEWAY_CONFIG"].endswith("config.yaml")
    unit = cli.systemd_unit()
    assert "Restart=on-failure" in unit and "-m aar_gateway.cli serve" in unit


def test_service_install_and_uninstall_on_macos(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli, "_running_pid", lambda: None)
    monkeypatch.setattr(cli, "_wait_healthy", lambda url, seconds: True)
    monkeypatch.setattr(cli.os, "getuid", lambda: 501, raising=False)  # absent on Windows
    calls = []

    def run(cmd, **kw):  # never touch the real launchctl
        calls.append(cmd)
        return cli.subprocess.CompletedProcess(cmd, 1 if cmd[1] == "print" else 0)  # print: job gone
    assert cli._service("install", run=run) == 0
    plist = tmp_path / "Library" / "LaunchAgents" / "com.aar.gateway.plist"
    assert plist.exists() and calls[-1][:2] == ["launchctl", "bootstrap"]
    assert cli._service("uninstall", run=run) == 0
    assert not plist.exists() and calls[-1][:2] == ["launchctl", "bootout"]


def test_windows_never_probes_with_a_signal_and_needs_no_shell_env(tmp_path, monkeypatch):
    """os.kill(pid, 0) on Windows terminates the process; the hook command
    can't use the POSIX `VAR=value cmd` form there."""
    monkeypatch.setattr(cli, "WINDOWS", True)
    (tmp_path / "gateway.pid").write_text("4242")

    def boom(*a):
        raise AssertionError("os.kill must not be used to probe on Windows")
    monkeypatch.setattr(cli.os, "kill", boom)
    monkeypatch.setattr(cli, "_get", lambda url, timeout=1.0: {"pid": 4242})  # the gateway names itself
    assert cli._running_pid() == 4242
    command = cli._self_command()
    assert "PYTHONPATH=" not in command and command.endswith("-m aar_gateway.cli")


def _fake_ollama(monkeypatch, pulled):
    """Ollama answering /api/tags with ``pulled`` (None: not running); the gateway itself down."""
    def get(url, timeout=1.0):
        if "/api/tags" in url:
            return None if pulled is None else {"models": [{"name": m} for m in pulled]}
        return None
    monkeypatch.setattr(cli, "_get", get)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)


def test_local_status_says_how_to_install_ollama_when_missing(monkeypatch, capsys):
    _fake_ollama(monkeypatch, None)
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: None)
    assert cli.main(["local"]) == 0
    out = capsys.readouterr().out
    assert "not installed" in out and "to install Ollama" in out


def test_local_add_needs_ollama_running(monkeypatch):
    _fake_ollama(monkeypatch, None)
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/ollama")
    with pytest.raises(SystemExit, match="isn't running"):
        cli.main(["local", "add", "llama3.1"])


def test_local_add_pulls_once_adds_and_remove_takes_it_out(monkeypatch, capsys):
    from aar_gateway.config import load_settings, local_models_path

    _fake_ollama(monkeypatch, ["qwen2.5-coder:14b"])
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/ollama")
    pulls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: pulls.append(cmd) or
                        cli.subprocess.CompletedProcess(cmd, 0))
    assert cli.main(["local", "add", "llama3.1"]) == 0
    assert pulls == [["/usr/bin/ollama", "pull", "llama3.1:latest"]]
    models = load_settings(None)["models"]
    assert models["llama3.1:latest"]["lane"] == "openai" and models["llama3.1:latest"]["proxy"] is None
    assert "claude-opus-5-5" in models  # added to the defaults, never replacing them
    assert cli.main(["local", "remove", "llama3.1"]) == 0
    assert "llama3.1:latest" not in load_settings(None)["models"] and not local_models_path().exists()


@pytest.fixture
def claude_settings(monkeypatch):
    """~/.claude/settings.json (home is a temp folder, conftest); no real login service."""
    monkeypatch.setattr(cli, "_service", lambda action, run=None, at_next_login=False: 0)
    monkeypatch.setattr(cli, "_ensure_running", lambda url, quiet=False: False)
    path = cli._claude_settings_path()
    path.parent.mkdir(parents=True)
    return path


def test_enable_keeps_the_users_settings_and_disable_restores_them(claude_settings, capsys):
    import json

    mine = {"model": "opus", "env": {"FOO": "1"},
            "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "lint"}]}]}}
    claude_settings.write_text(json.dumps(mine))
    assert cli.main(["enable"]) == 0 and cli.main(["enable"]) == 0  # twice: nothing added twice
    on = json.loads(claude_settings.read_text())
    assert on["env"]["ANTHROPIC_BASE_URL"] == cli._base_url() and on["env"]["FOO"] == "1"
    assert len(on["hooks"]["SessionStart"]) == 1 and len(on["hooks"]["PreToolUse"]) == 2
    assert on["hooks"]["SessionStart"][0]["hooks"][0]["command"].endswith("check --session")
    assert on["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"].endswith("check")
    assert on["hooks"]["PostToolUse"][0]["hooks"][0]["command"].endswith("check")
    assert "statusline" in on["statusLine"]["command"] and cli.auto_enabled()
    assert (claude_settings.parent / "agents" / "lookup.md").exists()
    assert cli.main(["disable"]) == 0
    assert json.loads(claude_settings.read_text()) == mine and not cli.auto_enabled()


def test_enable_never_replaces_another_proxy_unless_forced(claude_settings):
    import json

    claude_settings.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://other:9000"}}))
    with pytest.raises(SystemExit, match="--force"):
        cli.main(["enable"])
    assert cli.main(["enable", "--force"]) == 0 and cli.auto_enabled()


def test_the_session_start_hook_prints_nothing(monkeypatch, capsys):
    """A SessionStart hook's output becomes part of the conversation."""
    def down(url, quiet=False):
        raise SystemExit("the gateway did not come up")
    monkeypatch.setattr(cli, "_ensure_running", down)
    assert cli.main(["start", "--quiet"]) == 0
    assert capsys.readouterr() == ("", "")


def test_setup_changes_claude_code_only_on_a_yes(claude_settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # Enter: the default is no
    cli.main(["setup"])
    assert not cli.auto_enabled() and "aar-gateway code" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    cli.main(["setup"])
    assert cli.auto_enabled()


def test_setup_without_a_terminal_never_asks_or_changes_anything(claude_settings, monkeypatch):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cli.main(["setup"])
    assert not cli.auto_enabled()
    cli.main(["setup", "--yes"])
    assert cli.auto_enabled()


def test_code_adds_no_second_hook_once_enabled(claude_settings):
    cli.main(["enable"])
    args, env = cli._claude_command(cli._base_url(), [])
    assert "--settings" not in args and env["ANTHROPIC_BASE_URL"] == cli._base_url()


def test_installing_the_service_with_a_gateway_serving_leaves_it_serving(tmp_path, monkeypatch, capsys):
    """Sessions already going through the gateway must never see it go away."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_stop", lambda: pytest.fail("stopped a serving gateway"))
    calls = []
    assert cli._service("install", run=lambda cmd, **kw: calls.append(cmd), at_next_login=True) == 0
    assert calls == [] and (cli.Path.home() / "Library" / "LaunchAgents" / "com.aar.gateway.plist").exists()
    assert "keeps serving" in capsys.readouterr().out


def test_a_session_waits_for_the_login_services_gateway_instead_of_starting_another(monkeypatch):
    monkeypatch.setattr(cli, "_service_running", lambda: True)
    answers = iter([False, False, True])
    monkeypatch.setattr(cli, "_healthy", lambda url: next(answers))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: pytest.fail("started a second gateway"))
    assert cli._ensure_running("http://127.0.0.1:1") is False


def test_stop_ends_a_gateway_that_has_a_dashboard_open(tmp_path, monkeypatch):
    """The dashboard's live stream never closes by itself; a stopped gateway
    used to wait on it forever, still running, port closed."""
    import socket
    import subprocess
    import time

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("AAR_GATEWAY_PORT", str(port))
    monkeypatch.delenv("AAR_GATEWAY_CONFIG", raising=False)
    env = {**cli._src_env(), "AAR_GATEWAY_PORT": str(port), "AAR_GATEWAY_LOG": str(tmp_path / "requests.jsonl")}
    env.pop("AAR_GATEWAY_CONFIG", None)
    proc = subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert cli._wait_healthy(cli._base_url(), 60)
        stream = socket.create_connection(("127.0.0.1", port))
        stream.sendall(b"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        time.sleep(0.5)
        assert cli._stop() is True
        code = proc.wait(timeout=5)
        # a clean exit: dying of SIGTERM reads as a crash to launchd, which restarts it
        assert code == 0 or cli.WINDOWS
    finally:
        if proc.poll() is None:
            proc.kill()


def test_enable_refuses_a_copy_in_a_temporary_folder(claude_settings, monkeypatch):
    """Every session would point at files the system deletes."""
    import tempfile

    monkeypatch.setattr(cli.sys, "prefix", str(cli.Path(tempfile.gettempdir()) / "x"))
    with pytest.raises(SystemExit, match="temporary folder"):
        cli.main(["enable"])
    assert not cli.auto_enabled()
    assert cli.main(["enable", "--force"]) == 0 and cli.auto_enabled()


def test_starting_gives_up_as_soon_as_the_gateway_crashes(monkeypatch):
    class Crashed:
        returncode = 1

        def poll(self):
            return 1
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_service_running", lambda: False)
    monkeypatch.setattr(cli, "_service_start", lambda: False)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: Crashed())
    started = cli.time.monotonic()
    with pytest.raises(SystemExit, match="exited while starting"):
        cli._ensure_running("http://127.0.0.1:1", quiet=True)
    assert cli.time.monotonic() - started < 5


def test_starting_goes_through_the_login_service_when_installed(monkeypatch):
    """So the service, not a stray process, owns it, and restarts it after a crash."""
    monkeypatch.setattr(cli, "_service_running", lambda: False)
    monkeypatch.setattr(cli, "_service_start", lambda: True)
    answers = iter([False, True])
    monkeypatch.setattr(cli, "_healthy", lambda url: next(answers))
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: pytest.fail("started its own gateway"))
    assert cli._ensure_running("http://127.0.0.1:1", quiet=True) is True


def _base_url_in_settings(path):
    import json

    return (json.loads(path.read_text()).get("env") or {}).get("ANTHROPIC_BASE_URL")


def test_a_gateway_that_wont_start_hands_the_session_to_anthropic_and_back(claude_settings, monkeypatch, capsys):
    """Never "connection refused" on every message: Claude Code rereads its
    settings, so without the gateway's address the next request goes to Anthropic."""
    import json

    cli.main(["enable"])
    up = {"now": False}
    monkeypatch.setattr(cli, "_healthy", lambda url: up["now"])

    def cannot_start(url, quiet=False, timeout=90):
        raise SystemExit("the gateway exited while starting")
    monkeypatch.setattr(cli, "_ensure_running", cannot_start)
    spawned = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd))
    capsys.readouterr()

    assert cli.main(["check"]) == 0
    assert _base_url_in_settings(claude_settings) is None
    assert "talking to Anthropic directly" in json.loads(capsys.readouterr().out)["systemMessage"]
    assert cli.auto_enabled()  # still enabled, only bypassed

    assert cli.main(["check"]) == 0  # still down: no notice again, a start tried without waiting
    assert capsys.readouterr().out == "" and spawned and spawned[-1][-2:] == ["start", "--quiet"]

    up["now"] = True
    assert cli.main(["check"]) == 0
    assert _base_url_in_settings(claude_settings) == cli._base_url()
    assert "routing is back on" in json.loads(capsys.readouterr().out)["systemMessage"]


def test_check_starts_a_stopped_gateway_and_says_nothing(claude_settings, monkeypatch, capsys):
    cli.main(["enable"])
    up = {"now": False}
    monkeypatch.setattr(cli, "_healthy", lambda url: up["now"])
    monkeypatch.setattr(cli, "_ensure_running", lambda url, quiet=False, timeout=90: up.update(now=True))
    capsys.readouterr()
    assert cli.main(["check", "--session"]) == 0
    assert capsys.readouterr().out == "" and _base_url_in_settings(claude_settings) == cli._base_url()


def test_disable_while_bypassed_restores_the_settings(claude_settings, monkeypatch):
    import json

    claude_settings.write_text(json.dumps({"model": "opus"}))
    cli.main(["enable"])
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_ensure_running", lambda url, quiet=False, timeout=90: False)
    cli.main(["check"])
    assert cli.main(["disable"]) == 0
    assert json.loads(claude_settings.read_text()) == {"model": "opus"} and not cli.auto_enabled()


def test_check_never_blocks_a_message_on_its_own_error(monkeypatch):
    monkeypatch.setattr(cli, "_check", lambda session_start: 1 / 0)
    assert cli.main(["check"]) == 0


def test_a_second_gateway_started_at_the_same_time_leaves_the_first_one_stoppable(tmp_path, monkeypatch):
    """The second used to overwrite the pid file while loading, then delete it
    on failing to get the port: the first kept serving and `stop` couldn't find it."""
    import socket
    import subprocess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("AAR_GATEWAY_PORT", str(port))
    env = {**cli._src_env(), "AAR_GATEWAY_PORT": str(port), "AAR_GATEWAY_LOG": str(tmp_path / "requests.jsonl")}
    env.pop("AAR_GATEWAY_CONFIG", None)
    serve = [sys.executable, "-m", "aar_gateway.cli", "serve"]
    first = subprocess.Popen(serve, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert cli._wait_healthy(cli._base_url(), 60)
        second = subprocess.run(serve, env=env, capture_output=True, timeout=60)
        assert second.returncode != 0  # port taken: gone at once
        assert cli._running_pid() == first.pid
        assert cli._stop() is True
        first.wait(timeout=10)
    finally:
        if first.poll() is None:
            first.kill()


def test_enable_again_from_a_new_install_points_every_hook_at_it(claude_settings, monkeypatch):
    import json

    monkeypatch.setattr(cli, "_self_command", lambda: "/old/python -m aar_gateway.cli")
    cli.main(["enable"])
    monkeypatch.setattr(cli, "_self_command", lambda: "/new/python -m aar_gateway.cli")
    cli.main(["enable"])
    text = claude_settings.read_text()
    assert "/old/" not in text and json.loads(text)["statusLine"]["command"].startswith("/new/")


def test_the_login_service_only_counts_for_the_port_it_serves(tmp_path, monkeypatch):
    """A gateway on another port must not wait on, or start, the service's."""
    unit = tmp_path / "svc"
    unit.write_text("ExecStart=python -m aar_gateway.cli serve\n")  # no port: the default
    monkeypatch.setenv("AAR_GATEWAY_PORT", "8787")
    assert cli._service_serves_this_port(unit)
    monkeypatch.setenv("AAR_GATEWAY_PORT", "58802")
    assert not cli._service_serves_this_port(unit)
    unit.write_text("<key>AAR_GATEWAY_PORT</key>\n<string>58802</string>\n")
    assert cli._service_serves_this_port(unit)


def test_output_with_marks_survives_a_legacy_code_page(monkeypatch):
    import io

    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(cli.sys, "stdout", out)
    cli._printable_output()
    print("✓ router")  # the doctor's mark: not in cp1252
    out.flush()
    assert out.buffer.getvalue().startswith(b"? router")
