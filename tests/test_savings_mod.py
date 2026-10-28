"""The Claude Code savings plugin: its files are valid, and setup adds or
removes it only when asked, and never fails setup when Claude Code can't."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from aar_gateway import cli

# the real functions: conftest replaces them with no-ops for every other test
_INSTALL, _REMOVE, _INSTALLED = cli._mod_install, cli._mod_remove, cli._mod_installed


@pytest.fixture(autouse=True)
def real_plugin_functions(monkeypatch):
    monkeypatch.setattr(cli, "_mod_install", _INSTALL)
    monkeypatch.setattr(cli, "_mod_remove", _REMOVE)
    monkeypatch.setattr(cli, "_mod_installed", _INSTALLED)

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods" / "aptic"


def test_the_marketplace_and_plugin_files_are_valid():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert market["name"] == "deuce-labs"
    (entry,) = market["plugins"]
    assert entry["name"] == "aptic" and entry["source"] == "./mods/aptic"
    assert (MOD / entry["source"].removeprefix("./mods/aptic") / ".claude-plugin" / "plugin.json").exists()
    assert json.loads((MOD / ".claude-plugin" / "plugin.json").read_text())["name"] == "aptic"
    assert json.loads((MOD / "hooks" / "hooks.json").read_text())["modules"] == ["./register.tsx"]


def test_the_mod_only_talks_to_the_local_gateway_and_never_changes_settings():
    code = (MOD / "hooks" / "register.tsx").read_text()
    # its own requests go only to the local gateway; the other two URLs are the source uv
    # installs the gateway from, and the help link in its "needs uv" message. The SVG xmlns
    # in the inline icons is a namespace id, not a request, so it is not a URL here.
    assert "$.http.fetch(GATEWAY + path" in code and code.count("$.http.fetch(") == 1
    urls = set(re.findall(r"https?://[^'\"`\s#]+", code)) - {"http://www.w3.org/2000/svg"}
    assert urls == {
        "http://127.0.0.1:8787", "https://github.com/wkukka1/aar-gateway@main", "https://docs.astral.sh/uv/"}
    assert "/api/preferences" not in code and "/v1/messages" not in code and "/api/shutdown" not in code


class _Claude:
    """Stands in for the `claude` command: records each call and answers as asked."""

    def __init__(self, version="2.1.299 (Claude Code)", fail_install=False):
        self.calls, self.version, self.fail_install = [], version, fail_install

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[1:])
        if cmd[1:] == ["--version"]:
            return subprocess.CompletedProcess(cmd, 0, self.version + "\n", "")
        if cmd[1:2] == ["plugin"] and cmd[2:3] == ["install"] and self.fail_install:
            return subprocess.CompletedProcess(cmd, 1, "", "plugin not found")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def claude(monkeypatch):
    stub = _Claude()
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(cli.subprocess, "run", stub)
    return stub


def test_install_adds_the_marketplace_then_the_plugin_at_user_scope(claude):
    assert cli._mod_install() is True
    assert ["plugin", "marketplace", "add", "wkukka1/aar-gateway"] in claude.calls
    assert ["plugin", "install", "aptic@deuce-labs", "--scope", "user"] in claude.calls


def test_an_old_claude_code_is_told_how_to_update_and_setup_continues(monkeypatch, capsys):
    stub = _Claude(version="2.1.200 (Claude Code)")
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/claude")
    monkeypatch.setattr(cli.subprocess, "run", stub)
    assert cli._mod_install() is False
    assert not any("install" in c for c in stub.calls)
    assert "needs Claude Code 2.1.287" in capsys.readouterr().out


def test_no_claude_code_skips_the_plugin_quietly(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: None)
    assert cli._mod_install() is False
    assert "isn't installed" in capsys.readouterr().out


def test_a_failed_install_never_raises(monkeypatch, capsys):
    stub = _Claude(fail_install=True)
    monkeypatch.setattr(cli.shutil, "which", lambda name, path=None: "/usr/bin/claude")
    monkeypatch.setattr(cli.subprocess, "run", stub)
    assert cli._mod_install() is False
    assert "not installed (plugin not found)" in capsys.readouterr().out


def test_setup_asks_once_and_only_a_yes_installs(monkeypatch, claude, capsys):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli, "_mod_installed", lambda: False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "" if "Show your gateway savings" in prompt else "n")
    cli.main(["setup", "--no-enable", "--no-share"])
    assert not any("install" in c for c in claude.calls)
    monkeypatch.setattr("builtins.input", lambda prompt: "y" if "Show your gateway savings" in prompt else "n")
    cli.main(["setup", "--no-enable", "--no-share"])
    assert ["plugin", "install", "aptic@deuce-labs", "--scope", "user"] in claude.calls


def test_the_flags_decide_without_asking(monkeypatch, claude):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli, "_mod_installed", lambda: False)
    cli.main(["setup", "--no-enable", "--no-share", "--mod"])
    assert ["plugin", "install", "aptic@deuce-labs", "--scope", "user"] in claude.calls


def test_disable_removes_the_plugin(monkeypatch, claude, tmp_path):
    monkeypatch.setattr(cli, "_claude_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(cli, "_service", lambda action, run=None: 0)
    cli._mod_remove()
    assert ["plugin", "marketplace", "remove", "deuce-labs"] in claude.calls


def test_the_mods_own_tests_pass():
    """mods/aptic/tests run under Claude Code's test kit (needs `claude`)."""
    import shutil as sh

    if sh.which("claude") is None:
        pytest.skip("Claude Code isn't installed")
    done = subprocess.run(["claude", "plugin", "test", str(MOD)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
