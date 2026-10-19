"""Importing the interceptor never fails, configure() starts from clean state, and memory it keeps is bounded."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from aar_gateway import interceptor
from aar_gateway.config import load_settings

SRC = str(Path(__file__).resolve().parents[1] / "src")  # aar-gateway/src


def test_importing_the_interceptor_survives_a_broken_local_models_file(tmp_path):
    (tmp_path / "local-models.yaml").write_text("models:\n  m: {lane: bogus}\n", encoding="utf-8")
    code = f"import sys; sys.path.insert(0, {SRC!r}); import aar_gateway.interceptor as i; print(sorted(i._settings))"
    env = {"AAR_GATEWAY_LOG": str(tmp_path / "requests.jsonl"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr[-400:]
    assert "models" in done.stdout


def test_configure_clears_every_cache_it_owns():
    interceptor._log_cache["records"].append({"x": 1})
    interceptor._log_cache["offset"] = 99
    interceptor._archive_cache["some-archive"] = ([1, 2], {}, [])
    interceptor._message_models["m"] = "claude-haiku-5-5"
    interceptor._message_models_loaded = True
    interceptor._blocked[("model", "ua")] = ("why", 0.0)
    interceptor.configure(load_settings(None), None)
    assert interceptor._log_cache["records"] == [] and interceptor._log_cache["offset"] == 0
    assert interceptor._archive_cache == {}
    assert not interceptor._message_models and interceptor._message_models_loaded is False
    assert not interceptor._blocked


def test_the_blocked_models_memory_is_bounded():
    class _Req:
        headers = {}

    class _T:
        target = "m"
        request = _Req()

    for i in range(interceptor._MAX_BLOCKED * 2):
        _T.request = type("R", (), {"headers": {"user-agent": f"agent-{i}"}})()
        interceptor._remember_block(_T, 400, "this model does not support the request")
    assert len(interceptor._blocked) == interceptor._MAX_BLOCKED
