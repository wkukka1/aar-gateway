"""Bad gateway configs fail at load time, not on someone's first request."""

from __future__ import annotations

import pytest

from aar_gateway.config import load_settings


def write(tmp_path, text):
    path = tmp_path / "gateway.yaml"
    path.write_text(text)
    return path


def test_defaults_without_a_config():
    s = load_settings(None)
    assert s["router"]["enabled"] is False
    assert s["strongest"] == "claude-opus-5-5"  # the first listed, not Opus 5, which costs more
    assert s["models"]["claude-haiku-4-5"]["prices"]["input"] == 1.0


def test_openai_lane_needs_a_base_url(tmp_path):
    with pytest.raises(ValueError, match="base_url"):
        load_settings(write(tmp_path, "interceptor: {models: {m: {lane: openai}}}"))


def test_missing_api_key_env_fails_at_load(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    cfg = write(tmp_path, "interceptor: {models: {m: {lane: openai, base_url: 'http://x', api_key_env: SOME_KEY}}}")
    with pytest.raises(ValueError, match="SOME_KEY"):
        load_settings(cfg)


def test_unknown_lane(tmp_path):
    with pytest.raises(ValueError, match="lane"):
        load_settings(write(tmp_path, "interceptor: {models: {m: {lane: carrier-pigeon}}}"))


def test_router_settings_merge_over_defaults(tmp_path):
    s = load_settings(write(tmp_path, "interceptor: {router: {margin: 0.1}}"))
    assert s["router"]["margin"] == 0.1 and s["router"]["horizon_requests"] == 20
