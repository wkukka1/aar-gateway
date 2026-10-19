"""Files the gateway boots from are replaced atomically: a crash mid-write cannot leave a torn one."""

from __future__ import annotations

import pytest

from aar_gateway import cli, config


@pytest.fixture
def written(monkeypatch):
    calls = []
    real = config.atomic_write_text
    monkeypatch.setattr(config, "atomic_write_text", lambda path, text: (calls.append(path), real(path, text))[1])
    return calls


def test_local_models_file_is_written_atomically(written):
    cli._write_local_models({"qwen2.5:3b": {"lane": "openai", "base_url": "http://127.0.0.1:11434/v1"}})
    assert config.local_models_path() in written
    assert "qwen2.5:3b" in config.local_models_path().read_text(encoding="utf-8")


def test_agent_files_are_written_atomically(written, tmp_path):
    target = tmp_path / "agents"
    cli._agents(["--install", str(target)])
    files = sorted(target.glob("*.md"))
    assert files and all(f in written for f in files)


def test_no_boot_critical_file_is_written_with_a_bare_write_text():
    """config.yaml, local-models.yaml, the service unit, the bypass marker and the agent files
    all go through the atomic writer; only a throwaway writability probe may use write_text."""
    import inspect

    source = inspect.getsource(cli)
    assert source.count(".write_text(") == 1 and 'probe.write_text("ok")' in source
