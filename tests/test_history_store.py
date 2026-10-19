"""The on-device history: every user turn with its prompt text and outcome, kept for good, never uploaded."""

from __future__ import annotations

import gzip
import hashlib
import json
import os

import pytest
from summary_fixture import make_records

from aar_gateway import history, interceptor
from aar_gateway.config import read_preferences


def _row(i, prompt=None):
    return history.build_row(make_records(1, start=i)[0], prompt or f"message {i}", f"/work/{i}")


def test_it_is_on_by_default_and_the_choice_survives_other_preferences():
    assert history.enabled() is True
    interceptor.configure({**interceptor._settings}, None)
    interceptor.set_allow_upgrade(True)
    history.set_enabled(False)
    assert history.enabled() is False and read_preferences()["allow_upgrade"] is True
    history.set_enabled(True)
    assert history.enabled() is True


def test_a_row_carries_the_text_and_the_outcome_in_the_shape_the_grader_reads():
    row = _row(7, "fix the loader")
    assert row["prompt"] == "fix the loader" and row["cwd"] == "/work/7"
    assert row["prompt_hash"] == hashlib.sha1(b"fix the loader").hexdigest()[:16]
    assert {"ts", "conversation", "requested_model", "routed_model", "status", "cost_usd", "subagent"} <= set(row)


def test_append_then_read_returns_the_newest_first_across_archives(monkeypatch):
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", 3_000)
    for i in range(60):
        history.append(_row(i))
    interceptor.join_archivers(10)
    assert history.archives()  # it rotated, gzipped, and kept everything
    assert [r["prompt"] for r in history.read(5)] == [f"message {i}" for i in (59, 58, 57, 56, 55)]
    assert len(history.read()) == 60
    assert history.stats()["rows"] == 60


def test_nothing_is_written_when_it_is_off():
    history.set_enabled(False)
    history.record(make_records(1)[0], "secret prompt", "/work")
    assert not history.history_path().exists()


def test_a_full_disk_never_raises(monkeypatch):
    history.history_path().parent.mkdir(parents=True, exist_ok=True)
    history.history_path().mkdir()  # appending to a directory fails like a full disk
    history.record(make_records(1)[0], "prompt", "/work")  # must not raise


def test_export_and_delete(tmp_path):
    for i in range(5):
        history.append(_row(i))
    out = tmp_path / "out.jsonl"
    assert history.export(out) == 5
    assert [json.loads(line)["prompt"] for line in out.read_text(encoding="utf-8").splitlines()] == [
        f"message {i}" for i in range(5)]
    assert history.delete() == 5
    assert history.read() == [] and not history.history_path().exists()


def test_the_prompt_text_is_never_in_the_share_path():
    from aar_gateway import share

    record = make_records(1)[0]
    shared = share.build_record(record, "a very identifiable prompt text")
    assert "identifiable" not in json.dumps(shared)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_history_is_private_to_the_user():
    history.append(_row(1))
    assert oct(history.history_path().stat().st_mode & 0o777) == "0o600"
    assert oct(history.history_path().parent.stat().st_mode & 0o777) == "0o700"


def test_an_old_archive_is_gzip_and_readable_by_anything(monkeypatch):
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", 1_000)
    for i in range(20):
        history.append(_row(i))
    interceptor.join_archivers(10)
    (first, *_) = history.archives()
    with gzip.open(first, "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["prompt"] == "message 0"
