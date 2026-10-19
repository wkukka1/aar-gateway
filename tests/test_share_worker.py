"""The sharing worker is bounded, reports failures by type, cannot hang on an upload, and flushes at shutdown."""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from aar_gateway import share

LOGGED = {"ts": "2026-10-17T13:45:12+00:00", "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest", "status": 200, "subagent": False, "input_tokens": 1, "output_tokens": 1}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(share, "_consent", {"mtime": None, "on": False})
    monkeypatch.setattr(share, "_embed", lambda text: [1] * share.DIMS)
    monkeypatch.setattr(share, "_worker", None)
    monkeypatch.setattr(share, "_pending", 0)
    yield
    share.shutdown(5)


def test_a_backed_up_worker_keeps_every_record_but_skips_the_embedding(monkeypatch, caplog):
    share.set_enabled(True)
    gate, started = threading.Event(), threading.Event()
    real = share._append

    def slow(log_record, human):
        started.set()
        gate.wait(10)
        real(log_record, human)

    monkeypatch.setattr(share, "_append", slow)
    monkeypatch.setattr(share, "MAX_PENDING", 3)
    with caplog.at_level(logging.WARNING):
        for i in range(10):
            share.record(LOGGED, f"message {i}")
    assert started.wait(5)
    assert share._pending <= 3  # memory stays bounded: no more message text is held
    gate.set()
    share.flush(10)
    rows = share.queued()
    assert len(rows) == 10  # nothing was dropped
    assert sum(r["embedding"] is None for r in rows) == 7 and all(r["prompt_chars"] for r in rows)
    assert "without its embedding" in caplog.text


def test_the_share_queue_is_never_trimmed():
    """It used to keep only the newest 20,000 records (a 30 MB file), dropping the oldest unsent ones."""
    assert not hasattr(share, "_cap") and not hasattr(share, "MAX_RECORDS")
    share.set_enabled(True)
    for i in range(30):
        share._queue_line(json.dumps({"n": i, "pad": "x" * 400}) + "\n")
    share._queue_line(json.dumps({"n": 30, "pad": "x" * 400}) + "\n")
    assert [r["n"] for r in share.queued()] == list(range(31))


def test_a_failing_record_is_logged_by_type_never_by_message(monkeypatch, caplog):
    share.set_enabled(True)

    def boom(*a):
        raise ValueError("hf_secret_token_12345")

    monkeypatch.setattr(share, "build_record", boom)
    with caplog.at_level(logging.WARNING):
        share.record(LOGGED, "hello")
        share.flush(5)
    assert "ValueError" in caplog.text and "hf_secret_token_12345" not in caplog.text


def test_an_upload_that_hangs_is_given_up_and_the_records_are_kept(monkeypatch):
    share.set_enabled(True)
    share.set_endpoint("https://collect.example.org/v1/records")
    share.record(LOGGED, "hello")
    share.flush(5)
    monkeypatch.setattr(share, "UPLOAD_TIMEOUT_S", 0.3)

    def hung(url, body, headers):
        time.sleep(5)

    t0 = time.monotonic()
    assert share.upload(post=hung) == {"error": "TimeoutError"}
    assert time.monotonic() - t0 < 3 and len(share.queued()) == 1


def test_shutdown_writes_what_is_pending_then_stops_the_worker():
    share.set_enabled(True)
    for i in range(5):
        share.record(LOGGED, f"message {i}")
    share.shutdown(10)
    assert len(share.queued()) == 5 and share._worker is None


def test_the_queue_length_is_counted_once_not_after_every_record(monkeypatch):
    share.set_enabled(True)
    counted = []
    real = share._count_lines
    monkeypatch.setattr(share, "_count_lines", lambda path: counted.append(1) or real(path))
    for i in range(6):
        assert share._queue_line(json.dumps({"n": i}) + "\n") == i + 1
    assert len(counted) == 1
    other = share.queue_path()  # another process appends: the size no longer adds up, so recount
    with other.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"n": "x"}) + "\n")
    assert share._queue_line(json.dumps({"n": 7}) + "\n") == 8 and len(counted) == 2
