"""One corrupt line in a gateway jsonl file must not take the gateway down."""

from __future__ import annotations

import json
import logging

import pytest

from aar_gateway import interceptor, share
from aar_gateway.config import load_settings, read_jsonl
from aar_gateway.graded import load_graded


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_read_jsonl_skips_a_corrupt_line_wherever_it_is(tmp_path, where, caplog):
    good = [json.dumps({"n": 1}), json.dumps({"n": 2})]
    lines = {"first": ["{bad"] + good, "middle": [good[0], "{bad", good[1]], "last": good + ["{bad"]}[where]
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert read_jsonl(path) == [{"n": 1}, {"n": 2}]
    assert str(path) in caplog.text and "{bad" not in caplog.text  # where, never what


def test_read_jsonl_skips_undecodable_bytes(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"n": 1}\n\xff\xfe{"n"\n{"n": 2}\n')
    assert read_jsonl(path) == [{"n": 1}, {"n": 2}]


def test_read_jsonl_an_unreadable_path_raises(tmp_path):
    with pytest.raises(OSError):
        read_jsonl(tmp_path)  # a directory: unreadable, not "empty"


def test_a_corrupt_request_log_line_does_not_stop_startup_or_stats(tmp_path):
    log = interceptor.log_path()
    log.write_text('{"model": "a"}\n{corrupt\n{"model": "b"}\n', encoding="utf-8")
    interceptor.configure(load_settings(None), None)
    assert [r["model"] for r in interceptor.read_records()] == ["a", "b"]


def test_a_corrupt_graded_line_does_not_stop_load_graded(tmp_path):
    p = tmp_path / "graded.jsonl"
    row = {"prompt_hash": "h", "prompt": "p", "model": "m", "score": 1.0}
    p.write_text(json.dumps(row) + "\n{corrupt\n", encoding="utf-8")
    prompts, models, scores = load_graded(p)
    assert prompts == ["p"] and models == ["m"]


def test_a_corrupt_share_queue_line_is_skipped():
    share.queue_path().parent.mkdir(parents=True, exist_ok=True)
    share.queue_path().write_text('{"a": 1}\n{corrupt\n', encoding="utf-8")
    assert share.queued() == [{"a": 1}]


def test_compacting_thread_models_after_a_restart_keeps_the_persisted_entries(monkeypatch):
    path = interceptor._thread_models_path()
    monkeypatch.setattr(interceptor, "_MAX_MESSAGES", 10)  # compacts once the file passes 2000 bytes
    old = [{"m": f"old{i}", "model": "m" * 200} for i in range(9)]  # persisted before the restart
    path.write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
    interceptor._message_models.clear()  # a restart: memory empty, nothing looked up yet
    interceptor._message_models_loaded = False
    interceptor._remember_message("new", "x")  # pushes the file past the threshold -> compaction
    kept = {json.loads(line)["m"] for line in path.read_text(encoding="utf-8").splitlines()}
    assert kept == {f"old{i}" for i in range(9)} | {"new"}
