"""The request log is never deleted: at the size limit it becomes a dated, gzipped archive."""

from __future__ import annotations

import gzip
import json

from summary_fixture import make_records

from aar_gateway import interceptor
from aar_gateway.config import load_settings, log_archives


def _fill(n, *, start=0):
    for r in make_records(n, start=start):
        interceptor._append_record(r)


def test_a_full_log_becomes_an_archive_and_nothing_is_lost(monkeypatch):
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 20_000)
    _fill(300)
    interceptor.join_archivers(10)
    archives = log_archives()
    assert archives and all(p.name.endswith(".jsonl.gz") for p in archives)
    assert [r["input_tokens"] for r in interceptor.read_records()] == [100 + i for i in range(300)]  # all, in order


def test_a_second_rotation_keeps_the_first_archive(monkeypatch):
    """The old '.1' was overwritten by the next rotation: the oldest history vanished."""
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 10_000)
    _fill(500)
    interceptor.join_archivers(10)
    assert len(log_archives()) >= 3
    assert len(interceptor.read_records()) == 500


def test_totals_do_not_change_when_the_log_rotates(monkeypatch):
    _fill(40)
    before = interceptor.summarize()["totals"]["requests"]
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 1)  # the next append rotates
    _fill(1, start=40)
    interceptor.join_archivers(10)
    assert log_archives() and interceptor.summarize()["totals"]["requests"] == before + 1


def test_the_archive_summary_is_stored_with_the_archive(monkeypatch):
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 20_000)
    _fill(200)
    interceptor.join_archivers(10)
    (first, *_) = log_archives()
    side = first.with_name(first.name + ".summary.json")
    stored = json.loads(side.read_text(encoding="utf-8"))
    with gzip.open(first, "rt", encoding="utf-8") as f:
        assert stored["state"]["total"]["requests"] == len([json.loads(line) for line in f])
    interceptor._archive_cache.clear()
    reads = []
    real = interceptor.iter_jsonl
    monkeypatch.setattr(interceptor, "iter_jsonl", lambda path: reads.append(path) or real(path))
    interceptor.summarize()
    assert first not in reads  # its stored summary was used


def test_an_archive_present_both_plain_and_gzipped_counts_once(tmp_path):
    records = make_records(30)
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    (tmp_path / "requests-20261001T000000.jsonl").write_bytes(data)           # compression was interrupted
    (tmp_path / "requests-20261001T000000.jsonl.gz").write_bytes(gzip.compress(data))
    (tmp_path / "requests-20261001T000000.jsonl.gz.part").write_bytes(b"half")
    assert [p.name for p in log_archives()] == ["requests-20261001T000000.jsonl.gz"]
    assert interceptor.summarize()["totals"]["requests"] == 30


def test_configure_finishes_an_interrupted_compression(tmp_path):
    data = "".join(json.dumps(r) + "\n" for r in make_records(30)).encode()
    plain = tmp_path / "requests-20261001T000000.jsonl"
    plain.write_bytes(data)
    interceptor.configure(load_settings(None), None)
    interceptor.join_archivers(10)
    assert not plain.exists() and plain.with_name(plain.name + ".gz").exists()
    assert len(interceptor.read_records()) == 30


def test_rotation_never_unlinks_anything_else(monkeypatch, tmp_path):
    (tmp_path / "preferences.json").write_text("{}")
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 5_000)
    _fill(200)
    interceptor.join_archivers(10)
    assert (tmp_path / "preferences.json").exists()
