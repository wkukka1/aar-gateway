"""Summaries that merge: the sum of the parts is the whole, and an archive is read once."""

from __future__ import annotations

import gzip
import json

import pytest
from summary_fixture import make_records

from aar_gateway import interceptor, rollup
from aar_gateway.config import iter_jsonl, log_archives


def _state(records):
    state = rollup.new_state()
    for r in records:
        rollup.accumulate(state, r)
    return state


def _finish(state, records):
    return rollup.finish(state, recent=records[-200:], log="LOG", default_strongest="claude-opus-5-5")


def _same(a, b):
    assert json.loads(json.dumps(a, sort_keys=True)) == _approx(json.loads(json.dumps(b, sort_keys=True)))


def _approx(value):
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-9, abs=1e-12)
    if isinstance(value, dict):
        return {k: _approx(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_approx(v) for v in value]
    return value


@pytest.mark.parametrize("cut", [0, 1, 7, 133, 399, 400])
def test_merging_two_halves_is_the_same_as_summarising_the_whole(cut):
    records = make_records(400)
    older, newer = records[:cut], records[cut:]
    merged = rollup.merge(_state(older), _state(newer))
    _same(_finish(merged, records), _finish(_state(records), records))


def test_merging_three_logs_in_order():
    records = make_records(600)
    a, b, c = records[:150], records[150:400], records[400:]
    merged = rollup.merge(rollup.merge(_state(a), _state(b)), _state(c))
    _same(_finish(merged, records), _finish(_state(records), records))


def test_merge_does_not_change_its_inputs():
    a, b = _state(make_records(50)), _state(make_records(50, start=50))
    before = json.dumps([a, b], sort_keys=True)
    rollup.merge(a, b)
    assert json.dumps([a, b], sort_keys=True) == before


def test_a_state_survives_being_stored_as_json():
    state = _state(make_records(200))
    again = json.loads(json.dumps(state))
    more = make_records(100, start=200)
    _same(_finish(rollup.merge(again, _state(more)), make_records(300)), _finish(_state(make_records(300)), make_records(300)))


def _archive(tmp_path, name, records, *, gz=False):
    path = tmp_path / name
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def test_archives_of_every_kind_are_found_oldest_first_and_summaries_are_not_archives(tmp_path):
    log = interceptor.log_path()
    _archive(tmp_path, "requests-20261002T000000.jsonl.gz", [], gz=True)
    _archive(tmp_path, "requests-20261001T000000.jsonl", [])
    _archive(tmp_path, log.name + ".1", [])
    (tmp_path / "requests-20261003T000000.jsonl.gz.part").write_bytes(b"x")
    (tmp_path / "requests-20261001T000000.jsonl.summary.json").write_text("{}")
    assert [p.name for p in log_archives()] == [log.name + ".1", "requests-20261001T000000.jsonl",
                                                "requests-20261002T000000.jsonl.gz"]


def test_iter_jsonl_reads_gzip_and_skips_a_bad_line(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"n": 1}\n{oops\n{"n": 2}\n'))
    assert list(iter_jsonl(path)) == [{"n": 1}, {"n": 2}]


def test_stats_cover_archives_and_the_live_log_and_read_each_archive_once(tmp_path, monkeypatch):
    records = make_records(600)
    _archive(tmp_path, "requests-20261001T000000.jsonl", records[:200])
    _archive(tmp_path, "requests-20261002T000000.jsonl.gz", records[200:450], gz=True)
    interceptor.append_jsonl(interceptor.log_path(), *records[450:])
    expected = _finish(_state(records), records)
    expected["log"] = str(interceptor.log_path())

    _same(interceptor.summarize(), expected)
    assert interceptor.summarize()["recent"] == list(reversed(records[-200:]))
    assert (tmp_path / "requests-20261001T000000.jsonl.summary.json").exists()

    interceptor._archive_cache.clear()  # a restart: the stored summaries are used, the archives are not read
    reads = []
    real = interceptor.iter_jsonl
    monkeypatch.setattr(interceptor, "iter_jsonl", lambda path: reads.append(path) or real(path))
    interceptor._summary_cache.update(key=None, value=None)
    _same(interceptor.summarize(), expected)
    assert reads == []


def test_an_archive_that_changed_is_summarised_again(tmp_path):
    records = make_records(120)
    path = _archive(tmp_path, "requests-20261001T000000.jsonl", records[:100])
    assert interceptor.summarize()["totals"]["requests"] == 100
    with path.open("ab") as f:
        f.write(b"".join((json.dumps(r) + "\n").encode() for r in records[100:]))
    assert interceptor.summarize()["totals"]["requests"] == 120


def test_recent_comes_from_the_archive_when_the_live_log_is_short(tmp_path):
    records = make_records(300)
    _archive(tmp_path, "requests-20261001T000000.jsonl", records[:290])
    interceptor.append_jsonl(interceptor.log_path(), *records[290:])
    recent = interceptor.summarize()["recent"]
    assert len(recent) == 200 and recent[0] == records[-1] and recent[-1] == records[100]


def test_the_default_profile_is_seeded_from_archived_traffic_after_a_restart(tmp_path):
    from aar_gateway.config import load_settings

    rec = {"status": 200, "lane": "anthropic", "reason": "x", "input_tokens": 10, "output_tokens": 5,
           "cache_creation_input_tokens": 0}
    _archive(tmp_path, "requests-20261001T000000.jsonl", [rec] * 3)
    interceptor.configure(load_settings(None), None)
    assert interceptor._global_profile
