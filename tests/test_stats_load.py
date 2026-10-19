"""/stats walks the whole request log: it must not run on the event loop, redo work it already did, or race itself."""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from aar_gateway import interceptor, rollup, server


def _write(n, start=0):
    interceptor.append_jsonl(interceptor.log_path(), *[
        {"ts": f"2026-10-18T10:00:{i % 60:02d}", "routed_model": "claude-haiku-5-5", "requested_model": "claude-opus-5-5",
         "status": 200, "cost_usd": 0.01, "cost_if_requested_usd": 0.05, "cost_if_strongest_usd": 0.05, "n": i}
        for i in range(start, start + n)])


def test_stats_and_statusline_run_off_the_event_loop(monkeypatch):
    seen = {}
    monkeypatch.setattr(interceptor, "summarize", lambda: seen.setdefault("stats", threading.current_thread().name) and {})
    monkeypatch.setattr(interceptor, "session_summary",
                        lambda sid: seen.setdefault("line", threading.current_thread().name) and {})
    with TestClient(server.app) as client:
        loop_thread = client.portal.call(threading.current_thread).name
        client.get("/stats")
        client.get("/api/statusline")
    assert seen["stats"] != loop_thread and seen["line"] != loop_thread


def test_an_unchanged_log_is_summarised_once(monkeypatch):
    calls = []
    real = rollup.accumulate
    monkeypatch.setattr(rollup, "accumulate", lambda state, r: calls.append(1) or real(state, r))
    _write(5)
    first = interceptor.summarize()
    assert interceptor.summarize() == first and len(calls) == 5  # second call: nothing new, nothing recomputed
    _write(2, start=5)
    assert interceptor.summarize()["totals"]["requests"] == 7 and len(calls) == 7  # only the 2 new records


def test_concurrent_stats_calls_never_double_count_or_drop_records():
    _write(300)
    results, errors = [], []

    def hit():
        try:
            for _ in range(10):
                results.append(interceptor.summarize()["totals"]["requests"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not errors and set(results) == {300}
    assert json.loads(json.dumps(interceptor.summarize()))["totals"]["requests"] == 300


def test_read_records_hands_out_a_list_the_caller_may_change():
    _write(4)
    interceptor.read_records().clear()
    assert len(interceptor.read_records()) == 4
