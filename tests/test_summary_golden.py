"""/stats for a log that fits one file must not change when summaries become streaming and mergeable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from summary_fixture import make_records

from aar_gateway import interceptor

GOLDEN = Path(__file__).with_name("golden_summary.json")


def _normalised(summary: dict) -> dict:
    return json.loads(json.dumps({**summary, "log": "LOG"}, sort_keys=True))


def _approx(value):
    """Floats within summation-order noise (Python's sum() compensates; a running total does not)."""
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-12, abs=1e-12)
    if isinstance(value, dict):
        return {k: _approx(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_approx(v) for v in value]
    return value


def test_stats_for_a_single_log_file_are_unchanged():
    interceptor.append_jsonl(interceptor.log_path(), *make_records(400))
    assert _normalised(interceptor.summarize()) == _approx(json.loads(GOLDEN.read_text(encoding="utf-8")))


def test_the_same_records_summarised_directly_give_the_same_answer():
    records = make_records(400)
    interceptor.append_jsonl(interceptor.log_path(), *records)
    assert _normalised(interceptor.summarize(records)) == _normalised(interceptor.summarize())
