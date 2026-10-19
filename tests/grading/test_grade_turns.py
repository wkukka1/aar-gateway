"""The grader's turns: the ones mined from Claude Code's history file plus the on-device history."""

from __future__ import annotations

from aar_gateway.grading import grade
from aar_gateway import history

RECORD = {"ts": "2026-10-18T10:00:00+00:00", "status": 200, "routed_model": "claude-haiku-5-5",
          "requested_model": "claude-opus-5-5", "subagent": False}


def test_without_a_file_the_grader_takes_turns_jsonl_and_the_history():
    mined = {"prompt_hash": "aaaa", "prompt": "old message", "cwd": "/w", "subagent": False}
    grade.append_jsonl(grade._home() / "turns.jsonl", mined)
    history.record(RECORD, "new message", "/w2")
    prompts = {t["prompt"] for t in grade._turns(None)}
    assert prompts == {"old message", "new message"}


def test_a_message_in_both_is_graded_once():
    history.record(RECORD, "same message", "/w")
    row = history.read()[0]
    grade.append_jsonl(grade._home() / "turns.jsonl", {**row, "self_contained": False})
    todo = grade.pending_turns(grade._turns(None), [])
    assert todo == []  # the turns.jsonl copy says it cannot be replayed on its own, and wins


def test_an_explicit_turns_file_is_the_only_source(tmp_path):
    only = tmp_path / "pool.jsonl"
    grade.append_jsonl(only, {"prompt_hash": "bbbb", "prompt": "pool prompt"})
    history.record(RECORD, "history message", "/w")
    assert [t["prompt"] for t in grade._turns(only)] == ["pool prompt"]
