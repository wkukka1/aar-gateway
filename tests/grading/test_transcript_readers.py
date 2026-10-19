"""Claude Code's transcripts can have a bad line (a crash mid-write); the rest of the file still counts."""

from __future__ import annotations

import json

from aar_gateway.grading.history import history_turns
from aar_gateway.grading.subagent_baseline import subagent_runs


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(line if isinstance(line, bytes) else (line + "\n").encode() for line in lines))


def _user(text):
    return json.dumps({"type": "user", "origin": {"kind": "human"}, "message": {"content": text}, "timestamp": "t"})


def test_one_corrupt_subagent_line_does_not_drop_the_whole_run(tmp_path):
    usage = {"input_tokens": 10, "output_tokens": 2}
    assistant = json.dumps({"type": "assistant", "cwd": "/w", "message": {"model": "claude-haiku-5-5", "usage": usage}})
    _write(tmp_path / "proj" / "sess" / "subagents" / "agent-1.jsonl", [
        json.dumps({"type": "user", "message": {"content": "look into the loader"}}),
        "{this line was cut short by a crash",
        assistant, assistant])
    (run,) = subagent_runs(tmp_path)
    assert run["task"] == "look into the loader" and len(run["messages"]) == 2


def test_a_transcript_with_undecodable_bytes_still_yields_its_good_messages(tmp_path):
    good = "please refactor the request loader to stream its records"
    _write(tmp_path / "proj" / "session.jsonl", [b'\xff\xfe{"broken": \n', _user(good)])
    assert [r["prompt"] for r in history_turns(tmp_path)] == [good]


def test_a_transcript_with_a_corrupt_line_keeps_the_messages_around_it(tmp_path):
    first, second = "first long enough message about the router", "second long enough message about the gateway"
    _write(tmp_path / "proj" / "session.jsonl", [_user(first), "{oops", _user(second)])
    assert [r["prompt"] for r in history_turns(tmp_path)] == [first, second]
