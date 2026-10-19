"""Past Claude Code messages from the local transcripts: only what the person
typed, once each."""

from __future__ import annotations

import json

from aar_gateway.grading.history import history_turns


def line(text, kind="human", **extra):
    return json.dumps({"type": "user", "origin": {"kind": kind}, "cwd": "/repo", "timestamp": "2026-10-01T00:00:00Z",
                       "message": {"role": "user", "content": text}, **extra})


def test_only_typed_messages_once_each(tmp_path):
    proj = tmp_path / "-Users-me-repo"
    proj.mkdir()
    (proj / "s1.jsonl").write_text("\n".join([
        line("refactor the pricing module into a dataclass"),
        line("refactor the pricing module into a dataclass"),           # continued session repeats it
        line("<system-reminder>x</system-reminder>explain how the cache pricing works"),
        line("continue"),                                                # too short to grade
        line("/model opus please do the thing now"),                     # slash command
        line("a scripted prompt that nobody typed by hand", kind=None),  # not human
        line("a subagent's prompt that nobody typed by hand", isSidechain=True),
        line([{"type": "tool_result", "content": "ok"}]),
    ]))
    tmp = tmp_path / "-private-tmp-x"
    tmp.mkdir()
    (tmp / "s.jsonl").write_text(line("a prompt from a temporary folder session"))
    rows = history_turns(tmp_path)
    assert [r["prompt"] for r in rows] == ["refactor the pricing module into a dataclass",
                                           "explain how the cache pricing works"]
    assert rows[0]["cwd"] == "/repo" and rows[0]["source"] == "history"


def test_subagent_baseline_prices_each_run_on_its_roles_model(tmp_path):
    from aar_gateway.grading.subagent_baseline import price_run, report, subagent_runs
    from aar_gateway.config import load_settings

    folder = tmp_path / "proj" / "sess" / "subagents"
    folder.mkdir(parents=True)
    usage = {"input_tokens": 100, "output_tokens": 1000, "cache_read_input_tokens": 10_000}
    lines = [{"type": "user", "message": {"role": "user", "content": "Label each prompt with a task type"}},
             {"type": "assistant", "message": {"model": "claude-opus-5-5", "usage": usage}}]
    (folder / "agent-a1.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    (folder / "agent-a1.meta.json").write_text(json.dumps({"agentType": "general-purpose"}))
    (run,) = subagent_runs(tmp_path)
    row = price_run(run, load_settings(None))
    assert (row["role"], row["source"], row["tiered_model"]) == ("lookup", "task", "claude-haiku-5-5")
    assert 0 < row["tiered_cost_usd"] < row["cost_usd"]
    assert "lookup" in report([row])


def test_only_subagent_tasks_whose_inputs_still_exist_are_replayed(tmp_path):
    from aar_gateway.grading.grade import pending_turns
    from aar_gateway.grading.subagent_baseline import subagent_turns

    folder = tmp_path / "proj" / "sess" / "subagents"
    folder.mkdir(parents=True)
    usage = {"input_tokens": 1, "output_tokens": 1}
    for name, task in (("a1", f"Find X in {tmp_path}/proj"), ("a2", "Label the lines of /private/tmp/gone.txt")):
        lines = [{"type": "user", "cwd": str(tmp_path), "message": {"role": "user", "content": task}},
                 {"type": "assistant", "message": {"model": "claude-sonnet-5-5", "usage": usage}}]
        (folder / f"agent-{name}.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    (row,) = subagent_turns(tmp_path)
    assert (row["agent_role"], row["subagent"], row["cwd"]) == ("lookup", True, str(tmp_path))
    mine = {"prompt_hash": "h", "prompt": "my own question"}
    assert pending_turns([row, mine], [], subagents=True) == [row]
    assert pending_turns([row, mine], [], subagents=False) == [mine]
