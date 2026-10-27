"""Claude Code subagents: a kind (lookup / make / judge, a built-in, or task
keywords) decides the tier -- set at spawn time by the hook, or at the
gateway for a subagent the hook didn't see -- never above the session's
model, sticky for the subagent's tool loop, logged and summarised per kind."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import cli, interceptor, server
from aar_gateway.config import load_settings
from aar_gateway.subagents import (
    ROLES,
    agent_definitions,
    agent_markdown,
    classify_task,
    detect,
    role_decision,
    spawn_decision,
)

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}
TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
BILLING = "x-anthropic-billing-header: cc_version=2.1.293; cc_entrypoint=cli; cc_is_subagent=true;"
MAIN_BILLING = "x-anthropic-billing-header: cc_version=2.1.293; cc_entrypoint=cli;"
EXPLORE = "You are a file search specialist for Claude Code, Anthropic's official CLI for Claude."
PLAN = "You are a software architect and planning specialist for Claude Code."


def subagent_body(task, system_prompt="You are an agent.", model="claude-opus-5-5", messages=None, **extra):
    return {"model": model, "max_tokens": 64, "tools": [TOOL],
            "system": [{"type": "text", "text": BILLING}, {"type": "text", "text": system_prompt}],
            "messages": messages or [{"role": "user", "content": [{"type": "text", "text": task}]}], **extra}


@pytest.mark.parametrize("task, role", [
    ("Find where CLAUDE_PRICES is defined", "lookup"),
    ("Explore the training pipeline and report the entry points", "lookup"),
    ("Label real user prompts with a task type. Read the file, output labels", "lookup"),
    ("Summarize the failing CI log", "lookup"),
    ("Review the diff in src/router for bugs", "make"),
    ("Find bugs in the interceptor", "make"),
    ("Write unit tests for pricing.py", "make"),
    ("Propose an approach for caching embeddings", "make"),
    ("Generate 100 prompts that a real person would send", "make"),
    ("Write 100 prompts, all examples of the task type SUMMARIZE", "make"),
    ("Implement the cache and find where it is read", "make"),
    ("Find the root cause of the flaky test", "judge"),
    ("Weigh the trade-offs between a queue and polling here", "judge"),
    ("Tell me a joke about compilers", None),
])
def test_tasks_are_classified_by_what_they_ask_for(task, role):
    found = classify_task(task)
    assert (found[0] if found else None) == role


def test_only_requests_marked_as_subagents_are_subagents():
    main = subagent_body("Find X")
    main["system"][0]["text"] = MAIN_BILLING
    assert detect(main, "Find X") is None
    assert detect(subagent_body("Find X"), "Find X")["role"] == "lookup"


def test_definition_beats_builtin_beats_task():
    marked = subagent_body("Find X", system_prompt=agent_definitions()["make"]["prompt"])
    assert detect(marked, "Find X") == {"role": "make", "source": "definition", "why": "aar-role: make"}
    explore = detect(subagent_body("Review this", system_prompt=EXPLORE), "Review this")
    assert (explore["role"], explore["source"]) == ("explore", "builtin")
    plan = detect(subagent_body("Find X", system_prompt=PLAN), "Find X")
    assert (plan["role"], plan["source"]) == ("plan", "builtin")
    assert role_decision(plan, subagent_body("Find X"), "claude-opus-5-5", load_settings(None), {},
                         1000)["model"] == "claude-sonnet-5-5"
    unknown = detect(subagent_body("Tell me a joke"), "Tell me a joke")
    assert unknown == {"role": None, "source": None, "why": None}


def test_every_role_definition_is_recognised_and_has_an_output_budget():
    for name, d in agent_definitions().items():
        assert detect(subagent_body("x", system_prompt=d["prompt"]), "x")["role"] == name
        assert "Output budget:" in d["prompt"]
    assert agent_markdown("lookup").startswith("---\nname: lookup\n")
    assert set(ROLES) == {"lookup", "make", "judge"}
    defs = agent_definitions()
    for read_only in ("lookup", "judge"):  # read and run, never edit
        assert not {"Edit", "Write", "NotebookEdit"} & set(defs[read_only]["tools"])
    assert "tools" not in defs["make"]


def test_role_tier_is_a_ceiling_on_cost_not_an_upgrade():
    settings = load_settings(None)
    agent = {"role": "make", "source": "task", "why": "review"}
    body = subagent_body("Review it")
    assert role_decision(agent, body, "claude-opus-5-5", settings, {}, 1000)["model"] == "claude-sonnet-5-5"
    kept = role_decision(agent, body, "claude-haiku-5-5", settings, {}, 1000)
    assert kept["model"] == "claude-haiku-5-5" and "would cost more" in kept["reason"]
    architect = {"role": "judge", "source": "task", "why": "root cause"}
    assert role_decision(architect, body, "claude-opus-5-5", settings, {}, 1000)["model"] == "claude-opus-5-5"


def test_a_role_model_that_cant_take_the_request_is_left_to_the_router():
    settings = load_settings(None)
    settings["subagents"]["roles"] = {"lookup": "claude-haiku-4-5"}  # rejects `effort`
    agent = {"role": "lookup", "source": "task", "why": "find"}
    body = subagent_body("Find X", output_config={"effort": "high"})
    assert role_decision(agent, body, "claude-opus-5-5", settings, {}, 1000) is None
    assert role_decision({"role": "lookup", "source": "task", "why": "find"}, subagent_body("Find X"),
                         "claude-opus-5-5", load_settings(None), {"claude-haiku-5-5": "400"}, 1000) is None
    assert role_decision({"role": None, "source": None, "why": None}, body, "claude-opus-5-5",
                         load_settings(None), {}, 1000) is None


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        sent.append(body["model"])
        return httpx.Response(200, json={"type": "message", "model": body["model"], "content": [],
                                         "usage": {"input_tokens": 10, "output_tokens": 5,
                                                   "cache_read_input_tokens": 1000}})

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(server.app) as client:
        yield client, sent
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def test_a_subagent_is_routed_by_role_and_its_tool_loop_stays_there(gateway):
    client, sent = gateway
    picked = []
    interceptor.configure(load_settings(None), lambda body, ctx: picked.append(ctx) or None)
    task = "Find where CLAUDE_PRICES is defined"
    first = [{"role": "user", "content": [{"type": "text", "text": task}]}]
    client.post("/v1/messages", json=subagent_body(task, messages=first), headers=AUTH)
    loop = first + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    client.post("/v1/messages", json=subagent_body(task, messages=loop), headers=AUTH)
    assert sent == ["claude-haiku-5-5", "claude-haiku-5-5"]
    assert picked == []  # the role decided; the general router was not asked
    recs = interceptor.read_records()
    assert [(r["subagent"], r["agent_role"], r["role_source"]) for r in recs] == [(True, "lookup", "task")] * 2
    assert recs[1]["reason"] == "same turn as the previous request"

    (row,) = interceptor.summarize()["subagents"]
    assert (row["role"], row["spawns"], row["requests"], row["models"]) == ("lookup", 1, 2, {"claude-haiku-5-5": 2})
    assert row["cost_usd"] < row["cost_if_requested_usd"]
    assert row["cache_hit_rate"] == pytest.approx(1000 / 1010)


def test_an_unrecognised_subagent_goes_to_the_general_router(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), lambda body, ctx: "claude-sonnet-5-5")
    client.post("/v1/messages", json=subagent_body("Tell me a joke about compilers"), headers=AUTH)
    assert sent == ["claude-sonnet-5-5"]
    (rec,) = interceptor.read_records()
    assert (rec["subagent"], rec["agent_role"]) == (True, None)


def test_subagent_roles_can_be_turned_off(gateway, tmp_path):
    client, sent = gateway
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("interceptor:\n  subagents: {enabled: false}\n")
    interceptor.configure(load_settings(cfg), None)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-opus-5-5"]


def test_claude_code_is_given_the_roles_unless_the_user_brings_their_own(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_STATUSLINE", "0")
    args, _ = cli._claude_command("http://127.0.0.1:8787", [])
    assert json.loads(args[args.index("--agents") + 1]) == agent_definitions()
    assert args[args.index("--append-system-prompt") + 1] == cli._DELEGATE_NOTE
    args, _ = cli._claude_command("http://127.0.0.1:8787", ["--agents", "{}"])
    assert args.count("--agents") == 1
    monkeypatch.setenv("AAR_GATEWAY_AGENTS", "0")
    assert "--agents" not in cli._claude_command("http://127.0.0.1:8787", [])[0]


def test_plain_claude_gets_the_delegation_note_in_claude_md_until_disable(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("# My notes\n\nUse uv.\n")
    cli._note_file_set(md)
    cli._note_file_set(md)  # enabling again replaces the block, not a second copy
    text = md.read_text()
    assert text.count(cli._DELEGATE_NOTE) == 1 and text.startswith("# My notes\n\nUse uv.\n")
    cli._note_file_clear(md)
    assert md.read_text() == "# My notes\n\nUse uv.\n"
    solo = tmp_path / "only.md"
    cli._note_file_set(solo)
    cli._note_file_clear(solo)
    assert not solo.exists()


def test_agents_install_writes_files_and_keeps_existing_ones(tmp_path, capsys):
    (tmp_path / "lookup.md").write_text("mine")
    cli.main(["agents", "--install", str(tmp_path)])
    assert (tmp_path / "lookup.md").read_text() == "mine"
    assert (tmp_path / "make.md").read_text(encoding="utf-8") == agent_markdown("make")  # UTF-8, not the console codepage


def main_turn(text, model="claude-opus-5-5"):
    return {"model": model, "max_tokens": 64, "tools": [TOOL], "system": [{"type": "text", "text": MAIN_BILLING}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}


def test_routing_off_keeps_the_asked_model_and_pin_sends_every_agent_turn_to_one(gateway, tmp_path):
    client, sent = gateway
    interceptor.configure(load_settings(None), lambda body, ctx: "claude-haiku-5-5")
    assert client.post("/api/preferences", json={"mode": "off"}).json()["mode"] == "off"
    client.post("/v1/messages", json=main_turn("fix the bug"), headers=AUTH)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-opus-5-5", "claude-opus-5-5"]

    prefs = client.post("/api/preferences", json={"mode": "pin", "pinned_model": "claude-sonnet-5-5"}).json()
    assert (prefs["mode"], prefs["pinned_model"]) == ("pin", "claude-sonnet-5-5")
    client.post("/v1/messages", json=main_turn("another question", model="claude-haiku-5-5"), headers=AUTH)
    assert sent[-1] == "claude-sonnet-5-5"  # a pin may be pricier: the user chose it
    assert interceptor.read_records()[-1]["reason"] == "pinned to claude-sonnet-5-5 (dashboard)"

    assert client.post("/api/preferences", json={"mode": "pin", "pinned_model": "gpt-9"}).status_code == 400
    interceptor.configure(load_settings(None), None)  # a restart keeps the choice
    assert interceptor.preferences()["mode"] == "pin"
    kinds = [json.loads(line)["kind"] for line in interceptor.overrides_path().read_text().splitlines()]
    assert kinds == ["routing", "routing"]


def test_a_pinned_model_that_cant_take_the_request_is_not_forced(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), None)
    client.post("/api/preferences", json={"mode": "pin", "pinned_model": "claude-haiku-4-5"})
    body = main_turn("hi")
    body["output_config"] = {"effort": "high"}  # Haiku 4.5 rejects it
    client.post("/v1/messages", json=body, headers=AUTH)
    assert sent == ["claude-opus-5-5"]
    assert "can't take this request" in interceptor.read_records()[-1]["reason"]


def test_a_role_tier_can_be_changed_reset_and_set_to_the_sessions_model(gateway):
    client, sent = gateway
    interceptor.configure(load_settings(None), None)
    prefs = client.post("/api/preferences", json={"role": "lookup", "model": "claude-sonnet-5-5"}).json()
    scout = next(r for r in prefs["roles"] if r["role"] == "lookup")
    assert (scout["model"], scout["default"], scout["overridden"]) == ("claude-sonnet-5-5", "claude-haiku-5-5", True)
    client.post("/v1/messages", json=subagent_body("Find X"), headers=AUTH)
    assert sent == ["claude-sonnet-5-5"]

    client.post("/api/preferences", json={"role": "lookup", "model": "session"})
    client.post("/v1/messages", json=subagent_body("Find Y"), headers=AUTH)
    assert sent[-1] == "claude-opus-5-5"

    prefs = client.post("/api/preferences", json={"role": "lookup", "model": "default"}).json()
    assert next(r for r in prefs["roles"] if r["role"] == "lookup")["overridden"] is False
    client.post("/v1/messages", json=subagent_body("Find Z"), headers=AUTH)
    assert sent[-1] == "claude-haiku-5-5"
    assert client.post("/api/preferences", json={"role": "wizard", "model": "default"}).status_code == 400


def test_allow_upgrade_reaches_the_router_and_is_kept(gateway):
    client, sent = gateway
    seen = []
    interceptor.configure(load_settings(None), lambda body, ctx: seen.append(ctx["allow_upgrade"]))
    client.post("/v1/messages", json=main_turn("hard question", model="claude-haiku-5-5"), headers=AUTH)
    assert client.post("/api/preferences", json={"allow_upgrade": True}).json()["allow_upgrade"] is True
    client.post("/v1/messages", json=main_turn("another hard one", model="claude-haiku-5-5"), headers=AUTH)
    assert seen == [False, True]
    interceptor.configure(load_settings(None), None)
    assert interceptor.preferences()["allow_upgrade"] is True


def test_sessions_sum_plan_usage_and_count_turns_and_subagents():
    base = {"status": 200, "requested_model": "claude-opus-5-5", "cost_if_strongest_usd": 1.0}
    records = [
        {**base, "ts": "2026-10-10T10:00:00+00:00", "claude_session": "s1", "conversation": "c1", "new_turn": True,
         "routed_model": "claude-sonnet-5-5", "cost_usd": 0.4, "cost_if_requested_usd": 0.8},
        {**base, "ts": "2026-10-10T10:01:00+00:00", "claude_session": "s1", "conversation": "c2", "new_turn": True,
         "subagent": True, "routed_model": "claude-haiku-5-5", "cost_usd": 0.01, "cost_if_requested_usd": 0.2},
        {**base, "ts": "2026-10-10T10:02:00+00:00", "claude_session": "s1", "conversation": "c1", "new_turn": True,
         "reason": "side request (no tools), not routed", "routed_model": "claude-opus-5-5", "cost_usd": 0.01,
         "cost_if_requested_usd": None},
        {**base, "ts": "2026-10-09T09:00:00+00:00", "claude_session": None, "conversation": "c0", "new_turn": True,
         "routed_model": "claude-opus-5-5", "cost_usd": 0.5, "cost_if_requested_usd": 0.5},
    ]
    s1, before = interceptor.summarize_sessions(records)
    assert (s1["session"], s1["requests"], s1["turns"], s1["subagent_spawns"], s1["rerouted"]) == ("s1", 3, 1, 1, 2)
    assert s1["cost_usd"] == pytest.approx(0.42) and s1["cost_if_requested_usd"] == pytest.approx(1.01)
    assert s1["models"] == {"claude-sonnet-5-5": 1, "claude-haiku-5-5": 1, "claude-opus-5-5": 1}
    assert (before["session"], before["requests"]) == (None, 1)


def test_subagent_summary_keeps_a_zero_requested_cost():
    """A subagent request priced at 0.0 at the requested model is a real 0.0,
    not missing: only ``None`` falls back to the billed cost."""
    base = {"subagent": True, "agent_role": "lookup", "role_source": "task", "conversation": "c1",
            "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5"}
    records = [
        {**base, "cost_usd": 0.3, "cost_if_requested_usd": 0.0},
        {**base, "cost_usd": 0.2, "cost_if_requested_usd": None},
    ]
    [row] = interceptor.summarize_subagents(records)
    assert row["cost_if_requested_usd"] == pytest.approx(0.2)


def test_spawn_decision_picks_the_agent_tool_alias_and_respects_choices_already_made():
    settings = load_settings(None)
    d = spawn_decision("general-purpose", "Find where X is defined", "claude-opus-5-5", settings)
    assert (d["role"], d["source"], d["model"], d["alias"]) == ("lookup", "task", "claude-haiku-5-5", "haiku")
    assert spawn_decision("Explore", "anything", "claude-opus-5-5", settings)["alias"] == "haiku"
    assert spawn_decision("Plan", "anything", "claude-opus-5-5", settings)["alias"] == "sonnet"
    assert spawn_decision("make", "x", "claude-opus-5-5", settings)["source"] == "definition"
    assert spawn_decision("judge", "x", "claude-opus-5-5", settings)["alias"] is None      # session's model
    assert spawn_decision("make", "x", "claude-haiku-5-5", settings)["alias"] is None      # never above it
    assert spawn_decision("lookup", "x", "claude-opus-5-5", settings, asked_model="opus")["alias"] is None
    assert spawn_decision("general-purpose", "Tell me a joke", "claude-opus-5-5", settings)["alias"] is None


def test_a_hooked_spawn_is_labelled_and_measured_against_the_sessions_model(gateway):
    """The hook sets the subagent's model, so its requests already ask for
    Haiku; the savings must still be counted against the session's Opus, and
    its final report's size logged."""
    client, sent = gateway
    interceptor.configure(load_settings(None), None)
    main = main_turn("look into the pricing", model="claude-opus-5-5")
    client.post("/v1/messages", json=main, headers={**AUTH, "x-claude-code-session-id": "s1"})
    task = "Find where CLAUDE_PRICES is defined"
    spawn = client.post("/api/subagent-spawn", json={"session_id": "s1", "subagent_type": "general-purpose",
                                                     "prompt": task}).json()
    assert spawn["alias"] == "haiku"
    client.post("/v1/messages", json=subagent_body(task, model="claude-haiku-5-5"),
                headers={**AUTH, "x-claude-code-session-id": "s1"})
    rec = interceptor.read_records()[-1]
    assert (rec["agent_role"], rec["role_source"], rec["routed_model"]) == ("lookup", "hook", "claude-haiku-5-5")
    assert rec["baseline_model"] == "claude-opus-5-5"
    assert rec["cost_if_requested_usd"] > rec["cost_usd"]
    assert rec["report_tokens"] is None  # the mock reply has no stop_reason: not a final answer
    assert interceptor.session_summary("s1")["model"] == "claude-opus-5-5"  # the main agent's, not the subagent's


def test_routing_off_leaves_spawns_alone(gateway):
    client, _ = gateway
    interceptor.configure(load_settings(None), None)
    client.post("/api/preferences", json={"mode": "off"})
    spawn = client.post("/api/subagent-spawn", json={"subagent_type": "Explore", "prompt": "find x"}).json()
    assert spawn["alias"] is None


def test_the_hook_sets_the_model_only_when_the_gateway_says_so(monkeypatch, capsys):
    import io
    import urllib.request

    class Reply:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    event = {"tool_name": "Agent", "session_id": "s1",
             "tool_input": {"subagent_type": "Explore", "prompt": "find x", "description": "d"}}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Reply({"alias": "haiku"}))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    cli.main(["hook"])
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["updatedInput"] == {**event["tool_input"], "model": "haiku"}

    def down(req, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", down)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    cli.main(["hook"])
    assert capsys.readouterr().out == ""  # gateway down: the call is left exactly as it was
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"tool_name": "Bash", "tool_input": {}})))
    cli.main(["hook"])
    assert capsys.readouterr().out == ""


def test_claude_code_is_launched_with_the_spawn_hook(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_STATUSLINE", "0")
    args, _ = cli._claude_command("http://127.0.0.1:8787", [])
    settings = json.loads(args[args.index("--settings") + 1])
    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Agent|Task" and entry["hooks"][0]["command"].endswith("aar_gateway.cli hook")
    monkeypatch.setenv("AAR_GATEWAY_HOOK", "0")
    assert "--settings" not in cli._claude_command("http://127.0.0.1:8787", [])[0]


def test_a_subagent_tier_on_a_model_that_cannot_take_the_request_is_not_forced():
    """Found by audit: a kind's tier set to an openai-lane model skipped the
    lossless-translation check every other route makes."""
    settings = load_settings(None)
    settings["models"]["local-coder"] = {"lane": "openai", "base_url": "http://local.test/v1", "max_tools": 1,
                                         "prices": {"input": 0, "output": 0}}
    settings["subagents"]["roles"] = {"lookup": "local-coder"}
    agent = {"role": "lookup", "source": "task", "why": "find"}
    two_tools = subagent_body("Find X", tools=[TOOL, {**TOOL, "name": "Grep"}])
    assert role_decision(agent, two_tools, "claude-opus-5-5", settings, {}, 1000) is None
    assert role_decision(agent, subagent_body("Find X"), "claude-opus-5-5", settings, {}, 1000)["model"] == "local-coder"


def test_a_subagents_final_streamed_reply_has_its_size_logged():
    """Claude Code streams; the report size comes from the stream's
    stop_reason, so a streamed final answer must set report_tokens and a
    tool-call step must not."""
    def sse(stop_reason, output_tokens):
        return (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
                + ("event: message_delta\ndata: " + json.dumps({"type": "message_delta",
                   "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": output_tokens}}) + "\n\n").encode()
                + b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    replies = [sse("tool_use", 40), sse("end_turn", 97)]
    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=replies.pop(0), headers={"content-type": "text/event-stream"}))))
    try:
        with TestClient(server.app) as client:
            interceptor.configure(load_settings(None), None)
            task = "Find where X is defined"
            first = [{"role": "user", "content": [{"type": "text", "text": task}]}]
            client.post("/v1/messages", json=subagent_body(task, messages=first, stream=True), headers=AUTH)
            loop = first + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {}}]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]
            client.post("/v1/messages", json=subagent_body(task, messages=loop, stream=True), headers=AUTH)
        assert [r["report_tokens"] for r in interceptor.read_records()] == [None, 97]
        (row,) = interceptor.summarize()["subagents"]
        assert row["avg_report_tokens"] == 97
    finally:
        interceptor.set_upstream_client(None)
        interceptor.configure(load_settings(None), None)
