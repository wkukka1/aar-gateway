"""The interceptor end to end with routing on: the non-Anthropic lane keeps
tools, a turn's tool loop stays on one model, a failed reroute falls back,
costs and baselines are logged, and the user's cost preference reaches the
router."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import history, interceptor, rollup, server
from aar_gateway.config import load_settings
from aar_gateway.pricing import CLAUDE_PRICES, cost_usd

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}
TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
OPENAI_TOOL_CALL = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
    {"id": "call_1", "function": {"name": "Read", "arguments": '{"path": "a.py"}'}}]}}],
    "usage": {"prompt_tokens": 30, "completion_tokens": 5}}
OPENAI_STREAM = (
    'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


def turn(text_or_blocks, model="claude-opus-5", **extra):
    content = text_or_blocks if isinstance(text_or_blocks, list) else [{"type": "text", "text": text_or_blocks}]
    return {"model": model, "max_tokens": 64, "tools": [TOOL],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]},
                         {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                         {"role": "user", "content": content}], **extra}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  models:
    claude-opus-5:    {lane: anthropic, proxy: x}
    claude-haiku-4-5: {lane: anthropic, proxy: x}
    local-coder:
      lane: openai
      base_url: http://local.test/v1
      prices: {input: 0, output: 0}
""")
    calls = {"anthropic": [], "openai": []}
    state = {"anthropic_status": {}}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        if req.url.host == "local.test":
            calls["openai"].append(body)
            if state.get("openai_status", 200) != 200:
                return httpx.Response(state["openai_status"], json={"error": {"message": state.get("openai_error", "nope")}})
            if body.get("stream"):
                return httpx.Response(200, text=OPENAI_STREAM, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=OPENAI_TOOL_CALL)
        calls["anthropic"].append(body)
        status = state["anthropic_status"].get(body.get("model"), 200)
        if status != 200:
            return httpx.Response(status, json={"type": "error",
                                                "error": {"message": state.get("error_message", "nope")}})
        return httpx.Response(200, json={"type": "message", "model": body["model"], "content": [],
                                         "id": f"msg_{len(calls['anthropic'])}",
                                         "usage": state.get("usage", {"input_tokens": 10, "output_tokens": 2})})

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(server.app) as client:
        settings = load_settings(cfg)
        yield client, settings, calls, state
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def records():
    return interceptor.read_records()


def test_openai_lane_keeps_tools_and_needs_no_anthropic_login(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, None)
    resp = client.post("/v1/messages", json=turn("open a.py", model="local-coder"))
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "tool_use", "id": "call_1", "name": "Read",
                                       "input": {"path": "a.py"}}]
    (sent,) = calls["openai"]
    assert sent["tools"][0]["function"]["name"] == "Read"
    assert calls["anthropic"] == []
    (rec,) = records()
    assert (rec["lane"], rec["cost_usd"], rec["input_tokens"]) == ("openai", 0.0, 30)


def test_openai_lane_streams_back_as_anthropic_events(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, None)
    resp = client.post("/v1/messages", json=turn("hi", model="local-coder", stream=True))
    text = resp.text
    assert "event: message_start" in text and "event: message_stop" in text
    deltas = [json.loads(line[5:]) for line in text.splitlines()
              if line.startswith("data:") and "text_delta" in line]
    assert "".join(d["delta"]["text"] for d in deltas) == "hello"
    assert records()[0]["output_tokens"] == 2


def test_a_turns_tool_loop_stays_on_the_turns_model(env):
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("fix the bug"), headers=AUTH)
    client.post("/v1/messages", json=turn([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]),
                headers=AUTH)
    assert len(asked) == 1 and asked[0]["human_text"] == "fix the bug"
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5", "claude-haiku-4-5"]
    assert records()[1]["reason"] == "same turn as the previous request"


def test_system_reminders_are_not_human_text(env):
    client, settings, _, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx["human_text"]))
    client.post("/v1/messages", json=turn([{"type": "text", "text": "<system-reminder>x</system-reminder>"},
                                           {"type": "text", "text": "real question"}]), headers=AUTH)
    assert asked == ["real question"]


def test_a_failed_reroute_falls_back_to_the_model_the_client_asked_for(env):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    resp = client.post("/v1/messages", json=turn("hi"), headers=AUTH)
    assert resp.status_code == 200
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5", "claude-opus-5"]
    (rec,) = records()
    assert rec["fallback"] is True and rec["routed_model"] == "claude-opus-5"


def test_a_refused_request_logs_the_providers_reason(env):
    """So a run of 400s can be diagnosed from the log, without the prompt."""
    client, settings, calls, state = env
    interceptor.configure(settings, None)
    state["anthropic_status"]["claude-opus-5"] = 400
    state["error_message"] = "messages.3.content.0: invalid   signature in thinking block"
    assert client.post("/v1/messages", json=turn("hi"), headers=AUTH).status_code == 400
    (rec,) = records()
    assert rec["status"] == 400 and rec["error"] == "error: messages.3.content.0: invalid signature in thinking block"


def test_a_fallback_keeps_why_the_reroute_was_refused(env):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    state["error_message"] = "thinking: not supported"
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    assert client.post("/v1/messages", json=turn("hi"), headers=AUTH).status_code == 200
    (rec,) = records()
    assert rec["status"] == 200 and rec["error"] == "claude-haiku-4-5: error: thinking: not supported"


def test_an_error_event_mid_stream_is_logged():
    tap = interceptor._SSEUsageTap()
    tap.feed(b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", '
             b'"message": "Overloaded"}}\n\n')
    assert tap.error == "overloaded_error: Overloaded"


def test_baselines_assume_the_other_model_kept_its_cache(env):
    """Turn 1 on Opus caches a 1,000-token prefix. Turn 2 is moved to Haiku,
    which has to rewrite that prefix -- but the "if you had stayed on Opus"
    baseline must read it from cache, or the switch looks free."""
    client, settings, _, state = env
    interceptor.configure(settings, lambda body, ctx: None)
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1000}
    client.post("/v1/messages", json=turn("first"), headers=AUTH)
    interceptor.set_model_picker(lambda body, ctx: "claude-haiku-4-5")
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1200}
    client.post("/v1/messages", json=turn("second"), headers=AUTH)
    rec = records()[1]
    warm = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 1000, "cache_creation": {}}
    assert rec["cost_if_requested_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-opus-5"], warm))
    assert rec["cost_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], state["usage"]))
    totals = client.get("/stats").json()["totals"]
    assert totals["rerouted"] == 1 and totals["requests"] == 2


def test_cost_preference_is_saved_and_reaches_the_router(env, tmp_path):
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["quality_margin"]))
    save_more = interceptor.PRESETS["save_more"]
    assert client.post("/api/preferences", json={"preset": "save_more"}).json()["quality_margin"] == save_more
    assert json.loads((tmp_path / "preferences.json").read_text())["quality_margin"] == save_more
    client.post("/v1/messages", json=turn("a"), headers=AUTH)
    client.post("/v1/messages", json=turn("b"), headers={**AUTH, interceptor.MARGIN_HEADER: "0.02"})
    assert seen == [save_more, 0.02]
    assert client.post("/api/preferences", json={"quality_margin": 2}).status_code == 400


def test_dashboard_page_is_served(env):
    client, *_ = env
    resp = client.get("/dashboard")
    assert resp.status_code == 200 and "Cost preference" in resp.text


def test_human_text_looks_past_claude_codes_trailing_system_message():
    """Found live: Claude Code appends hook output as a `system` message after
    the user's, and puts reminders inside the same text block as the prompt."""
    body = {"messages": [
        {"role": "user", "content": [{"type": "text", "text":
            "<system-reminder>\ncontext\n</system-reminder>\nWhat is the capital of France?"}]},
        {"role": "system", "content": [{"type": "text", "text": "SessionStart hook success"}]}]}
    assert interceptor.human_text(body) == "What is the capital of France?"


def test_side_requests_without_tools_are_not_routed(env):
    """Found live: Claude Code's title request switches thinking off, which
    Opus/Sonnet 5.5 reject -- routing it only ever bought a 400 and a retry."""
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(1) or "claude-haiku-4-5")
    body = {**turn("name this session"), "tools": []}
    client.post("/v1/messages", json=body, headers=AUTH)
    assert asked == [] and calls["anthropic"][0]["model"] == "claude-opus-5"
    assert records()[0]["reason"] == "side request (no tools), not routed"


def test_models_that_reject_the_requests_settings_are_skipped():
    from aar_gateway.routing import incompatibility

    agent = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    assert incompatibility(agent, {"rejects": ["effort"]}) == "rejects output_config.effort"
    assert incompatibility(agent, {"rejects": ["thinking_disabled"]}) is None
    assert incompatibility({"thinking": {"type": "disabled"}}, {"rejects": ["thinking_disabled"]})


def test_a_model_the_upstream_rejects_for_this_client_is_not_tried_again(env):
    """Found live: Anthropic answers 400 to a model the client's Claude Code
    version doesn't support. One fallback is fine; retrying it every turn
    is not."""
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers={**AUTH, "user-agent": "claude-cli/1.0"})
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]},
                headers={**AUTH, "user-agent": "claude-cli/1.0"})
    assert seen[0] == {} and "claude-haiku-4-5" in seen[1]


def test_subagent_hand_backs_and_task_notifications_are_not_new_turns(env):
    """Found live: with background subagents, Claude Code tells the main agent
    each one finished with two user messages of its own. Routing each as a new
    turn moved the main conversation between models mid-task."""
    client, settings, calls, _ = env
    asked = []
    interceptor.configure(settings, lambda body, ctx: asked.append(ctx["human_text"]) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("research the gateway with subagents"), headers=AUTH)
    hand_back = ('Another Claude session sent a message:\n<agent-message from="a41b9e">\n[Subagent hand-back] '
                 'The text below is the final report of a subagent.\nsrc/x.py:12\n</agent-message>')
    notification = ("<task-notification>\n<task-id>a41b9e</task-id>\n<status>completed</status>\n"
                    "</task-notification>")
    for text in (hand_back, notification):
        client.post("/v1/messages", json=turn(text), headers=AUTH)
    assert asked == ["research the gateway with subagents"]
    assert [c["model"] for c in calls["anthropic"]] == ["claude-haiku-4-5"] * 3
    assert [r["reason"] for r in records()[1:]] == ["same turn as the previous request"] * 2
    client.post("/v1/messages", json=turn([{"type": "text", "text": notification},
                                           {"type": "text", "text": "now summarise it"}]), headers=AUTH)
    assert asked[-1] == "now summarise it"  # the human's own words still start a turn


def test_a_400_about_the_request_itself_does_not_block_the_model(env):
    """Anthropic answers 400 to a prompt too long for a model: that says
    nothing about the next request, which may be short."""
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    state["error_message"] = "prompt is too long: 210000 tokens > 200000 maximum"
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert seen == [{}, {}]


def test_an_openai_lane_400_blocks_that_model_like_an_anthropic_one_does(env):
    client, settings, calls, state = env
    state["openai_status"] = 400
    state["openai_error"] = "this model does not support the request"
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "local-coder")
    first = client.post("/v1/messages", json=turn("one"), headers=AUTH)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert first.status_code == 200  # fell back to the client's own model
    assert seen[0] == {} and list(seen[1]) == ["local-coder"]


def test_an_openai_lane_400_about_the_request_itself_does_not_block_the_model(env):
    client, settings, calls, state = env
    state["openai_status"] = 400
    state["openai_error"] = "prompt is too long: 210000 tokens > 200000 maximum"
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "local-coder")
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert seen == [{}, {}]


def test_a_blocked_model_is_tried_again_after_a_while(env, monkeypatch):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(dict(ctx["blocked"])) or "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    monkeypatch.setattr(interceptor, "BLOCK_SECONDS", 0.0)
    client.post("/v1/messages", json={**turn("two"), "messages": [{"role": "user", "content": "x"}]}, headers=AUTH)
    assert seen == [{}, {}]


def test_a_new_session_sharing_the_prefix_counts_the_cache_an_earlier_one_wrote(env):
    """Found live: session A cached Claude Code's system prompt + tools on
    Opus; session B (same prefix) was moved to another model and rewrote it.
    Opus would have *read* it -- the baseline must say so, and the router must
    see Opus as warm."""
    client, settings, _, state = env
    seen = []

    def body(first):
        return {**turn("x"), "messages": [{"role": "user", "content": first}]}

    interceptor.configure(settings, lambda b, ctx: seen.append(ctx["warm_tokens"]))
    state["usage"] = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 1000}
    client.post("/v1/messages", json=body("session A"), headers=AUTH)

    interceptor.set_model_picker(lambda b, ctx: seen.append(ctx["warm_tokens"]) or "claude-haiku-4-5")
    state["usage"] = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 1100}
    client.post("/v1/messages", json=body("session B"), headers=AUTH)

    assert seen == [{}, {"claude-opus-5": 1000}]
    rec = records()[1]
    opus_warm = {"input_tokens": 0, "output_tokens": 5, "cache_creation_input_tokens": 100,
                 "cache_read_input_tokens": 1000, "cache_creation": {}}
    assert rec["cost_if_requested_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-opus-5"], opus_warm))


def test_status_line_shows_the_model_that_actually_answered(env):
    client, settings, _, _ = env
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("hi"), headers={**AUTH, "x-claude-code-session-id": "s-1"})
    line = client.get("/api/statusline", params={"session_id": "s-1"}).json()["line"]
    assert "claude-haiku-4-5 (routed)" in line and "less plan usage" in line


def test_session_id_falls_back_to_metadata_user_id(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    body = {**turn("hi"), "metadata": {"user_id": "user_abc_account_x_session_s-2"}}
    client.post("/v1/messages", json=body, headers=AUTH)
    assert "claude-opus-5 (as asked)" in client.get("/api/statusline", params={"session_id": "s-2"}).json()["line"]


def test_an_unreachable_provider_is_a_logged_502_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))

    def down(req):
        raise httpx.ConnectError("no route to host")

    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(down)))
    with TestClient(server.app) as client:
        interceptor.configure(load_settings(None), None)
        resp = client.post("/v1/messages", json=turn("hi"), headers=AUTH)
    interceptor.set_upstream_client(None)
    assert resp.status_code == 502 and resp.json()["type"] == "error"
    (rec,) = records()
    assert rec["status"] == 502 and "unreachable" in rec["reason"]


def test_history_keeps_each_users_turn_with_its_outcome_and_folder(env, tmp_path):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    env_block = {"type": "text", "text": "<system-reminder>\n - Primary working directory: /repo/x\n</system-reminder>"}
    body = turn([env_block, {"type": "text", "text": "refactor the loader"}])
    client.post("/v1/messages", json=body, headers=AUTH)
    client.post("/v1/messages", json=turn([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]),
                headers=AUTH)  # the same turn's tool loop: not another row
    (rec,) = history.read()
    assert rec["prompt"] == "refactor the loader" and rec["cwd"] == "/repo/x"
    assert "system-reminder" not in rec["prompt"]
    assert rec["status"] == 200 and rec["routed_model"] and rec["prompt_hash"]


def test_history_off_keeps_nothing(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    history.set_enabled(False)
    client.post("/v1/messages", json=turn("fix it"), headers=AUTH)
    assert history.read() == [] and records()  # the request itself was still logged, without text


def test_the_log_says_what_started_a_turn_without_its_text():
    assert interceptor.turn_trigger(None) is None
    assert interceptor.turn_trigger("fix the login bug") == "typed"
    assert interceptor.turn_trigger("<command-name>/compact</command-name> secret") == "command-name"


def test_the_dashboard_is_told_which_models_are_not_claude(env):
    _, settings, _, _ = env
    settings["models"]["llama3.1:latest"] = {"lane": "openai", "base_url": "http://127.0.0.1:11434/v1"}
    interceptor.configure(settings, None)
    notes = interceptor.preferences()["model_notes"]
    assert notes == {"local-coder": "other provider", "llama3.1:latest": "local"}


def test_the_previous_turns_prediction_reaches_the_router(env):
    client, settings, _, _ = env
    seen = []

    def picker(body, ctx):
        seen.append(ctx["prev_quality"])
        return {"model": "claude-haiku-4-5", "reason": "r", "model_quality": {"claude-haiku-4-5": 0.6}}
    interceptor.configure(settings, picker)
    client.post("/v1/messages", json=turn("one"), headers=AUTH)
    client.post("/v1/messages", json=turn("two"), headers=AUTH)  # same conversation, next human turn
    assert seen == [None, {"claude-haiku-4-5": 0.6}]


def test_the_thread_mode_is_logged_never_its_ids():
    assert interceptor.thread_mode({"messages": []}) is None
    assert interceptor.thread_mode({"thread": {"type": "create"}}) == "create"
    assert interceptor.thread_mode({"thread": {"previous_message_id": "msg_secret"}}) == "continue"


def test_a_thread_continuation_stays_on_the_model_that_wrote_its_message(env):
    """The API keeps a threaded conversation server-side; a request that
    only adds to it must not move to another model mid-thread."""
    client, settings, calls, state = env
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("first"), headers=AUTH)  # routed to Haiku, writes msg_1
    interceptor.set_model_picker(lambda body, ctx: "claude-opus-5")
    cont = {**turn("next"), "thread": {"previous_message_id": "msg_1"}}
    client.post("/v1/messages", json=cont, headers=AUTH)
    assert calls["anthropic"][-1]["model"] == "claude-haiku-4-5"
    rec = records()[-1]
    assert rec["thread"] == "continue" and rec["reason"].startswith("continues a server-side thread")
    unknown = {**turn("other"), "thread": {"previous_message_id": "msg_from_before_the_gateway"}}
    client.post("/v1/messages", json=unknown, headers=AUTH)
    assert calls["anthropic"][-1]["model"] == "claude-opus-5"  # the model asked for


def test_a_thread_error_never_blocks_the_model(env):
    client, settings, calls, state = env
    state["anthropic_status"]["claude-haiku-4-5"] = 404
    state["error_message"] = "No thread state was found for the requested `previous_message_id`."
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("hi"), headers=AUTH)
    state["anthropic_status"]["claude-haiku-4-5"] = 400
    client.post("/v1/messages", json=turn("again"), headers=AUTH)
    assert not interceptor._blocked


@pytest.mark.parametrize("text", [
    "that's wrong, the cache key is the model name", "No, use the other file", "not quite what I asked",
    "try again", "you missed the second case", "still failing on the tests", "this doesn't work",
    "redo that with the retry logic"])
def test_corrections_are_recognised(text):
    assert interceptor.is_correction(text)


@pytest.mark.parametrize("text", [
    None, "what's wrong with this function?", "no need to change the tests", "add a wrong-type check",
    "<agent-message from='a1'>Another subagent wrote: that's wrong</agent-message>",
    "Another Claude session sent a message: not quite right", "You are grading answers to a developer's request",
    "that's wrong " * 40])
def test_ordinary_messages_and_automated_ones_are_not_corrections(text):
    assert not interceptor.is_correction(text)


def test_a_correction_of_a_cheaper_answer_holds_the_next_turns_on_the_asked_model(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("first"), headers=AUTH)  # Haiku answers
    client.post("/v1/messages", json=turn("that's wrong, redo it"), headers=AUTH)  # the correction
    assert calls["anthropic"][-1]["model"] == "claude-opus-5"
    rec = records()[-1]
    assert rec["correction"] is True and rec["routed_model"] == "claude-opus-5"
    for i in range(interceptor.CORRECTION_HOLD_TURNS - 1):
        client.post("/v1/messages", json=turn(f"next task {i}"), headers=AUTH)
        assert calls["anthropic"][-1]["model"] == "claude-opus-5", i
    client.post("/v1/messages", json=turn("another task"), headers=AUTH)
    assert calls["anthropic"][-1]["model"] == "claude-haiku-4-5"  # the hold is over


def test_a_correction_after_an_answer_on_the_asked_model_holds_nothing(env):
    client, settings, calls, _ = env
    interceptor.configure(settings, lambda body, ctx: None)
    client.post("/v1/messages", json=turn("first"), headers=AUTH)
    client.post("/v1/messages", json=turn("that's wrong"), headers=AUTH)
    assert records()[-1]["correction"] is False
    assert records()[-1]["reason"] != "after a correction: staying on the model you asked for (7 more turns)"


def test_an_expired_thread_is_not_counted_as_an_error():
    expired = {"status": 404, "error": "not_found_error: No thread state was found for the requested `previous_message_id`."}
    real = {"status": 404, "error": "not_found_error: model not found"}
    assert rollup.expected_thread_expiry(expired)
    assert not rollup.expected_thread_expiry(real)
    stats = interceptor.summarize([{**expired, "requested_model": "claude-opus-5-5", "routed_model": "claude-opus-5-5"},
                                   {"status": 200, "requested_model": "claude-opus-5-5", "routed_model": "claude-opus-5-5"}])
    assert stats["totals"]["errors"] == 0


MISMATCH = ("thread: the request does not match the stored Thread for the requested `previous_message_id`; "
            "differing field(s): model. These fields must stay the same across a Thread's turns")


def test_a_thread_continuation_with_an_unknown_model_finds_it_by_retrying(env):
    """The thread started on Haiku while the gateway wasn't watching: the
    continuation goes to the model asked for, is rejected, then lands on Haiku."""
    client, settings, calls, state = env
    interceptor.configure(settings, None)
    state["anthropic_status"]["claude-opus-5"] = 400
    state["error_message"] = MISMATCH
    cont = {**turn("next"), "thread": {"previous_message_id": "msg_from_before"}}
    resp = client.post("/v1/messages", json=cont, headers=AUTH)
    assert resp.status_code == 200
    assert [c["model"] for c in calls["anthropic"]] == ["claude-opus-5", "claude-haiku-4-5"]
    assert records()[-1]["routed_model"] == "claude-haiku-4-5"
    assert interceptor._message_model("msg_from_before") == "claude-haiku-4-5"  # remembered


def test_another_400_on_a_thread_is_returned_unchanged(env):
    client, settings, calls, state = env
    interceptor.configure(settings, None)
    state["anthropic_status"]["claude-opus-5"] = 400
    state["error_message"] = "max_tokens: too large"
    cont = {**turn("next"), "thread": {"previous_message_id": "msg_x"}}
    resp = client.post("/v1/messages", json=cont, headers=AUTH)
    assert resp.status_code == 400 and "max_tokens" in resp.text
    assert len(calls["anthropic"]) == 1  # no retries for a different error


def test_thread_models_survive_a_restart(env):
    interceptor._remember_message("msg_kept", "claude-haiku-4-5")
    interceptor._message_models.clear()  # what a restart loses
    interceptor._message_models_loaded = False
    assert interceptor._message_model("msg_kept") == "claude-haiku-4-5"



def test_a_request_on_the_asked_model_is_its_own_baseline(env):
    """Seen live as +1251%: a new session on Opus, not rerouted, had to write
    its cache; the baseline borrowed another conversation's cached prefix and
    priced the same request as if read. Nothing was rerouted, so nothing changed."""
    client, settings, _, state = env
    interceptor.configure(settings, lambda body, ctx: None)
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 20000}
    client.post("/v1/messages", json=turn("first project"), headers=AUTH)
    other = {**turn("another project"), "metadata": {"user_id": "user_x_account_y_session_other"}}
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 23000,
                      "cache_read_input_tokens": 0}
    client.post("/v1/messages", json=other, headers=AUTH)
    rec = records()[-1]
    assert rec["routed_model"] == rec["requested_model"]
    assert rec["cost_if_requested_usd"] == pytest.approx(rec["cost_usd"])


def test_switching_back_still_counts_the_cache_the_reroute_cost(env):
    """Turn 2 went to Haiku; turn 3 returns to Opus and rewrites its cache.
    Without the router Opus would have kept it, so that rewrite is the router's."""
    client, settings, _, state = env
    interceptor.configure(settings, lambda body, ctx: None)
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1000}
    client.post("/v1/messages", json=turn("first"), headers=AUTH)
    interceptor.set_model_picker(lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=turn("second"), headers=AUTH)
    interceptor.set_model_picker(lambda body, ctx: None)
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 1300}
    client.post("/v1/messages", json=turn("third"), headers=AUTH)
    rec = records()[-1]
    assert rec["routed_model"] == "claude-opus-5"
    assert rec["cost_if_requested_usd"] < rec["cost_usd"]  # the warm baseline: the reroute cost this


def test_the_route_endpoint_answers_for_the_mod_and_the_proxy_keeps_its_choice(env):
    """The mod asks which model, switches the turn itself; the proxy then keeps
    that model (no second routing) and still counts against the model asked for."""
    client, settings, calls, _ = env
    interceptor.configure(settings, lambda body, ctx: {"model": "claude-haiku-4-5", "reason": "cheapest"})
    sid = "s-mod-0123456789"
    picked = client.post("/api/route", json={"text": "rename this variable", "model": "claude-opus-5",
                                            "session_id": sid}).json()
    assert picked["model"] == "claude-haiku-4-5" and picked["requested"] == "claude-opus-5"
    interceptor.set_model_picker(lambda body, ctx: pytest.fail("the proxy routed a turn the mod had routed"))
    body = {**turn("rename this variable", model="claude-haiku-4-5"),
            "metadata": {"user_id": f"user_x_account_y_session_{sid}"}}
    client.post("/v1/messages", json=body, headers=AUTH)
    assert calls["anthropic"][-1]["model"] == "claude-haiku-4-5"
    rec = records()[-1]
    assert rec["reason"] == "routed in Claude Code: cheapest"
    assert rec["requested_model"] == "claude-opus-5" and rec["routed_model"] == "claude-haiku-4-5"


def test_the_route_endpoint_keeps_the_model_when_routing_is_off_or_fails(env):
    client, settings, _, _ = env
    interceptor.configure(settings, lambda body, ctx: 1 / 0)
    assert client.post("/api/route", json={"text": "hi", "model": "claude-opus-5"}).json()["model"] is None
    assert client.post("/api/route", json=["not", "an", "object"]).status_code == 400


def test_session_requests_lists_one_sessions_requests_without_text(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    sid = "s-pane-0123456789"
    client.post("/v1/messages", json={**turn("secret words"), "metadata": {"user_id": f"user_x_account_y_session_{sid}"}},
                headers=AUTH)
    client.post("/v1/messages", json=turn("someone else's"), headers=AUTH)
    data = client.get("/api/session-requests", params={"session_id": sid}).json()
    assert data["requests"] == 1 and data["recent"][0]["requested_model"] == "claude-opus-5"
    assert "secret" not in json.dumps(data)


def test_every_request_of_a_turn_the_mod_routed_counts_against_the_model_asked_for(env):
    """Seen live: the tool loop's later requests were counted against the mod's
    model (no saving shown) instead of the model asked for."""
    client, settings, calls, state = env
    interceptor.configure(settings, lambda body, ctx: {"model": "claude-haiku-4-5", "reason": "cheapest"})
    sid = "s-loop-0123456789"
    client.post("/api/route", json={"text": "read calc.py", "model": "claude-opus-5", "session_id": sid})
    meta = {"metadata": {"user_id": f"user_x_account_y_session_{sid}"}}
    client.post("/v1/messages", json={**turn("read calc.py", model="claude-haiku-4-5"), **meta}, headers=AUTH)
    loop = turn([{"type": "tool_result", "tool_use_id": "t1", "content": "def add(a, b): ..."}], model="claude-haiku-4-5")
    client.post("/v1/messages", json={**loop, **meta}, headers=AUTH)
    first, second = records()[-2:]
    assert first["reason"] == "routed in Claude Code: cheapest"
    assert second["requested_model"] == "claude-opus-5" and second["routed_model"] == "claude-haiku-4-5"
    assert second["cost_if_requested_usd"] > second["cost_usd"]
    # the pane's view: one turn, both requests in it, on Haiku, cheaper than Opus
    turns = client.get("/api/session-requests", params={"session_id": sid}).json()["turns"]
    assert len(turns) == 1 and turns[0]["requests"] == 2 and turns[0]["models"] == ["claude-haiku-4-5"]
    assert turns[0]["reason"] == "routed in Claude Code: cheapest" and turns[0]["saved_pct"] > 0
    assert len(turns[0]["time"]) == 5  # HH:MM, local time


def test_a_prefix_the_routed_model_held_but_did_not_read_is_not_credited_to_the_baseline(env):
    """Seen live: a prefix 'shared' with an earlier request on the routed model
    still had to be written in full there, yet the baseline priced it as read on
    the model asked for (0.18 vs 0.012, shown as +1400%)."""
    client, settings, calls, state = env
    interceptor.configure(settings, lambda body, ctx: "claude-haiku-4-5")
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 20000}
    client.post("/v1/messages", json=turn("first session"), headers=AUTH)
    other = {**turn("second session"), "metadata": {"user_id": "user_x_account_y_session_another"}}
    state["usage"] = {"input_tokens": 0, "output_tokens": 10, "cache_creation_input_tokens": 20000,
                      "cache_read_input_tokens": 0}
    client.post("/v1/messages", json=other, headers=AUTH)
    rec = records()[-1]
    assert rec["routed_model"] == "claude-haiku-4-5"
    assert rec["cost_if_requested_usd"] > rec["cost_usd"]  # Opus writing 20k costs more than Haiku writing it


def test_turn_smoothing_is_on_by_default_and_turning_it_off_warns(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    prefs = client.get("/api/preferences").json()
    assert prefs["turn_smoothing"] is True and prefs["turn_smoothing_warning"] is None
    off = client.post("/api/preferences", json={"turn_smoothing": False}).json()
    assert off["turn_smoothing"] is False and "switch model back and forth" in off["turn_smoothing_warning"]
    seen = []
    interceptor.set_model_picker(lambda body, ctx: seen.append(ctx["turn_smoothing"]))
    client.post("/v1/messages", json=turn("next"), headers=AUTH)
    assert seen == [False]  # reaches the router
    assert client.post("/api/preferences", json={"turn_smoothing": True}).json()["turn_smoothing_warning"] is None


def test_a_request_the_mod_reports_is_priced_and_shows_in_its_session(env):
    """A plugin-only session: Claude Code talks to Anthropic directly, so the
    mod reports each request's usage and the gateway prices it as its own."""
    client, settings, calls, _ = env
    interceptor.configure(settings, None)
    sid = "s-direct-0123456789"
    usage = {"input_tokens": 5, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 20000}
    same = client.post("/api/mod-record", json={"session_id": sid, "requested_model": "claude-opus-5",
                                                "routed_model": "claude-opus-5", "new_turn": True, **usage}).json()
    assert same["cost_usd"] == pytest.approx(same["cost_if_requested_usd"])  # nothing rerouted: its own baseline
    moved = client.post("/api/mod-record", json={"session_id": sid, "requested_model": "claude-opus-5",
                                                 "routed_model": "claude-haiku-4-5", "new_turn": True, **usage}).json()
    # Opus held this loop's 20k-token prompt, so without the switch it would have read it
    warm = {"input_tokens": 5, "output_tokens": 100, "cache_read_input_tokens": 20000, "cache_creation_input_tokens": 0}
    assert moved["cost_if_requested_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-opus-5"], warm))
    assert moved["cost_usd"] == pytest.approx(cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], usage))
    assert calls["anthropic"] == []  # nothing was sent anywhere
    data = client.get("/api/session-requests", params={"session_id": sid}).json()
    assert data["requests"] == 2


def test_mod_record_refuses_what_isnt_an_object(env):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    assert client.post("/api/mod-record", json=[1, 2]).status_code == 400


def test_the_mod_asks_the_router_knowing_which_model_holds_the_conversation(env):
    """Seen live: the mod's route question carried only the prompt text, so the
    router moved a 250k-token Opus conversation to Haiku (a full cache rewrite,
    +267%) and back (+900%). It must see the session's model and prompt size."""
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx) or {"model": None, "reason": "kept"})
    sid = "s-warm-0123456789"
    usage = {"input_tokens": 2, "output_tokens": 50, "cache_read_input_tokens": 240000, "cache_creation_input_tokens": 9000}
    client.post("/api/mod-record", json={"session_id": sid, "requested_model": "claude-opus-5",
                                         "routed_model": "claude-opus-5", "new_turn": True, **usage})
    client.post("/api/route", json={"text": "rename this variable", "model": "claude-opus-5", "session_id": sid})
    assert seen[-1]["last_model"] == "claude-opus-5" and seen[-1]["last_prompt_tokens"] == 249002
    assert seen[-1]["est_prompt_tokens"] == 249002
    # through the proxy the same: its main-loop requests are remembered for the mod
    meta = {"metadata": {"user_id": "user_x_account_y_session_s-proxy-0123456789"}}
    client.post("/v1/messages", json={**turn("first"), **meta}, headers=AUTH)
    client.post("/api/route", json={"text": "next", "model": "claude-opus-5", "session_id": "s-proxy-0123456789"})
    assert seen[-1]["last_model"] is not None and seen[-1]["last_prompt_tokens"]


def _mod_session(client, sid, routed="claude-haiku-4-5"):
    client.post("/api/mod-record", json={"session_id": sid, "requested_model": "claude-opus-5", "routed_model": routed,
                                         "new_turn": True, "input_tokens": 5, "output_tokens": 50,
                                         "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 100})


def test_the_mod_route_keeps_the_proxys_rules(env):
    """The rules the proxy applies to a turn apply when the mod asks too:
    smoothing gets the last turn's quality, a correction holds the model asked
    for, a model that failed in the session isn't picked, a pin must fit."""
    client, settings, _, _ = env
    seen = []

    def picker(body, ctx):
        seen.append(ctx)
        return {"model": "claude-haiku-4-5", "reason": "cheapest", "model_quality": {"claude-haiku-4-5": 0.6}}
    interceptor.configure(settings, picker)
    sid = "s-rules-0123456789"
    _mod_session(client, sid)
    route = lambda text, **kw: client.post("/api/route", json={"text": text, "model": "claude-opus-5",  # noqa: E731
                                                              "session_id": sid, **kw}).json()
    route("rename this variable")
    assert route("now the next one")["model"] == "claude-haiku-4-5"
    assert seen[-1]["prev_quality"] == {"claude-haiku-4-5": 0.6}  # smoothing sees the last turn
    assert seen[-1]["usage_profile"]  # this session's own token profile
    # a correction after a rerouted answer: the model asked for, for the next turns too
    calls = len(seen)
    held = route("that's wrong, try again")
    assert held["model"] == "claude-opus-5" and "after a correction" in held["reason"]
    assert route("ok continue")["model"] == "claude-opus-5" and len(seen) == calls  # the router wasn't asked
    # a model that failed in this session reaches the router as blocked
    for _ in range(interceptor.CORRECTION_HOLD_TURNS):  # the hold runs out
        route("next")
    route("another", blocked=["claude-haiku-4-5"])
    assert "after a correction" not in seen[-1].get("reason", "")
    assert seen[-1]["blocked"] == {"claude-haiku-4-5": "failed earlier in this session"}
    # a pinned model that can't take the conversation isn't used
    client.post("/api/preferences", json={"mode": "pin", "pinned_model": "claude-haiku-4-5"})
    settings["models"]["claude-haiku-4-5"]["context_window"] = 1000
    pinned = route("anything")
    assert pinned["model"] is None and "can't take this request" in pinned["reason"]


def test_a_resumed_session_is_routed_against_the_model_first_asked_for(env):
    """Seen live: after `claude -c` the session came back on the model the mod
    had switched to (Sonnet), so Sonnet became 'asked': savings counted against
    it and the router could never go back to Opus (pricier than asked)."""
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["requested"]) or "claude-haiku-4-5")
    sid = "s-resume-0123456789"
    client.post("/api/route", json={"text": "rename x", "model": "claude-opus-5", "session_id": sid})
    _mod_session(client, sid, routed="claude-haiku-4-5")
    again = client.post("/api/route", json={"text": "next", "model": "claude-haiku-4-5", "session_id": sid,
                                            "resumed": True}).json()
    assert again["requested"] == "claude-opus-5" and seen[-1] == "claude-opus-5"


def test_a_model_choice_of_the_model_already_routed_to_is_not_read_as_a_resume(env):
    """A /model choice of the model the session is already routed to isn't a
    resume: the person asked for that model, and it stays."""
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["requested"]) or "claude-haiku-4-5")
    sid = "s-model-0123456789"
    client.post("/api/route", json={"text": "rename x", "model": "claude-opus-5", "session_id": sid})
    _mod_session(client, sid, routed="claude-haiku-4-5")
    chosen = client.post("/api/route", json={"text": "next", "model": "claude-haiku-4-5", "session_id": sid}).json()
    assert chosen["requested"] == "claude-haiku-4-5" and seen[-1] == "claude-haiku-4-5"


def test_lambda_is_saved_overridden_per_request_and_switched_off(env, tmp_path):
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["cost_weight"]))
    assert client.get("/api/preferences").json()["cost_weight"] is None
    assert client.post("/api/preferences", json={"cost_weight": 0.1}).json()["cost_weight"] == 0.1
    assert json.loads((tmp_path / "preferences.json").read_text())["cost_weight"] == 0.1
    client.post("/v1/messages", json=turn("a"), headers=AUTH)
    client.post("/v1/messages", json=turn("b"), headers={**AUTH, interceptor.COST_WEIGHT_HEADER: "0.3"})
    client.post("/v1/messages", json=turn("c"), headers={**AUTH, interceptor.COST_WEIGHT_HEADER: "off"})
    assert seen == [0.1, 0.3, None]
    balanced = interceptor.COST_PRESETS["balanced"]
    assert client.post("/api/preferences", json={"cost_preset": "balanced"}).json()["cost_weight"] == balanced
    assert client.post("/api/preferences", json={"cost_weight": None}).json()["cost_weight"] is None
    assert json.loads((tmp_path / "preferences.json").read_text())["cost_weight"] is None
    assert client.post("/api/preferences", json={"cost_weight": 2}).status_code == 400
    assert client.post("/api/preferences", json={"cost_preset": "nope"}).status_code == 400


def test_a_saved_lambda_survives_a_restart(env, tmp_path):
    client, settings, _, _ = env
    interceptor.configure(settings, None)
    client.post("/api/preferences", json={"cost_weight": 0.07})
    settings["router"]["cost_weight"] = None
    interceptor.configure(settings, None)
    assert interceptor.preferences()["cost_weight"] == 0.07


def test_a_nan_lambda_header_falls_back_to_the_saved_value(env):
    client, settings, _, _ = env
    seen = []
    interceptor.configure(settings, lambda body, ctx: seen.append(ctx["cost_weight"]))
    client.post("/api/preferences", json={"cost_weight": 0.1})
    client.post("/v1/messages", json=turn("a"), headers={**AUTH, interceptor.COST_WEIGHT_HEADER: "nan"})
    client.post("/v1/messages", json=turn("b"), headers={**AUTH, interceptor.COST_WEIGHT_HEADER: "inf"})
    assert seen == [0.1, 0.1]
