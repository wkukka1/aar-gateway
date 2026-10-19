"""A Claude Code request sent to a non-Anthropic model must keep its tools:
definitions, tool calls, tool results, images, streaming -- or the model is
not a candidate for that request at all."""

from __future__ import annotations

import json

from aar_gateway.translate import (
    OpenAIStreamTranslator,
    anthropic_to_openai,
    openai_to_anthropic,
    untranslatable_reason,
)

TOOL = {"name": "Read", "description": "read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
AGENT_TURN = {
    "model": "local", "max_tokens": 512, "stream": False,
    "system": [{"type": "text", "text": "you are an agent", "cache_control": {"type": "ephemeral"}}],
    "tools": [TOOL],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "open a.py"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "x"},
            {"type": "text", "text": "Reading it."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "print(1)"}]},
            {"type": "text", "text": "what does it do?"}]},
    ],
}


def _sse_events(chunks: list[bytes]) -> list[tuple[str, dict]]:
    out = []
    for raw in chunks:
        event, data = raw.decode().strip().split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


class TestRequest:
    def test_agent_turn_keeps_tools_calls_and_results(self):
        req = anthropic_to_openai(AGENT_TURN, "qwen")
        assert req["tools"] == [{"type": "function", "function": {
            "name": "Read", "description": "read a file", "parameters": TOOL["input_schema"]}}]
        system, user, assistant, tool, user2 = req["messages"]
        assert system == {"role": "system", "content": "you are an agent"}
        assert assistant["content"] == "Reading it."
        assert assistant["tool_calls"] == [{"id": "toolu_1", "type": "function", "function": {
            "name": "Read", "arguments": json.dumps({"path": "a.py"})}}]
        assert tool == {"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"}
        assert user2 == {"role": "user", "content": "what does it do?"}
        assert req["max_tokens"] == 512

    def test_failed_tool_result_is_marked(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "is_error": True, "content": "no such file"}]}]}
        assert anthropic_to_openai(body, "m")["messages"][0]["content"] == "Error: no such file"

    def test_images_become_data_urls(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "what is this"}]}]}
        (msg,) = anthropic_to_openai(body, "m")["messages"]
        assert msg["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    def test_tool_choice_and_streaming_options(self):
        body = {**AGENT_TURN, "stream": True, "tool_choice": {"type": "tool", "name": "Read"}}
        req = anthropic_to_openai(body, "m", max_tokens_param="max_completion_tokens")
        assert req["tool_choice"] == {"type": "function", "function": {"name": "Read"}}
        assert req["stream_options"] == {"include_usage": True}
        assert req["max_completion_tokens"] == 512


class TestLosslessOrNotAtAll:
    def test_a_normal_agent_turn_translates(self):
        assert untranslatable_reason(AGENT_TURN, max_tools=128) is None

    def test_too_many_tools(self):
        assert "tools" in untranslatable_reason({**AGENT_TURN, "tools": [TOOL] * 3}, max_tools=2)

    def test_server_tools_cannot_cross(self):
        body = {**AGENT_TURN, "tools": [{"type": "web_search_20260209", "name": "web_search"}]}
        assert "web_search" in untranslatable_reason(body)

    def test_documents_and_images_in_tool_results_cannot_cross(self):
        doc = {"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]}
        img = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                "content": [{"type": "image", "source": {"type": "base64"}}]}]}]}
        assert "document" in untranslatable_reason(doc)
        assert "tool result" in untranslatable_reason(img)


class TestResponse:
    def test_tool_call_becomes_tool_use(self):
        resp = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "call_9", "function": {"name": "Read", "arguments": '{"path": "b.py"}'}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                          "prompt_tokens_details": {"cached_tokens": 40}}}
        msg = openai_to_anthropic(resp, "qwen")
        assert msg["content"] == [{"type": "tool_use", "id": "call_9", "name": "Read", "input": {"path": "b.py"}}]
        assert msg["stop_reason"] == "tool_use"
        assert msg["usage"] == {"input_tokens": 60, "output_tokens": 7,
                                "cache_read_input_tokens": 40, "cache_creation_input_tokens": 0}

    def test_tool_calls_with_a_stop_finish_reason_still_stop_for_tool_use(self):
        resp = {"choices": [{"finish_reason": "stop", "message": {"content": "", "tool_calls": [
            {"id": "c", "function": {"name": "Read", "arguments": "{}"}}]}}]}
        assert openai_to_anthropic(resp, "m")["stop_reason"] == "tool_use"

    def test_stream_text_then_tool_call(self):
        tr = OpenAIStreamTranslator("qwen")
        chunks = [
            {"choices": [{"delta": {"content": "Let me "}}]},
            {"choices": [{"delta": {"content": "look."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                                    "function": {"name": "Read", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "c.py"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
        ]
        events = _sse_events([e for c in chunks for e in tr.feed(c)] + tr.finish())
        kinds = [k for k, _ in events]
        assert kinds == ["message_start", "content_block_start", "content_block_delta", "content_block_delta",
                         "content_block_stop", "content_block_start", "content_block_delta",
                         "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        tool_start = events[5][1]
        assert tool_start["index"] == 1
        assert tool_start["content_block"] == {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}}
        partial = "".join(d["delta"]["partial_json"] for k, d in events if d.get("delta", {}).get("type") == "input_json_delta")
        assert json.loads(partial) == {"path": "c.py"}
        assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
        assert events[-2][1]["usage"]["output_tokens"] == 9
