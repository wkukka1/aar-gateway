"""An upstream that dies mid-stream is an error to the client and in the log, never a clean finish."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import interceptor, server
from aar_gateway.config import load_settings

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}
BODY = {"model": "claude-opus-5", "max_tokens": 8, "stream": True,
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hello"}]}
OPENAI_START = 'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'


class _Stream(httpx.AsyncByteStream):
    """Yields the chunks, then (optionally) the connection drops."""

    def __init__(self, chunks, then_fail=False):
        self.chunks, self.then_fail = chunks, then_fail

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk if isinstance(chunk, bytes) else chunk.encode()
        if self.then_fail:
            raise httpx.ReadError("connection reset")


@pytest.fixture
def run(tmp_path):
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  models:
    claude-opus-5: {lane: anthropic, proxy: x}
    local-coder: {lane: openai, base_url: "http://local.test/v1", proxy: x, prices: {input: 0, output: 0}}
""")

    def go(stream, *, target):
        def handler(req):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with TestClient(server.app) as c:
            interceptor.configure(load_settings(cfg), lambda body, ctx: target)
            text = c.post("/v1/messages", json=BODY, headers=AUTH).text
        records = [json.loads(line) for line in interceptor.log_path().read_text().splitlines()]
        return text, records

    yield go
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


def test_an_openai_stream_that_drops_is_an_error_event_and_a_502_in_the_log(run):
    text, (rec,) = run(_Stream([OPENAI_START], then_fail=True), target="local-coder")
    assert "event: error" in text and "message_stop" not in text
    assert rec["status"] == 502 and rec["lane"] == "openai"


def test_an_openai_stream_that_ends_before_the_model_finished_is_an_error_not_end_turn(run):
    text, (rec,) = run(_Stream([OPENAI_START]), target="local-coder")  # clean EOF: no finish_reason, no [DONE]
    assert "event: error" in text and "end_turn" not in text
    assert rec["status"] == 502


def test_a_finished_openai_stream_is_still_a_clean_end(run):
    done = ['data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n', "data: [DONE]\n\n"]
    text, (rec,) = run(_Stream([OPENAI_START, *done]), target="local-coder")
    assert "message_stop" in text and "event: error" not in text and rec["status"] == 200


def test_an_openai_stream_that_ends_with_done_but_no_finish_reason_is_accepted(run):
    text, (rec,) = run(_Stream([OPENAI_START, "data: [DONE]\n\n"]), target="local-coder")
    assert "message_stop" in text and rec["status"] == 200


def test_an_anthropic_stream_that_drops_is_logged_as_a_failure(run):
    start = ('event: message_start\ndata: {"type":"message_start","message":{"id":"m1",'
             '"usage":{"input_tokens":5}}}\n\n')
    text, (rec,) = run(_Stream([start], then_fail=True), target="claude-opus-5")
    assert "event: error" in text
    assert rec["status"] == 502 and rec["error"]
