"""The interceptor against a fake Anthropic upstream: Claude Code's request
must arrive upstream untouched (credentials, tools, beta query), the answer
must come back byte-for-byte, and the request log must get token counts but
never the credential."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import history, interceptor, server

SECRET = "Bearer sk-ant-oat-not-a-real-token"
CLAUDE_CODE_BODY = {
    "model": "claude-opus-5",
    "max_tokens": 64,
    "stream": False,
    "system": [{"type": "text", "text": "you are claude code"}],
    "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "messages": [{"role": "user", "content": "hi"}],
}
SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,'
    b'"cache_read_input_tokens":3}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"pong"}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


@pytest.fixture
def upstream_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_LOG", str(tmp_path / "requests.jsonl"))
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        body = json.loads(req.content) if req.content else {}
        if body.get("stream"):
            return httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})
        if req.url.path.endswith("count_tokens"):
            return httpx.Response(200, json={"input_tokens": 5})
        return httpx.Response(200, json={
            "type": "message", "model": body.get("model"),
            "content": [{"type": "text", "text": "pong"}],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        })

    interceptor.set_upstream_client(
        httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://upstream.test")
    )
    interceptor.set_model_picker(None)
    yield calls
    interceptor.set_upstream_client(None)
    interceptor.set_model_picker(None)


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def _log_records() -> list[dict]:
    path = interceptor.log_path()
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_request_reaches_upstream_untouched(client, upstream_calls):
    resp = client.post("/v1/messages?beta=true", json=CLAUDE_CODE_BODY,
                       headers={"authorization": SECRET, "anthropic-beta": "x-feature"})
    assert resp.status_code == 200
    (sent,) = upstream_calls
    assert sent.url.path == "/v1/messages" and sent.url.params["beta"] == "true"
    assert sent.headers["authorization"] == SECRET
    assert sent.headers["anthropic-beta"] == "x-feature"
    assert json.loads(sent.content) == CLAUDE_CODE_BODY  # tools, thinking, system all kept


def test_credential_never_reaches_the_log(client, upstream_calls):
    client.post("/v1/messages", json=CLAUDE_CODE_BODY, headers={"authorization": SECRET})
    text = interceptor.log_path().read_text()
    assert "sk-ant" not in text
    assert "you are claude code" not in text


def test_non_streaming_usage_is_logged(client, upstream_calls):
    client.post("/v1/messages", json=CLAUDE_CODE_BODY, headers={"authorization": SECRET})
    (rec,) = _log_records()
    assert rec["routed_model"] == "claude-opus-5"
    assert (rec["input_tokens"], rec["output_tokens"], rec["status"]) == (11, 7, 200)


def test_stream_is_relayed_byte_for_byte_and_usage_read_on_the_way(client, upstream_calls):
    body = {**CLAUDE_CODE_BODY, "stream": True}
    resp = client.post("/v1/messages", json=body, headers={"authorization": SECRET})
    assert resp.status_code == 200
    assert resp.content == SSE
    (rec,) = _log_records()
    assert (rec["input_tokens"], rec["output_tokens"], rec["cache_read_input_tokens"]) == (11, 7, 3)


def test_model_picker_reroutes_and_stats_show_it(client, upstream_calls):
    interceptor.set_model_picker(lambda body, ctx: "claude-haiku-4-5")
    client.post("/v1/messages", json=CLAUDE_CODE_BODY, headers={"authorization": SECRET})
    assert json.loads(upstream_calls[0].content)["model"] == "claude-haiku-4-5"
    (rec,) = _log_records()
    assert (rec["requested_model"], rec["routed_model"]) == ("claude-opus-5", "claude-haiku-4-5")
    stats = client.get("/stats").json()["models"]["claude-haiku-4-5"]
    assert stats["rerouted"] == 1 and stats["output_tokens"] == 7


def test_other_anthropic_endpoints_pass_through_without_a_log_record(client, upstream_calls):
    resp = client.post("/v1/messages/count_tokens?beta=true", json=CLAUDE_CODE_BODY,
                       headers={"authorization": SECRET})
    assert resp.json() == {"input_tokens": 5}
    assert upstream_calls[0].url.path == "/v1/messages/count_tokens"
    assert _log_records() == []


def test_without_credentials_nothing_is_forwarded(client, upstream_calls):
    resp = client.post("/v1/messages", json={"model": "claude-opus-5", "max_tokens": 8,
                                             "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401 and resp.json()["error"]["type"] == "authentication_error"
    assert upstream_calls == []


def test_catch_all_refuses_requests_without_credentials(client, upstream_calls):
    assert client.post("/v1/messages/count_tokens", json={}).status_code == 401
    assert upstream_calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_log_write_failure_does_not_break_a_served_response(client, upstream_calls, monkeypatch, stream):
    interceptor.log_path().mkdir()  # appending to a directory fails like a full disk or a locked file
    resp = client.post("/v1/messages", json={**CLAUDE_CODE_BODY, "stream": stream}, headers={"authorization": SECRET})
    assert resp.status_code == 200
    assert resp.content == SSE if stream else resp.json()["content"][0]["text"] == "pong"


def test_a_log_rotation_blocked_by_another_program_does_not_break_the_response(client, upstream_calls, monkeypatch):
    from pathlib import Path

    interceptor.log_path().write_text('{"model": "a"}\n', encoding="utf-8")
    monkeypatch.setattr(interceptor, "MAX_LOG_BYTES", 1)

    def blocked(self, target):
        raise PermissionError(13, "in use by another process")

    monkeypatch.setattr(Path, "replace", blocked)
    resp = client.post("/v1/messages", json=CLAUDE_CODE_BODY, headers={"authorization": SECRET})
    assert resp.status_code == 200


def test_a_failed_history_write_does_not_stop_the_request(client, upstream_calls):
    """A full disk under the history costs one kept turn, not the user's request."""
    history.history_path().parent.mkdir(parents=True, exist_ok=True)
    history.history_path().mkdir()  # appending to a directory fails like a full disk or a locked file
    resp = client.post("/v1/messages", json=CLAUDE_CODE_BODY, headers={"authorization": SECRET})
    assert resp.status_code == 200
    assert len(upstream_calls) == 1
