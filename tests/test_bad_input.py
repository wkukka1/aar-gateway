"""Malformed requests get a 400 in the API's own shape, never a 500 or an unhandled exception."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from aar_gateway import interceptor, server
from aar_gateway.config import load_settings

AUTH = {"authorization": "Bearer sk-ant-oat-not-a-real-token"}


@pytest.fixture
def client(tmp_path):
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  models:
    claude-opus-5: {lane: anthropic, proxy: x}
    local-coder: {lane: openai, base_url: "http://local.test/v1", proxy: x, prices: {input: 0, output: 0}}
""")
    seen = []
    interceptor.set_upstream_client(httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: seen.append(req) or httpx.Response(200, json={"input_tokens": 5}))))
    with TestClient(server.app) as c:
        interceptor.configure(load_settings(cfg), lambda body, ctx: "local-coder")
        c.seen = seen
        yield c
    interceptor.set_upstream_client(None)
    interceptor.configure(load_settings(None), None)


@pytest.mark.parametrize("raw", [b"[]", b'"text"', b"null", b"42", b"not json", b"\xff\xfe"])
def test_messages_with_a_body_that_is_not_a_json_object_is_a_400(client, raw):
    r = client.post("/v1/messages", content=raw, headers={**AUTH, "content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["type"] == "error" and r.json()["error"]["type"] == "invalid_request_error"


def test_a_content_block_the_openai_lane_cannot_translate_is_a_400(client):
    """A prior tool_use with no id passes the routing pre-check but not the translation."""
    body = {"model": "claude-opus-5", "max_tokens": 8, "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "start"},
                         {"role": "assistant", "content": [{"type": "tool_use", "input": {}}]},
                         {"role": "user", "content": "go on"}]}
    r = client.post("/v1/messages", json=body, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_a_message_with_no_role_stays_on_the_anthropic_lane(client):
    """The routing pre-check keeps what it cannot translate off the openai lane; the upstream judges it."""
    body = {"model": "claude-opus-5", "max_tokens": 8, "messages": [{"content": "no role"}]}
    assert client.post("/v1/messages", json=body, headers=AUTH).status_code == 200  # the mock upstream
    assert len(client.seen) == 1


def test_count_tokens_with_an_unreadable_body_is_left_to_the_upstream(client):
    r = client.post("/v1/messages/count_tokens", content=b"[]", headers={**AUTH, "content-type": "application/json"})
    assert r.status_code == 200 and len(client.seen) == 1  # forwarded, not a crash


@pytest.mark.parametrize("path", ["/api/preferences", "/api/subagent-spawn"])
@pytest.mark.parametrize("raw", [b"[]", b'"x"', b"not json"])
def test_api_routes_reject_a_body_that_is_not_a_json_object(client, path, raw):
    assert client.post(path, content=raw, headers={"content-type": "application/json"}).status_code == 400


@pytest.mark.parametrize("body", [{"quality_margin": []}, {"quality_margin": {"a": 1}}, {"role": []},
                                  {"role": "explore", "model": []}, {"mode": []}, {"preset": []}])
def test_preferences_with_the_wrong_value_types_is_a_400(client, body):
    assert client.post("/api/preferences", json=body).status_code == 400
