"""Every error the gateway itself produces has the Anthropic error shape clients parse."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aar_gateway import interceptor, server, translate


def test_error_body_is_the_anthropic_envelope():
    assert translate.error_body("api_error", "boom") == {"type": "error", "error": {"type": "api_error", "message": "boom"}}


def test_the_sse_error_event_carries_the_same_envelope():
    event = translate.stream_error("lost").decode()
    assert event.startswith("event: error\n") and '"type": "api_error"' in event and '"message": "lost"' in event


def test_every_response_the_gateway_makes_up_uses_it(monkeypatch):
    monkeypatch.delenv("AAR_GATEWAY_ALLOWED_HOSTS", raising=False)
    with TestClient(server.app, base_url="http://127.0.0.1:8787") as client:
        wrong_host = client.get("/health", headers={"host": "evil.example"})
        wrong_origin = client.post("/api/preferences", json={}, headers={"origin": "https://evil.example"})
        no_login = client.post("/v1/messages", json={"model": "claude-opus-5", "messages": []})
        bad_body = client.post("/v1/messages", content=b"[]")
    for response, status, kind in ((wrong_host, 403, "forbidden"), (wrong_origin, 403, "forbidden"),
                                   (no_login, 401, "authentication_error"), (bad_body, 400, "invalid_request_error")):
        assert response.status_code == status, (status, response.text)
        body = response.json()
        assert body["type"] == "error" and body["error"]["type"] == kind and body["error"]["message"]


def test_the_interceptor_helper_keeps_its_signature():
    r = interceptor.error_response(502, "down")
    assert r.status_code == 502 and r.body == b'{"type":"error","error":{"type":"api_error","message":"down"}}'
