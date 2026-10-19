"""The real HTTP path of usage sharing, against a collection server running on this computer."""

from __future__ import annotations

import gzip
import http.server
import json
import threading

import pytest

from aar_gateway import share

LOGGED = {"ts": "2026-10-18T10:00:00+00:00", "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest", "status": 200, "subagent": False, "input_tokens": 1, "output_tokens": 1}


@pytest.fixture
def server():
    received, status = [], {"code": 200}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, dict(self.headers), gzip.decompress(body).decode()))
            self.send_response(status["code"])
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1/records", received, status
    httpd.shutdown()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(share, "_consent", {"mtime": None, "on": False})
    monkeypatch.setattr(share, "_embed", lambda text: [1] * share.DIMS)


def _queue(n):
    share.state_path().parent.mkdir(parents=True, exist_ok=True)
    share.state_path().write_text('{"last_attempt": "2999-01-01T00:00:00+00:00"}')  # not due: only the explicit upload sends
    share.set_enabled(True)
    for i in range(n):
        share.record(LOGGED, f"message number {i}")
    share.flush(5)


def test_records_arrive_over_http_as_gzip_jsonl_with_no_text(server):
    url, received, _ = server
    share.set_endpoint(url)
    _queue(3)
    assert share.upload() == {"sent": 3}
    ((path, headers, body),) = received
    rows = [json.loads(line) for line in body.splitlines()]
    assert path == "/v1/records" and len(rows) == 3 and share.queued() == []
    assert headers["Content-Encoding"] == "gzip" and headers["X-Aar-Schema"] == "1"
    assert "message number" not in body and all("prompt" not in r and "text" not in r for r in rows)
    assert all(len(r["embedding"]) == share.DIMS for r in rows)


def test_a_server_error_keeps_the_records(server):
    url, received, status = server
    status["code"] = 503
    share.set_endpoint(url)
    _queue(2)
    assert share.upload() == {"error": "http_503"} and len(share.queued()) == 2


def test_a_server_that_is_not_there_keeps_the_records():
    share.set_endpoint("http://127.0.0.1:9/v1/records")  # nothing listens on the discard port
    _queue(2)
    assert "error" in share.upload() and len(share.queued()) == 2
