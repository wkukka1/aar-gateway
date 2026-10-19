"""Opt-in usage sharing: off by default, nothing identifying in a record,
and never in the way of a request."""

from __future__ import annotations

import gzip
import json

import pytest

from aar_gateway import cli, share


@pytest.fixture(autouse=True)
def fresh_consent(monkeypatch):
    monkeypatch.setattr(share, "_consent", {"mtime": None, "on": False})
    monkeypatch.setattr(share, "_embed", lambda text: [1] * share.DIMS)  # no encoder download in tests


LOGGED = {"ts": "2026-10-17T13:45:12+00:00", "claude_session": "abc123", "conversation": "8885ac1c",
          "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest within 0.05 utility of the best", "status": 400, "subagent": False,
          "error": "invalid_request_error: messages.3: the secret token sk-ant-xyz is wrong",
          "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01, "cost_if_requested_usd": 0.05,
          "correction": True, "thread": "continue", "latency_s": 1.2}


def test_nothing_is_kept_unless_the_user_opts_in():
    assert not share.enabled()
    share.record(LOGGED, "fix the bug in /Users/me/secret-project/app.py")
    share.flush(5)
    assert share.queued() == []


def test_a_record_holds_no_text_ids_paths_or_exact_times():
    share.set_enabled(True)
    prompt = "fix the bug in /Users/me/secret-project/app.py, my key is sk-ant-xyz"
    share.record(LOGGED, prompt)
    share.flush(5)
    (rec,) = share.queued()
    text = json.dumps(rec)
    for leak in ("secret-project", "sk-ant-xyz", "abc123", "8885ac1c", "13:45", "Users"):
        assert leak not in text, leak
    assert rec["day"] == "2026-10-17" and rec["error_type"] == "invalid_request_error"
    assert rec["decision"] == "router" and rec["correction"] is True and rec["thread"] == "continue"
    assert rec["prompt_chars"] == 64 and len(rec["embedding"]) == share.DIMS


def test_subagents_and_tool_steps_add_nothing():
    share.set_enabled(True)
    share.record({**LOGGED, "subagent": True}, "search for x")
    share.record({**LOGGED, "reason": "side request (no tools), not routed"}, "write a title for this chat")
    share.record(LOGGED, None)  # a tool-loop step: no typed message
    share.flush(5)
    assert share.queued() == []


def test_the_compressed_embedding_is_small_integers_from_a_fixed_projection():
    import numpy as np

    v = np.random.default_rng(0).normal(size=384)
    a, b = share._compress(v), share._compress(v)
    assert a == b and len(a) == 64 and all(-127 <= x <= 127 for x in a)


def test_the_cli_turns_it_on_and_off_and_deletes(capsys):
    assert cli.main(["share"]) == 0 and "sharing      off" in capsys.readouterr().out
    cli.main(["share", "on"])
    assert share.enabled()
    share.record(LOGGED, "hello there")
    share.flush(5)
    cli.main(["share", "show"])
    assert "1 record(s)" in capsys.readouterr().out
    cli.main(["share", "off"])
    assert not share.enabled() and len(share.queued()) == 1  # kept until deleted
    cli.main(["share", "delete"])
    assert share.queued() == []


def test_setup_asks_once_and_only_a_yes_opts_in(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.setattr(cli, "_fetch_router", lambda source, target: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "")
    cli.main(["setup"])
    assert any("Share anonymous usage data" in q for q in asked) and not share.enabled()
    asked.clear()
    cli.main(["setup"])
    assert not any("Share anonymous" in q for q in asked)  # asked once: the answer is remembered
    cli.main(["setup", "--share"])
    assert share.enabled()


class FakeEndpoint:
    """The collection server: records what each POST carried and answers with a status."""

    def __init__(self, status=200, fail=False, during=None):
        self.posts, self.status, self.fail, self.during = [], status, fail, during

    def __call__(self, url, body, headers):
        if self.during:
            self.during()
        if self.fail:
            raise ConnectionError("endpoint down: key=SECRET123")
        self.posts.append((url, gzip.decompress(body).decode(), headers))
        return self.status


URL = "https://collect.example.org/v1/records"


def _queue(n):
    share.set_enabled(True)
    for i in range(n):
        share.record(LOGGED, f"message number {i}")
    share.flush(5)


def test_without_an_endpoint_nothing_is_sent():
    _queue(2)
    server = FakeEndpoint()
    assert "skipped" in share.upload(post=server) and len(share.queued()) == 2 and server.posts == []


def test_an_upload_posts_the_queue_as_gzip_jsonl_and_empties_it():
    share.set_endpoint(URL)
    _queue(3)
    server = FakeEndpoint()
    assert share.upload(post=server)["sent"] == 3
    ((url, body, headers),) = server.posts
    assert url == URL and len(body.splitlines()) == 3 and share.queued() == []
    assert headers["content-encoding"] == "gzip" and headers["x-aar-schema"] == "1"
    assert headers["x-aar-install"] == share._install_id() and "message number" not in body
    assert share.state()["sent_total"] == 3


def test_a_failed_upload_keeps_the_records_and_never_stores_the_error_text():
    share.set_endpoint(URL)
    _queue(2)
    assert share.upload(post=FakeEndpoint(fail=True)) == {"error": "ConnectionError"}
    assert len(share.queued()) == 2 and "SECRET123" not in share.state_path().read_text()


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_a_refused_upload_keeps_the_records(status):
    share.set_endpoint(URL)
    _queue(2)
    assert share.upload(post=FakeEndpoint(status=status)) == {"error": f"http_{status}"}
    assert len(share.queued()) == 2


def test_records_written_during_an_upload_stay_queued():
    share.set_endpoint(URL)
    _queue(2)
    late = lambda: share._append(LOGGED, "a late message")  # noqa: E731
    assert share.upload(post=FakeEndpoint(during=late))["sent"] == 2
    assert len(share.queued()) == 1


@pytest.mark.parametrize("url", ["http://collect.example.org/v1/records", "ftp://x/y", "collect.example.org", ""])
def test_only_https_or_a_local_address_is_accepted_as_the_endpoint(url):
    with pytest.raises(ValueError):
        share.set_endpoint(url)


def test_a_local_http_endpoint_is_allowed_for_testing():
    share.set_endpoint("http://127.0.0.1:9000/v1/records")
    share.set_endpoint("http://localhost:9000/v1/records")
    assert share.endpoint() == "http://localhost:9000/v1/records"


def test_the_endpoint_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_SHARE_ENDPOINT", URL)
    assert share.endpoint() == URL
    monkeypatch.setenv("AAR_GATEWAY_SHARE_ENDPOINT", "http://evil.example/x")
    assert share.endpoint() is None  # an unsafe address is ignored, never posted to


def test_status_names_the_endpoint_and_the_old_token_command_points_to_it(capsys):
    cli.main(["share", "endpoint", URL])
    cli.main(["share", "status"])
    assert URL in capsys.readouterr().out
    with pytest.raises(SystemExit, match="endpoint"):
        cli.main(["share", "token", "hf_whatever"])


def test_it_uploads_by_itself_once_enough_records_are_queued(monkeypatch):
    sent = []
    monkeypatch.setattr(share, "upload", lambda post=None: sent.append(1) or {"sent": 0})
    monkeypatch.setattr(share, "UPLOAD_AFTER", 3)
    share.set_endpoint(URL)
    share.state_path().parent.mkdir(parents=True, exist_ok=True)
    share.state_path().write_text('{"last_attempt": "2999-01-01T00:00:00+00:00"}')  # not due by time
    _queue(2)
    assert sent == []
    _queue(1)
    assert sent == [1]
