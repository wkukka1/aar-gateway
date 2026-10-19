"""The dashboard's live path: new requests are announced on a stream the
moment they are logged, and re-reading the log parses only what was
appended -- and the gateway can be stopped by name."""

from __future__ import annotations

import asyncio
import json

from aar_gateway import cli, interceptor


def test_the_log_is_read_incrementally_and_survives_rotation():
    path = interceptor.log_path()
    path.write_text(json.dumps({"n": 1}) + "\n" + json.dumps({"n": 2}) + "\n")
    assert [r["n"] for r in interceptor.read_records()] == [1, 2]
    with path.open("a") as f:
        f.write(json.dumps({"n": 3}) + "\n" + '{"n": ')  # the last line is still being written
    assert [r["n"] for r in interceptor.read_records()] == [1, 2, 3]
    with path.open("a") as f:
        f.write("4}\n")
    assert [r["n"] for r in interceptor.read_records()] == [1, 2, 3, 4]
    path.replace(path.with_name(path.name + ".1"))  # rotated: its records still count
    path.write_text(json.dumps({"n": 5}) + "\n")
    assert [r["n"] for r in interceptor.read_records()] == [1, 2, 3, 4, 5]
    path.replace(path.with_name(path.name + ".1"))  # rotated again: only one old file is kept
    path.write_text(json.dumps({"n": 6}) + "\n")
    assert [r["n"] for r in interceptor.read_records()] == [5, 6]


def test_listeners_hear_about_each_logged_request():
    async def scenario():
        queue = interceptor.subscribe()
        try:
            interceptor._append_record({"status": 200})
            interceptor._append_record({"status": 200})  # coalesced: one pending update is enough
            assert await asyncio.wait_for(queue.get(), timeout=1) is True
            assert queue.empty()
        finally:
            interceptor.unsubscribe(queue)
        assert not interceptor._listeners

    asyncio.run(scenario())


def test_stop_without_a_running_gateway_says_so(capsys):
    cli.main(["stop"])
    assert "not running" in capsys.readouterr().out
