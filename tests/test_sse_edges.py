"""Streaming edge cases: CRLF event separators, a runaway event, and dashboard listeners that never started."""

from __future__ import annotations

import asyncio
import json

from aar_gateway import interceptor, server

START = {"type": "message_start", "message": {"id": "m1", "usage": {"input_tokens": 11}}}
DELTA = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}


def _event(payload, eol=b"\n"):
    return b"event: " + payload["type"].encode() + eol + b"data: " + json.dumps(payload).encode() + eol + eol


def test_usage_is_read_from_a_stream_that_separates_events_with_crlf():
    tap = interceptor._SSEUsageTap()
    for chunk in (_event(START, b"\r\n"), _event(DELTA, b"\r\n")):
        tap.feed(chunk)
    assert tap.usage == {"input_tokens": 11, "output_tokens": 7} and tap.stop_reason == "end_turn"


def test_a_crlf_split_across_two_chunks_is_still_one_separator():
    tap = interceptor._SSEUsageTap()
    raw = _event(START, b"\r\n") + _event(DELTA, b"\r\n")
    for i in range(0, len(raw), 3):  # every possible split, including between \r and \n
        tap.feed(raw[i:i + 3])
    assert tap.usage == {"input_tokens": 11, "output_tokens": 7} and tap.message_id == "m1"


def test_an_event_that_never_ends_cannot_grow_the_buffer_without_bound():
    tap = interceptor._SSEUsageTap()
    for _ in range(40):
        tap.feed(b"x" * 100_000)  # 4 MB with no separator at all
    assert len(tap._buf) <= interceptor._SSEUsageTap.MAX_EVENT_BYTES
    tap.feed(b"\n\n" + _event(DELTA))  # the stream recovers at the next event
    assert tap.stop_reason == "end_turn"


def test_a_dashboard_stream_that_is_never_iterated_leaves_no_listener():
    class Req:
        async def is_disconnected(self):
            return True

    async def open_and_drop():
        response = await server.events(Req())  # the client went away before the first byte
        return response, len(interceptor._listeners)

    before = len(interceptor._listeners)
    _, during = asyncio.run(open_and_drop())
    assert during == before
