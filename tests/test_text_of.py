"""One text extractor, with the four join/filter rules the gateway had written out by hand."""

from __future__ import annotations

from aar_gateway.translate import text_of

BLOCKS = [{"type": "text", "text": "a"}, {"type": "tool_use", "name": "x"}, {"type": "text", "text": "b"},
          {"type": "image", "text": "c"}, "stray"]


def test_a_string_is_its_own_text():
    assert text_of("hi") == "hi" and text_of("hi", sep="\n") == "hi" and text_of(None) == ""


def test_only_text_blocks_joined_flat():  # translate._text
    assert text_of(BLOCKS) == "ab"


def test_only_text_blocks_joined_by_newlines():  # history._text
    assert text_of(BLOCKS, sep="\n") == "a\nb"


def test_any_block_with_text_joined_by_newlines():  # subagents._system_text
    assert text_of(BLOCKS, sep="\n", typed=False) == "a\n\nb\nc"


def test_any_block_with_text_joined_by_spaces():  # subagent_baseline._first_task
    assert text_of(BLOCKS, sep=" ", typed=False) == "a  b c"
