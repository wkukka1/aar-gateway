"""Observed cost must match what Anthropic actually charges -- anchored to
numbers the real `claude` CLI reported for real requests in this repo."""

from __future__ import annotations

import pytest

from aar_gateway.pricing import CLAUDE_PRICES, canonical_model, cost_usd


def test_one_hour_cache_write_matches_what_claude_code_reported():
    """`claude -p ... --model sonnet` reported costUSD 0.18786 for 2 input,
    4 output and 46,954 tokens written to the 1-hour cache."""
    usage = {"input_tokens": 2, "output_tokens": 4, "cache_creation_input_tokens": 46954,
             "cache_creation": {"ephemeral_1h_input_tokens": 46954, "ephemeral_5m_input_tokens": 0}}
    assert cost_usd(CLAUDE_PRICES["claude-sonnet-5"], usage) == pytest.approx(0.18786, abs=1e-5)


def test_plain_tokens_match_what_claude_code_reported():
    """Same run: Haiku 4.5, 900 input + 11 output = costUSD 0.000955."""
    usage = {"input_tokens": 900, "output_tokens": 11}
    assert cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], usage) == pytest.approx(0.000955)


def test_cache_reads_use_the_models_own_read_rate():
    """Opus 5.5 reads cost 0.05x input, not the usual 0.1x."""
    usage = {"cache_read_input_tokens": 1_000_000}
    assert cost_usd(CLAUDE_PRICES["claude-opus-5-5"], usage) == pytest.approx(0.20)
    assert cost_usd(CLAUDE_PRICES["claude-opus-5"], usage) == pytest.approx(0.50)


def test_write_without_ttl_breakdown_is_priced_at_the_5_minute_rate():
    usage = {"cache_creation_input_tokens": 1_000_000}
    assert cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], usage) == pytest.approx(1.25)


def test_unknown_price_or_usage_is_none_not_zero():
    assert cost_usd(None, {"input_tokens": 5}) is None
    assert cost_usd(CLAUDE_PRICES["claude-haiku-4-5"], None) is None
    assert cost_usd({"input": None, "output": None}, {"input_tokens": 5}) is None


def test_context_suffix_is_stripped():
    assert canonical_model("claude-opus-5[1m]") == "claude-opus-5"


def test_sonnet_5_5_reads_cache_at_its_own_rate():
    """Live pricing page, 2026-10-07: $0.10/MTok (0.05x of $2), not Opus 5.5's $0.20."""
    assert cost_usd(CLAUDE_PRICES["claude-sonnet-5-5"], {"cache_read_input_tokens": 1_000_000}) \
        == pytest.approx(0.10)


def test_haiku_5_5_charges_more_past_100k_prompt_tokens():
    short = {"input_tokens": 50_000, "output_tokens": 1_000}
    long = {"input_tokens": 50_000, "cache_read_input_tokens": 60_000, "output_tokens": 1_000}
    assert cost_usd(CLAUDE_PRICES["claude-haiku-5-5"], short) == pytest.approx(50_000 * 0.10e-6 + 1_000 * 0.50e-6)
    assert cost_usd(CLAUDE_PRICES["claude-haiku-5-5"], long) == pytest.approx(
        50_000 * 0.50e-6 + 60_000 * 0.05e-6 + 1_000 * 2.50e-6)
