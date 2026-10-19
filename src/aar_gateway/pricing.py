"""Dollar cost of one request from its token usage -- *observed* cost, for the
request log and the dashboard. Predicted cost for routing lives in
:mod:`aar_gateway.routing`; the two are deliberately separate.

On a Claude subscription nothing is billed per token, so these are
API-equivalent dollars: what the same usage costs at API prices, which is
also the best available proxy for how much plan quota a request consumed.
"""

from __future__ import annotations

from collections.abc import Mapping

#: $/MTok, Anthropic first-party API rates (platform.claude.com/docs/en/about-
#: claude/pricing, read 2026-10-07). Cache writes are priced off ``input``:
#: 1.25x for the 5-minute TTL, 2x for the 1-hour TTL. ``over`` is a
#: prompt-length tier: Haiku 5.5 charges more for prompts over 100k tokens.
#: Override or extend in the gateway config.
CLAUDE_PRICES: dict[str, dict] = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.10},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                         "over": {"prompt_tokens": 100_000,
                                  "input": 0.50, "output": 2.50, "cache_read": 0.05}},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10},
}

WRITE_5M = 1.25
WRITE_1H = 2.0


def canonical_model(model: str | None) -> str | None:
    """Claude Code may tag a model with a context suffix (``claude-opus-5[1m]``)."""
    return model.split("[", 1)[0] if model else model


def has_price(price: Mapping | None) -> bool:
    """Both an input and an output rate. Free (``0``) is a price; a missing or
    half-given one is unknown -- never Claude's rate, never zero."""
    return bool(price) and price.get("input") is not None and price.get("output") is not None


def price_at(price: Mapping | None, prompt_tokens: float) -> Mapping | None:
    """The rates that apply to a prompt of this many tokens (input + cache
    read + cache write): the ``over`` tier past its threshold, else the base."""
    if not price:
        return price
    over = price.get("over")
    if over and prompt_tokens > over["prompt_tokens"]:
        return {k: v for k, v in over.items() if k != "prompt_tokens"}
    return price


def cache_read_rate(price: Mapping[str, float]) -> float:
    return price.get("cache_read", 0.1 * price["input"])


def prompt_tokens(usage: Mapping) -> int:
    """Every token of the prompt: fresh input, cache reads and cache writes."""
    return sum(usage.get(k) or 0 for k in
               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def cost_usd(price: Mapping[str, float] | None, usage: Mapping | None) -> float | None:
    """``None`` when the model has no price or there is no usage -- unknown,
    never silently ``0.0``. A write with no TTL breakdown is priced at the
    5-minute rate (the API default)."""
    if not has_price(price) or not usage:
        return None
    price = price_at(price, prompt_tokens(usage))
    per = 1e-6
    written = usage.get("cache_creation_input_tokens") or 0
    split = usage.get("cache_creation") or {}
    w1h = split.get("ephemeral_1h_input_tokens") or 0
    w5m = split.get("ephemeral_5m_input_tokens")
    if w5m is None:
        w5m = max(written - w1h, 0)
    return per * (
        (usage.get("input_tokens") or 0) * price["input"]
        + (usage.get("output_tokens") or 0) * price["output"]
        + (usage.get("cache_read_input_tokens") or 0) * cache_read_rate(price)
        + w5m * price["input"] * WRITE_5M
        + w1h * price["input"] * WRITE_1H
    )
