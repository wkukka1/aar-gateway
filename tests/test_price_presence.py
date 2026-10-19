"""A missing price is unknown: never Claude's rates, never zero, never a KeyError."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aar_gateway.config import load_settings
from aar_gateway.pricing import has_price
from aar_gateway.routing import build_picker, price_for
from aar_gateway.subagents import _pricier

BODY = {"model": "claude-opus-5", "tools": [], "messages": [{"role": "user", "content": "hi"}]}


class _Router:
    model_ids = ["strong", "weak"]

    def route_text(self, texts):
        return SimpleNamespace(score_frame=lambda: pd.DataFrame([{"strong": 0.8, "weak": 0.8}] * len(texts)))


def _settings(tmp_path, haiku_prices: str, local_prices: str = ""):
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text(f"""
interceptor:
  router: {{margin: 0.05}}
  models:
    claude-opus-5:    {{lane: anthropic, proxy: strong}}
    claude-haiku-4-5: {{lane: anthropic, proxy: weak, prices: {haiku_prices}}}
    local-coder: {{lane: openai, base_url: "http://local.test/v1", proxy: weak{local_prices}}}
""")
    return load_settings(cfg)


def _ctx():
    return {"requested": "claude-opus-5", "human_text": "hi", "est_prompt_tokens": 2000,
            "last_model": None, "last_prompt_tokens": None}


def test_has_price_needs_both_rates():
    assert has_price({"input": 1.0, "output": 2.0}) and has_price({"input": 0, "output": 0})  # free is a price
    assert not has_price(None) and not has_price({}) and not has_price({"input": 1.0})
    assert not has_price({"input": None, "output": 2.0})


def test_an_explicitly_empty_price_is_not_claudes_rate(tmp_path):
    settings = _settings(tmp_path, "{}")
    assert not has_price(price_for(settings, "claude-haiku-4-5"))
    decision = build_picker(settings, router_model=_Router())(BODY, _ctx())
    assert decision["skipped"]["claude-haiku-4-5"] == "no price"


def test_a_half_priced_model_is_skipped_not_a_keyerror(tmp_path):
    settings = _settings(tmp_path, "{input: 1.0}")
    decision = build_picker(settings, router_model=_Router())(BODY, _ctx())
    assert decision["skipped"]["claude-haiku-4-5"] == "no price"
    assert _pricier(settings, "claude-haiku-4-5", "claude-opus-5") is False


def test_a_subagent_kind_is_not_routed_by_price_when_either_price_is_half_given(tmp_path):
    """Comparing prices needs both rates of both models: otherwise the kind's choice is left to the general router."""
    from aar_gateway.subagents import role_decision

    agent = {"role": "explore", "source": "hook", "why": "x"}
    body = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "look around"}]}
    for haiku in ("{input: 1.0}", "{input: 1.0, output: 5.0}"):
        settings = _settings(tmp_path, haiku)
        settings["subagents"]["roles"] = {"explore": "claude-haiku-4-5"}
        decision = role_decision(agent, body, "claude-opus-5", settings, {}, 1000)
        if haiku == "{input: 1.0}":
            assert decision is None  # half-priced target
        else:
            assert decision and decision["model"] == "claude-haiku-4-5"  # fully priced and cheaper: routed
