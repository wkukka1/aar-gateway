"""The picker's decisions, with a fake router so the predicted qualities are
known: cheapest model within the user's quality margin, never pricier than
what the user picked, and never "saving" money by throwing away a warm cache."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aar_gateway.config import load_settings
from aar_gateway.routing import build_picker

EASY = {"strong": 0.80, "mid": 0.78, "weak": 0.77}
HARD = {"strong": 0.72, "mid": 0.55, "weak": 0.50}
TOOL = {"name": "Read", "input_schema": {"type": "object"}}


class FakeRouter:
    model_ids = ["strong", "mid", "weak"]

    def __init__(self):
        self.scores = EASY

    def route_text(self, texts):
        return SimpleNamespace(score_frame=lambda: pd.DataFrame([self.scores] * len(texts)))


@pytest.fixture
def settings(tmp_path):
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  router: {margin: 0.05}
  models:
    claude-opus-5:    {lane: anthropic, proxy: strong, context_window: 1000000}
    claude-sonnet-5:  {lane: anthropic, proxy: mid, context_window: 1000000}
    claude-haiku-4-5: {lane: anthropic, proxy: weak, context_window: 200000}
    local-coder:
      lane: openai
      base_url: http://local.test/v1
      proxy: weak
      context_window: 32000
      max_tools: 2
      prices: {input: 0, output: 0}
""")
    return load_settings(cfg)


@pytest.fixture
def router():
    return FakeRouter()


def ctx(**kw):
    return {"requested": "claude-opus-5", "human_text": "hi", "est_prompt_tokens": 2000,
            "last_model": None, "last_prompt_tokens": None, **kw}


BODY = {"model": "claude-opus-5", "tools": [TOOL], "messages": [{"role": "user", "content": "hi"}]}


def test_easy_turn_goes_to_the_cheapest_model_that_can_take_it(settings, router):
    pick = build_picker(settings, router_model=router)
    assert pick(BODY, ctx())["model"] == "local-coder"


def test_untranslatable_request_skips_the_openai_lane(settings, router):
    body = {**BODY, "tools": [TOOL] * 3}
    decision = build_picker(settings, router_model=router)(body, ctx())
    assert decision["model"] == "claude-haiku-4-5"
    assert "not translatable" in decision["skipped"]["local-coder"]


def test_hard_turn_stays_on_the_strongest(settings, router):
    router.scores = HARD
    decision = build_picker(settings, router_model=router)(BODY, ctx())
    assert decision["model"] == "claude-opus-5"
    assert decision["predicted_quality"] == decision["best_predicted_quality"] == 0.72


def test_never_pricier_than_the_model_the_user_picked(settings, router):
    router.scores = {"strong": 0.72, "mid": 0.62, "weak": 0.45}
    decision = build_picker(settings, router_model=router)(BODY, ctx(requested="claude-sonnet-5"))
    assert decision["model"] == "claude-sonnet-5"
    assert decision["skipped"]["claude-opus-5"] == "pricier than the requested model"


def test_with_upgrades_allowed_a_hard_turn_goes_up_to_the_stronger_model(settings, router):
    router.scores = {"strong": 0.72, "mid": 0.62, "weak": 0.45}
    decision = build_picker(settings, router_model=router)(
        BODY, ctx(requested="claude-sonnet-5", allow_upgrade=True))
    assert decision["model"] == "claude-opus-5"


def test_a_warm_cache_is_not_thrown_away_for_a_short_stay(settings, router):
    """200k-token prefix cached on Opus. Over one request, rewriting it into
    Sonnet's cache costs more than reading it from Opus's: stay."""
    settings["router"]["horizon_requests"] = 1
    c = ctx(est_prompt_tokens=201_000, last_model="claude-opus-5", last_prompt_tokens=200_000)
    decision = build_picker(settings, router_model=router)(BODY, c)
    assert decision["model"] == "claude-opus-5"
    assert decision["skipped"]["claude-haiku-4-5"] == "prompt exceeds context window"


def test_a_switch_that_pays_back_over_the_horizon_is_taken(settings, router):
    """Same conversation, 20-request horizon: Sonnet's cheaper cache reads
    repay the one-time rewrite, so an easy turn moves down."""
    c = ctx(est_prompt_tokens=201_000, last_model="claude-opus-5", last_prompt_tokens=200_000)
    assert build_picker(settings, router_model=router)(BODY, c)["model"] == "claude-sonnet-5"


def test_a_model_warm_from_another_session_counts_as_warm(settings, router):
    """Claude Code's system prompt and tools are shared across sessions: a
    new session on a model that cached them recently starts warm."""
    settings["router"]["horizon_requests"] = 1
    c = ctx(est_prompt_tokens=201_000, warm_tokens={"claude-opus-5": 195_000})
    assert build_picker(settings, router_model=router)(BODY, c)["model"] == "claude-opus-5"


def test_models_rejected_for_this_client_are_skipped(settings, router):
    c = ctx(blocked={"local-coder": "too old", "claude-haiku-4-5": "too old"})
    decision = build_picker(settings, router_model=router)(BODY, c)
    assert decision["model"] == "claude-sonnet-5"
    assert decision["skipped"]["local-coder"].startswith("rejected earlier")


def test_cold_start_with_a_big_prompt_still_routes_down(settings, router):
    c = ctx(est_prompt_tokens=150_000)
    assert build_picker(settings, router_model=router)(BODY, c)["model"] == "claude-haiku-4-5"


def test_user_cost_preference_changes_the_decision(settings, router):
    pick = build_picker(settings, router_model=router)
    assert pick(BODY, ctx(quality_margin=0.0))["model"] == "claude-opus-5"
    decision = pick(BODY, ctx(quality_margin=0.02))
    assert decision["model"] == "claude-sonnet-5" and decision["quality_margin"] == 0.02


def test_unconfigured_model_is_left_alone(settings, router):
    assert build_picker(settings, router_model=router)(BODY, ctx(requested="some-other-model")) is None


def test_router_disabled_means_no_picker(settings):
    assert build_picker(settings) is None


def test_a_conversation_does_not_flip_flop_on_a_small_quality_gap(settings, router):
    """Found live: a session went Opus -> Sonnet -> Opus on noisy scores, and
    the cache rewrites made it dearer than no router. The current model keeps
    the conversation unless the other is better by margin + switch_margin."""
    settings["router"]["horizon_requests"] = 1000  # make cost irrelevant: quality decides
    router.scores = {"strong": 0.80, "mid": 0.72, "weak": 0.40}  # Sonnet 0.08 behind: outside 0.05
    on_sonnet = ctx(requested="claude-opus-5", last_model="claude-sonnet-5", last_prompt_tokens=2000)
    decision = build_picker(settings, router_model=router)(BODY, on_sonnet)
    assert decision["model"] == "claude-sonnet-5" and "stays" in decision["reason"]
    assert decision["predicted_quality"] == 0.72  # the logged score is the router's, not the bonus
    router.scores = {"strong": 0.80, "mid": 0.60, "weak": 0.40}  # 0.20 behind: worth the move
    assert build_picker(settings, router_model=router)(BODY, on_sonnet)["model"] == "claude-opus-5"


def test_equal_cache_read_prices_still_switch_on_measured_cache_writes(tmp_path, router):
    """Found live: Opus 5.5 and Sonnet 5.5 both read cache at $0.20/MTok, so a
    reads-only estimate made a warm Opus look as cheap as Sonnet and nothing
    ever switched, at any margin. Each request also *writes* what the
    conversation added since -- at 2x input, $8 vs $4 per MTok -- and the
    conversation's measured profile prices that on each model."""
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text("""
interceptor:
  router: {margin: 0.15}
  models:
    claude-opus-5-5:   {lane: anthropic, proxy: strong, context_window: 1000000}
    claude-sonnet-5-5: {lane: anthropic, proxy: mid, context_window: 1000000}
""")
    settings = load_settings(cfg)
    pick = build_picker(settings, router_model=router)
    on_opus = ctx(requested="claude-opus-5-5", est_prompt_tokens=45_000,
                  last_model="claude-opus-5-5", last_prompt_tokens=44_000)
    reads_only = {"input_tokens": 2, "cache_creation_input_tokens": 0, "output_tokens": 250}
    assert pick(BODY, {**on_opus, "usage_profile": reads_only})["model"] == "claude-opus-5-5"
    measured = {"input_tokens": 2, "cache_creation_input_tokens": 5_000, "output_tokens": 250}
    assert pick(BODY, {**on_opus, "usage_profile": measured})["model"] == "claude-sonnet-5-5"


def test_graded_traffic_replaces_the_stand_ins_when_it_has_neighbours(settings, router):
    """The user's own graded turns say Haiku is as good as Opus here; the
    benchmark stand-ins say it is far worse. Graded data wins, and the
    decision says where its quality came from."""
    router.scores = HARD

    def graded(text):
        return {"quality": {"claude-opus-5": 0.8, "claude-haiku-4-5": 0.79},
                "neighbors": 5, "graded_prompts": 40, "max_similarity": 0.8}

    settings["models"]["claude-haiku-4-5"].pop("rejects", None)
    decision = build_picker(settings, router_model=router, graded=graded)(BODY, ctx())
    assert decision["model"] == "claude-haiku-4-5"
    assert decision["quality_source"].startswith("graded (5 similar")
    assert decision["skipped"]["claude-sonnet-5"] == "no graded data for this model"


def test_no_graded_neighbours_falls_back_to_the_stand_ins(settings, router):
    router.scores = HARD
    decision = build_picker(settings, router_model=router, graded=lambda text: None)(BODY, ctx())
    assert decision["model"] == "claude-opus-5" and decision["quality_source"] == "benchmark stand-ins"


def test_stand_ins_are_calibrated_on_the_users_grades(settings, router, tmp_path):
    """The stand-ins put Haiku 0.20 below Opus; the user's grades put it 0.01
    below. With enough grades the offset moves Haiku back within the margin."""
    import json as _json

    router.scores = {"strong": 0.70, "mid": 0.55, "weak": 0.50}
    graded = tmp_path / "graded.jsonl"
    rows = []
    for i in range(200):  # the offset is shrunk by n/(n+20): near full strength at 200
        for model, score in {"claude-opus-5": 0.80, "claude-haiku-4-5": 0.79}.items():
            rows.append({"prompt_hash": f"h{i}", "prompt": f"question {i}", "model": model, "score": score})
    graded.write_text("\n".join(_json.dumps(r) for r in rows))
    settings["models"]["claude-haiku-4-5"].pop("rejects", None)

    plain = build_picker(settings, router_model=router)(BODY, ctx())
    assert plain["model"] == "claude-opus-5"
    calibrated = build_picker(settings, router_model=router, graded_file=graded)(BODY, ctx())
    assert calibrated["model"] == "claude-haiku-4-5"
    assert calibrated["quality_source"] == "benchmark stand-ins, calibrated on your grades"


def test_calibration_is_shrunk_while_grades_are_few(settings, router, tmp_path):
    import json as _json

    from aar_gateway.routing import calibration_offsets

    router.scores = {"strong": 0.5, "mid": 0.5, "weak": 0.5}
    graded = tmp_path / "graded.jsonl"
    graded.write_text("\n".join(_json.dumps({"prompt_hash": f"h{i}", "prompt": f"q{i}",
                                             "model": "claude-opus-5", "score": 0.9}) for i in range(5)))
    off = calibration_offsets(graded, router, settings, shrink=20)
    assert abs(off["claude-opus-5"] - 0.4 * 5 / 25) < 1e-9


def test_a_turns_prediction_is_blended_with_the_previous_turns(settings, router):
    """One noisy message mustn't flip a conversation: the previous turn's
    per-model prediction carries half the weight (turn_smoothing 0.5)."""
    pick = build_picker(settings, router_model=router)
    router.scores = HARD
    first = pick(BODY, ctx())
    assert first["model"] == "claude-opus-5" and set(first["model_quality"]) >= {"claude-opus-5"}
    router.scores = {"strong": 0.72, "mid": 0.72, "weak": 0.72}  # this message alone: anything goes
    alone = pick(BODY, ctx())
    blended = pick(BODY, ctx(prev_quality=first["model_quality"]))
    assert alone["model"] != "claude-opus-5"
    assert blended["model_quality"]["claude-opus-5"] == pytest.approx(
        0.5 * 0.72 + 0.5 * first["model_quality"]["claude-opus-5"], abs=1e-3)
    settings["router"]["turn_smoothing"] = 1.0
    assert build_picker(settings, router_model=router)(BODY, ctx(prev_quality=first["model_quality"]))[
        "model_quality"] == alone["model_quality"]
    # turned off in the dashboard: each message judged alone
    off = pick(BODY, {**ctx(prev_quality=first["model_quality"]), "turn_smoothing": False})
    assert off["model_quality"] == alone["model_quality"]


def test_calibration_status_never_waits_for_the_picker_lock(settings, router, tmp_path):
    """/health reports calibration; the picker lock is held across encoder forwards,
    so a /health that took it would stall the event loop and read as 'gateway down'."""
    import json as _json
    import threading

    graded = tmp_path / "graded.jsonl"
    graded.write_text("\n".join(_json.dumps({"prompt_hash": f"h{i}", "prompt": f"q{i}",
                                             "model": "claude-opus-5", "score": 0.9}) for i in range(3)))
    pick = build_picker(settings, router_model=router, graded_file=graded)
    assert pick.calibration_status()["graded_prompts"] == 3  # known from the start, not after a first request

    inside, release = threading.Event(), threading.Event()
    real = router.route_text

    def slow(texts):
        inside.set()
        release.wait(10)
        return real(texts)

    router.route_text = slow
    worker = threading.Thread(target=lambda: pick(BODY, ctx()), daemon=True)
    worker.start()
    assert inside.wait(5)  # the picker now holds its lock inside an encoder forward
    answered = []
    probe = threading.Thread(target=lambda: answered.append(pick.calibration_status()), daemon=True)
    probe.start()
    probe.join(2)
    still_held = not answered  # checked while the lock is still held
    release.set()
    worker.join(5)
    assert not still_held and answered[0]["graded_prompts"] == 3
