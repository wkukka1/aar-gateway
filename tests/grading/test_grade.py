"""The grader with a fake `claude` CLI: every tier is replayed read-only in
the turn's folder, the judge sees anonymised answers, and grades land per
(turn, model) -- never twice for the same prompt."""

from __future__ import annotations

import json

import pytest

from aar_gateway.grading import grade as g
from aar_gateway.claude_cli import LLMResponse


class FakeClaude:
    calls: list = []

    def __init__(self, model, timeout=None):
        self.model = model

    def complete(self, prompt, **kw):
        FakeClaude.calls.append((self.model, kw.get("cwd"), tuple(kw.get("extra_args", ()))))
        if prompt.startswith(g.JUDGE_INSTRUCTIONS[:40]):
            labels = [line.split()[2] for line in prompt.splitlines() if line.startswith("=== Answer ")]
            # score by which tier wrote it -- the answers carry their author for the test
            verdict = {}
            for lab in labels:
                body = prompt.split(f"=== Answer {lab} ===\n", 1)[1].split("\n\n=== Answer", 1)[0]
                verdict[lab] = {"score": 9 if "opus" in body else 7 if "sonnet" in body else 4,
                                "reason": "ok"}
            return LLMResponse(content="Here you go:\n" + json.dumps(verdict))
        return LLMResponse(content=f"answer from {self.model}", meta={"subscription_cost_usd_equivalent": 0.01})


@pytest.fixture
def fake(monkeypatch):
    FakeClaude.calls = []
    monkeypatch.setattr(g, "ClaudeSubscriptionAdapter", FakeClaude)
    return FakeClaude


def turn(tmp_path, prompt="explain the router", h="h1"):
    return {"prompt_hash": h, "prompt": prompt, "cwd": str(tmp_path)}


def test_each_tier_replayed_read_only_in_the_turns_folder_then_judged(fake, tmp_path):
    rows = g.grade_turn(turn(tmp_path), g.DEFAULT_TIERS, g.DEFAULT_JUDGE, timeout=5, rng=__import__("random").Random(1))
    replays = [c for c in fake.calls[:3]]
    assert [c[0] for c in replays] == list(g.DEFAULT_TIERS)
    assert all(c[1] == str(tmp_path) and c[2] == g.READ_ONLY for c in replays)
    assert {r["model"]: r["score"] for r in rows} == {
        "claude-opus-5-5": 0.9, "claude-sonnet-5-5": 0.7, "claude-haiku-5-5": 0.4}


def test_judge_sees_letters_not_model_names(fake, tmp_path):
    g.grade_turn(turn(tmp_path), g.DEFAULT_TIERS, g.DEFAULT_JUDGE, timeout=5, rng=__import__("random").Random(1))
    judge_prompt = [c for c in fake.calls if c[0] == g.DEFAULT_JUDGE][-1]
    assert judge_prompt  # the judge was called
    assert "=== Answer A ===" in g.judge_prompt("q", {"A": "x"})


def test_a_prompt_is_graded_once(fake, tmp_path):
    turns, out = tmp_path / "turns.jsonl", tmp_path / "graded.jsonl"
    turns.write_text("\n".join(json.dumps(turn(tmp_path, h=h)) for h in ["a", "a", "b"]))
    g.grade(10, turns_file=turns, out_file=out, workers=1)
    g.grade(10, turns_file=turns, out_file=out, workers=1)
    rows = g.read_jsonl(out)
    assert sorted({r["prompt_hash"] for r in rows}) == ["a", "b"] and len(rows) == 6


def test_unparseable_verdict_is_no_score_not_zero():
    assert g.parse_verdicts("I refuse", ["A"]) == {"A": (None, "", None)}
    assert g.parse_verdicts('```json\n{"A": {"score": 15, "reason": "x"}}\n```', ["A"])["A"][0] == 1.0


def test_follow_ups_that_need_their_conversation_are_not_graded():
    turns = [{"prompt_hash": "a", "prompt": "explain the cache", "self_contained": True},
             {"prompt_hash": "b", "prompt": "okay do it", "self_contained": False},
             {"prompt_hash": "c", "prompt": "captured live, unmarked"}]
    assert [t["prompt_hash"] for t in g.pending_turns(turns, [])] == ["a", "c"]


def test_a_prompt_whose_grading_failed_entirely_is_tried_again():
    turns = [{"prompt_hash": "a", "prompt": "x"}, {"prompt_hash": "b", "prompt": "y"}]
    failed = [{"prompt_hash": "a", "model": m, "score": None, "error": "TimeoutExpired"} for m in ("o", "s")]
    assert [t["prompt_hash"] for t in g.pending_turns(turns, failed)] == ["a", "b"]
    scored = failed + [{"prompt_hash": "a", "model": "h", "score": 0.7}]
    assert [t["prompt_hash"] for t in g.pending_turns(turns, scored)] == ["b"]


def test_grading_stops_when_the_plans_usage_limit_is_reached(tmp_path, monkeypatch):
    turns = tmp_path / "turns.jsonl"
    turns.write_text("".join(json.dumps({"prompt_hash": f"h{i}", "prompt": f"p{i}"}) + "\n" for i in range(6)))
    graded_calls = []

    def limited(turn, tiers, judge_model, *, timeout, rng):
        graded_calls.append(turn["prompt_hash"])
        return [{"prompt_hash": turn["prompt_hash"], "prompt": turn["prompt"], "model": m, "score": None,
                 "error": "RuntimeError: claude exited 1: Claude AI usage limit reached|1760000000"}
                for m in tiers]
    monkeypatch.setattr(g, "grade_turn", limited)
    g.grade(6, workers=1, turns_file=turns, out_file=tmp_path / "graded.jsonl")
    assert len(graded_calls) == 1  # stopped after the first, instead of failing all six


def test_the_limit_messages_the_cli_really_prints_are_recognised():
    for message in ("RuntimeError: claude exited 1: You've hit your session limit · resets 4am (America/Toronto)",
                    "RuntimeError: claude exited 1: Claude AI usage limit reached|1760000000",
                    "RuntimeError: claude exited 1: API Error: 429 rate limit"):
        assert g._LIMIT.search(message), message
    assert not g._LIMIT.search("RuntimeError: claude exited 1: TimeoutExpired")
