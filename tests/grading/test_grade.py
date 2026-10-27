"""The grader with a fake `claude` CLI: every tier is replayed read-only in
the turn's folder, the judge sees anonymised answers, and grades land per
(turn, model) -- never twice for the same prompt."""

from __future__ import annotations

import json
import random

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
        # a bare tier word, not the model name: blind() would take a name out before the judge sees it
        return LLMResponse(content=f"answer from {self.model.split('-')[1]}",
                           meta={"subscription_cost_usd_equivalent": 0.01})


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


@pytest.mark.parametrize("answer, request_, want", [
    ('git commit -m "fix" -m "Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>"', "commit it",
     'git commit -m "fix" -m "Co-Authored-By: Claude <noreply@anthropic.com>"'),
    ("🤖 Generated with [Claude Code](https://claude.com/claude-code)", "open a PR",
     "🤖 Generated with [Claude](https://claude.com/claude-code)"),
    ("I'm Claude Opus 5.5, and here is the fix.", "fix it", "I'm Claude, and here is the fix."),
    ("Route it to claude-sonnet-5-5.", "what model?", "Route it to Claude."),
    ("Use a haiku example; an Opus judge scores it.", "pick examples",   # bare tier words are content
     "Use a haiku example; an Opus judge scores it."),
    ("Set it to claude-opus-5-5.", "should the default be claude-opus-5-5?",  # the request's own topic
     "Set it to claude-opus-5-5."),
    ("Co-Authored-By: Claude Opus 5.5 <x>", "the trailer says Claude Opus 5.5",  # a signature is never topic
     "Co-Authored-By: Claude <x>"),
])
def test_blind_takes_the_authors_name_out(answer, request_, want):
    assert g.blind(answer, request_) == want


def test_both_judge_prompts_are_blind():
    from aar_gateway.grading.rubric import criteria_prompt

    signed = {"A": "done.\nCo-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>"}
    assert "Haiku" not in g.judge_prompt("commit it", signed)
    assert "Haiku" not in criteria_prompt("commit it", signed, empty_folder=True)


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

    def limited(turn, tiers, judge_model, *, timeout, rng, **_kw):
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


def test_criteria_rubric_rows_carry_per_criterion_scores_and_the_sandbox_note(monkeypatch):
    from aar_gateway.grading import rubric

    prompts = []

    class Judge:
        def __init__(self, model, timeout=None):
            self.model = model

        def complete(self, prompt, **kw):
            prompts.append(prompt)
            if not prompt.startswith("You are grading"):
                return LLMResponse(content=f"answer from {self.model}", meta={"subscription_cost_usd_equivalent": 0.1})
            labels = [lab for lab in "ABC" if f"=== Answer {lab} ===" in prompt]
            body = {lab: {"relevance": 9, "accuracy": 3, "completeness": 8, "conciseness": 7, "consistency": 8,
                          "balance": None, "uncertainty": None, "hallucinations": ["made-up --flag"],
                          "overall": 4, "matches_best": False, "reason": "invented flag"} for lab in labels}
            return LLMResponse(content=json.dumps(body))

    monkeypatch.setattr(g, "ClaudeSubscriptionAdapter", Judge)
    rows = g.grade_turn({"prompt_hash": "h", "prompt": "fix it"}, ("claude-haiku-5-5", "claude-opus-5-5"),
                        "claude-sonnet-5-5", timeout=5, rng=random.Random(0), rubric="criteria")
    assert {r["score"] for r in rows} == {0.4} and rows[0]["criteria"]["accuracy"] == 0.3
    assert rows[0]["hallucinations"] == ["made-up --flag"] and rows[0]["rubric"] == f"criteria-{rubric.VERSION}"
    assert "EMPTY scratch folder" in prompts[-1]  # no cwd: the replay ran in an empty folder
    std = g.grade_turn({"prompt_hash": "h", "prompt": "fix it"}, ("claude-haiku-5-5",), "claude-sonnet-5-5",
                       timeout=5, rng=random.Random(0))
    assert std[0]["rubric"] == "standard" and "criteria" not in std[0]


def test_two_judgments_are_averaged_each_on_its_own_shuffle(monkeypatch):
    seen_orders, scores = [], iter([{"answer from haiku": 2, "answer from opus": 8},
                                    {"answer from haiku": 4, "answer from opus": 9}])

    class Judge:
        def __init__(self, model, timeout=None):
            self.model = model

        def complete(self, prompt, **kw):
            if not prompt.startswith("You are grading"):
                return LLMResponse(content=f"answer from {self.model.split('-')[1]}", meta={"subscription_cost_usd_equivalent": 0.1})
            import re as _re
            shown = dict(_re.findall(r"=== Answer (\w) ===\n(answer from [\w-]+)", prompt))
            seen_orders.append(tuple(shown.values()))
            want = next(scores)
            body = {lab: {"score": want[text], "matches_best": want[text] > 5, "reason": "r"} for lab, text in shown.items()}
            return LLMResponse(content=json.dumps(body), meta={"subscription_cost_usd_equivalent": 0.05})

    monkeypatch.setattr(g, "ClaudeSubscriptionAdapter", Judge)
    rows = g.grade_turn({"prompt_hash": "h", "prompt": "fix it"}, ("claude-haiku-5-5", "claude-opus-5-5"),
                        "claude-sonnet-5-5", timeout=5, rng=random.Random(1), judgments=2)
    by = {r["model"]: r for r in rows}
    assert by["claude-haiku-5-5"]["score"] == pytest.approx(0.3) and by["claude-haiku-5-5"]["judge_scores"] == [0.2, 0.4]
    assert by["claude-opus-5-5"]["matches_best"] is True and by["claude-haiku-5-5"]["matches_best"] is False
    assert by["claude-opus-5-5"]["judge_usage_usd"] == pytest.approx(0.1)
    assert len(seen_orders) == 2


def test_merge_verdicts_tie_counts_as_matching_and_hallucinations_are_unioned():
    m = g._merge_verdicts([
        {"score": 0.6, "reason": "a", "matches_best": True, "criteria": {"accuracy": 0.5, "balance": None},
         "hallucinations": ["x"]},
        {"score": 0.8, "reason": "b", "matches_best": False, "criteria": {"accuracy": 0.7, "balance": None},
         "hallucinations": ["x", "y"]}])
    assert m["score"] == pytest.approx(0.7) and m["matches_best"] is True
    assert m["criteria"] == {"accuracy": pytest.approx(0.6), "balance": None} and m["hallucinations"] == ["x", "y"]
    assert g._merge_verdicts([{"score": None, "reason": "", "matches_best": None}])["score"] is None
