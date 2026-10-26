"""The ``criteria`` judge rubric: each answer scored on relevance, accuracy (with
the hallucinations listed), completeness, conciseness, consistency, balance and
handling of uncertainty, then an overall score on the same scale as the
standard rubric (:data:`aar_gateway.grading.grade.JUDGE_INSTRUCTIONS`).

Measured on 30 synthetic prompts (same answers, Sonnet judging): it is no
more repeatable than the standard rubric -- the cheapest adequate tier agreed
62% of the time between two criteria judgments, 57% between two standard ones,
both within noise at n=30. Averaging judgments helps more (2 vs 2: 70%, 3 vs 3:
74%). What it adds is diagnosis: per-criterion scores and the invented things
listed. Without the sandbox note (ENV_EMPTY) most "hallucinations" it flagged
were the replay environment (plan files, empty-folder searches), not the model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

#: (key, what the judge checks); balance and uncertainty may be null when the request has no such side
CRITERIA = (
    ("relevance", "Does it answer the actual request (in a conversation, the latest message), not a neighbouring "
                  "question?"),
    ("accuracy", "Are the claims, code, APIs, flags, versions, quotes and URLs real and correct? Anything invented "
                 "(a function, option, file, statute or link that doesn't exist) is a serious flaw: list each one "
                 "in \"hallucinations\"."),
    ("completeness", "Does it cover everything this request needs, with the depth a human expert would give it? Too "
                     "brief, or skipping context the developer needs, loses points. Content nobody needed earns nothing."),
    ("conciseness", "Is it as short as it can be while complete? Padding, repetition, restating the question and "
                    "unrequested sections lose points. Length the request needs is not penalised."),
    ("consistency", "Does the reasoning hold together: no contradictions between parts or between prose and code, "
                    "no conclusion stronger than its evidence, no over-generalising?"),
    ("balance", "Where the request depends on trade-offs, alternatives or counter-arguments, are they weighed fairly "
                "rather than one-sided or stereotyped, and are important caveats left out? null if the request has "
                "no such side."),
    ("uncertainty", "Where something can't be known from the request (code not shown, environment, versions), does "
                    "it say so or ask, instead of guessing with confidence? null if nothing was uncertain."),
)
OPTIONAL = ("balance", "uncertainty")

#: where the answers were produced, so the judge doesn't read the sandbox as a hallucination
ENV_EMPTY = """Each assistant answered in a read-only planning mode inside an EMPTY scratch folder: it could not
edit files or run code, and the developer's repository was not there. So an answer may say it searched and
found nothing, that a file or repository is missing, or that it wrote a plan file: those statements describe
that sandbox and are true there. Don't count them as hallucinations or errors, and don't assume anything about
the developer's environment beyond what the request says. Judge the substance: the explanation, code or plan."""
ENV_REPO = """Each assistant answered in a read-only planning mode inside the developer's repository: it could
read files but not edit them or run code, and may say it wrote a plan file -- that is how the mode works, not
a hallucination. Judge the substance: the explanation, code or plan."""

CRITERIA_INSTRUCTIONS = """You are grading answers to a developer's request to an AI coding assistant.
The answers come from different assistants and are labelled only by letter, in random order.
For a request to change code, a correct, specific plan counts as a complete answer.

{env}

Score EACH answer 0-10 on each criterion below (10 = no flaw at all), judging it against what
this request needs, not against the other answers:
{criteria}

Then give "overall" 0-10: would a careful senior engineer accept it as a correct and complete
response to this request, as written? 10 = fully correct and complete; 7 = acceptable with minor
gaps; 4 = partly right or incomplete; 0 = wrong, irrelevant or refused. Weigh accuracy and
relevance most; a hallucination that would mislead the developer caps overall at 4.
Also "matches_best": would it serve the developer as well as the best answer here (the best itself
is true)?

Reply with ONLY one JSON object, nothing else:
{{"A": {{{keys}, "hallucinations": ["<each invented thing>"], "overall": <0-10>, "matches_best": <true|false>,
"reason": "<at most 25 words>"}}, "B": {{...}}, ...}}"""

#: changes whenever the rubric's text does, so rows judged under different wording are told apart
VERSION = hashlib.sha1((CRITERIA_INSTRUCTIONS + ENV_EMPTY + ENV_REPO + json.dumps(CRITERIA)).encode()).hexdigest()[:8]
RUBRICS = ("standard", "criteria")


def criteria_prompt(request: str, answers: dict[str, str], *, empty_folder: bool, reference: str | None = None,
                    reference_note: str = "", answer_chars: int = 12_000) -> str:
    """The judge prompt; ``reference`` (a known-good fix) is shown after the request with
    ``reference_note`` explaining it, as the standard rubric does."""
    listing = "\n".join(f"- {k}: {what}" for k, what in CRITERIA)
    keys = ", ".join(f'"{k}": <0-10{" or null" if k in OPTIONAL else ""}>' for k, _ in CRITERIA)
    head = CRITERIA_INSTRUCTIONS.format(env=ENV_EMPTY if empty_folder else ENV_REPO, criteria=listing, keys=keys)
    parts = [head, f"\n\n{reference_note}" if reference else "", f"\n\n=== Developer request ===\n{request}"]
    if reference:
        parts.append(f"\n\n=== Reference solution ===\n{reference[:answer_chars]}")
    parts += [f"\n\n=== Answer {lab} ===\n{text[:answer_chars]}" for lab, text in answers.items()]
    return "".join(parts)


def _object_in(text: str) -> dict:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _unit(v) -> float | None:
    try:
        return min(max(float(v), 0.0), 10.0) / 10.0
    except (TypeError, ValueError):
        return None


def parse_criteria(text: str, labels: Iterable[str]) -> dict[str, dict]:
    """``{label: {"score", "matches_best", "reason", "criteria": {key: 0-1 or None}, "hallucinations"}}``;
    a label the reply lacks gets ``score`` None. Never invents a score."""
    data = _object_in(text)
    out = {}
    for lab in labels:
        e = data.get(lab) if isinstance(data.get(lab), dict) else {}
        mb, hall = e.get("matches_best"), e.get("hallucinations")
        out[lab] = {"score": _unit(e.get("overall")), "matches_best": mb if isinstance(mb, bool) else None,
                    "reason": str(e.get("reason", ""))[:300],
                    "criteria": {k: _unit(e.get(k)) for k, _ in CRITERIA},
                    "hallucinations": [str(h)[:200] for h in hall][:10] if isinstance(hall, list) else []}
    return out
