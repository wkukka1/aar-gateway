"""Claude Code subagents: what kind of work each one does, and its model.

**Where the model is set.** When the main agent calls the Agent tool, Claude
Code runs its ``PreToolUse`` hooks first, with the call's ``subagent_type``
and ``prompt``; a hook may return the call with ``model`` set, and the
subagent then runs on that model (verified on Claude Code 2.1.296: the
subagent ran on Haiku 5.5 while the main agent stayed on Sonnet 5.5).
``aar-gateway code`` installs that hook (``aar-gateway hook``), which asks
the gateway for :func:`spawn_decision`. This is documented behaviour, so it
does not break when Claude Code rewords a prompt.

**At the gateway, a fallback.** A subagent's requests reach the gateway as a
conversation of their own. Its system prompt's billing line carries
``cc_is_subagent=true`` (undocumented, seen on 2.1.293-2.1.296), so the
gateway can still recognise one the hook didn't see -- without the hook, or a
spawn the hook could not reach the gateway for -- and route it by
:func:`role_decision`.

**Kinds.** Three, by what the subagent may do, each with a starting tier:

* ``lookup`` -- read-only, short answer: search, locate, count, extract,
  label, summarise. Haiku.
* ``make`` -- produce something: code, tests, a review, a plan. Sonnet.
* ``judge`` -- hard reasoning: root causes, design trade-offs. The
  session's own model.

A subagent's kind, first match wins: one of :data:`ROLES` (its
``subagent_type``, or ``aar-role: <kind>`` in its prompt); Claude Code's
built-in Explore (a lookup) or Plan (Sonnet, like make); else keywords at the
start of its task (:func:`classify_task`). Otherwise it has no kind and the
general router decides, as for any turn. A tier is a ceiling on cost, never
an upgrade: it never moves a subagent above the session's model.

A subagent is the cheapest place to change model: its system prompt and
tools differ from the main agent's, so it has a prompt cache of its own, and
moving it never makes the main conversation rewrite its cache.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .pricing import canonical_model, has_price
from .routing import price_for, unavailable_reason
from .translate import text_of

SUBAGENT_MARK = "cc_is_subagent=true"
_ROLE_MARK = re.compile(r"\baar-role:\s*([a-z-]+)")
#: Claude Code's built-in subagents: their ``subagent_type``, and the start of
#: their system prompt (read off Claude Code 2.1.293-2.1.296's requests)
_BUILTINS = (("explore", re.compile(r"You are a file search specialist")),
             ("plan", re.compile(r"You are a software architect and planning specialist")))

_HAIKU, _SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
#: role -> model. ``None`` keeps the session's model.
DEFAULT_ROLE_MODELS: dict[str, str | None] = {
    "lookup": _HAIKU, "explore": _HAIKU,
    "make": _SONNET, "plan": _SONNET,
    "judge": None,
}

#: Task keywords, in priority order: a task that asks for a root cause and
#: also says "find" is a judge's, and one that opens by asking for something
#: to be made ("Write 100 prompts ... of the type SUMMARIZE") is a make, whatever
#: it is about. Only the start of the task is read -- that is where the main
#: agent states what it wants.
_TASK_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("judge", re.compile(r"\broot[- ]cause|\bdebug|\barchitectur|\brace condition|\bdeadlock|\btrade-?offs?\b")),
    ("make", re.compile(r"^(?:please\s+)?(?:write|generate|create|compose|produce|build|implement|add|fix|refactor)\b")),
    ("make", re.compile(r"\breview|\baudit\b|\bcritique|\bfind (?:bugs|issues|problems)|\b(?:write|add) (?:\w+ )?tests?\b"
                        r"|\bplan (?:for|how|the)\b|\bdesign (?:a|an|the)\b|\bpropose\b|\boutline\b")),
    ("lookup", re.compile(r"\bfind\b|\blocate\b|\bsearch\b|\bwhere (?:is|are|does|do)\b|\blist (?:all|every)\b"
                          r"|\bgrep\b|\bcount\b|\bwhich files?\b|\bexplore\b|\blabel|\bclassify|\bextract"
                          r"|\bsummari[sz]e|\bcommit message|\bchangelog")),
)
TASK_CHARS = 400


def classify_task(task: str) -> tuple[str, str] | None:
    """``(kind, the words that decided it)`` for a subagent's task, or ``None``."""
    head = (task or "")[:TASK_CHARS].lower().lstrip()
    for kind, pattern in _TASK_RULES:
        found = pattern.search(head)
        if found:
            return kind, found.group(0).strip()
    return None


def _system_text(body: Mapping) -> str:
    return text_of(body.get("system"), sep="\n", typed=False)


def identify(subagent_type: str | None, task: str | None, system: str = "") -> dict:
    """``{"role", "source", "why"}`` for a subagent (``role`` ``None`` when
    nothing here recognises it)."""
    kind = (subagent_type or "").lower()
    if kind in ROLES:
        return {"role": kind, "source": "definition", "why": subagent_type}
    found = _ROLE_MARK.search(system)
    if found and found.group(1) in ROLES:
        return {"role": found.group(1), "source": "definition", "why": found.group(0)}
    for role, pattern in _BUILTINS:
        if kind == role or pattern.search(system):
            return {"role": role, "source": "builtin", "why": subagent_type or pattern.pattern}
    guessed = classify_task(task or "")
    if guessed:
        return {"role": guessed[0], "source": "task", "why": guessed[1]}
    return {"role": None, "source": None, "why": None}


def detect(body: Mapping, task: str | None) -> dict | None:
    """``None`` for anything but a subagent's request; else :func:`identify`."""
    system = _system_text(body)
    if SUBAGENT_MARK not in system:
        return None
    return identify(None, task, system)


def role_model(role: str | None, settings: Mapping) -> tuple[bool, str | None]:
    """``(known, model)`` for a role: ``model`` ``None`` keeps the session's."""
    models = {**DEFAULT_ROLE_MODELS, **((settings.get("subagents") or {}).get("roles") or {})}
    if role is None or role not in models:
        return False, None
    return True, canonical_model(models[role]) if models[role] else None


def _pricier(settings: Mapping, model: str, than: str | None) -> bool:
    mine, other = price_for(settings, model), price_for(settings, than)
    return has_price(mine) and has_price(other) and mine["output"] > other["output"]


def role_decision(agent: Mapping, body: Mapping, requested: str | None, settings: Mapping,
                  blocked: Mapping[str, str], est_prompt_tokens: int) -> dict | None:
    """At the gateway: the model for a subagent's turn from its kind, or
    ``None`` to leave it to the general router (no kind, or the kind's model
    can't take this request)."""
    known, target = role_model(agent.get("role"), settings)
    if not known:
        return None
    label = f"{agent['role']} subagent ({agent['source']}: {agent['why']})"
    if target is None:
        return {"model": requested, "reason": f"{label}: keeps the session's model"}
    if unavailable_reason(target, body, settings, blocked, est_prompt_tokens):
        return None  # e.g. too long for it, or an openai-lane model it can't be translated for
    if not has_price(price_for(settings, target)) or not has_price(price_for(settings, requested)):
        return None
    if _pricier(settings, target, requested):
        return {"model": requested, "reason": f"{label}: {target} would cost more than the session's model"}
    return {"model": target, "reason": f"{label} -> {target}"}


#: what the Agent tool's ``model`` parameter accepts, by model family
_ALIASES = ("haiku", "sonnet", "opus")


def spawn_decision(subagent_type: str | None, task: str | None, session_model: str | None,
                   settings: Mapping, asked_model: str | None = None) -> dict:
    """At spawn time (the hook): ``{"role", "source", "why", "model", "alias",
    "reason"}``. ``alias`` is what to set the Agent call's ``model`` to, or
    ``None`` to leave the call alone -- no kind, the main agent already chose a
    model, the kind keeps the session's model, or its model would cost more
    than the session's."""
    agent = identify(subagent_type, task)
    out = {**agent, "model": None, "alias": None}
    if asked_model:
        return {**out, "reason": f"the main agent asked for {asked_model}"}
    known, target = role_model(agent["role"], settings)
    if not known:
        return {**out, "reason": "no kind recognised: the gateway's router decides"}
    if target is None:
        return {**out, "reason": f"{agent['role']}: keeps the session's model"}
    if session_model and _pricier(settings, target, session_model):
        return {**out, "reason": f"{agent['role']}: {target} would cost more than the session's model"}
    alias = next((a for a in _ALIASES if a in target), None)
    if alias is None:
        return {**out, "model": target, "reason": f"{agent['role']}: {target} has no Agent-tool alias; "
                                                 "the gateway routes it"}
    return {**out, "model": target, "alias": alias, "reason": f"{agent['role']} -> {target}"}


#: The kinds ``aar-gateway code`` gives Claude Code (``--agents``). Each
#: prompt ends with an output budget -- a subagent's whole answer lands in the
#: main agent's context and is re-sent with every request after it -- and
#: lookup and judge are limited to tools that read and run, not edit.
#: ``model`` pins the tier in Claude Code itself, so the kinds keep their
#: tiers without the gateway too.
_BUDGET = ("Your final message is all the main agent sees, and it carries it for the rest of the "
           "conversation: keep it within the budget below. No preamble, no restating the task, no "
           "closing summary.")
_READ_TOOLS = ["Read", "Grep", "Glob", "Bash", "ToolSearch", "WebFetch", "WebSearch"]

ROLES: dict[str, dict] = {
    "lookup": {
        "model": "haiku",
        "tools": _READ_TOOLS,
        "description": "Read-only lookups with a short answer: find where something is defined or used, "
                       "which files do X, count things, extract or label items, summarise a log, draft a "
                       "commit message. Use for any search or mechanical, checkable task.",
        "prompt": "You find facts in the repository and do mechanical, checkable work. Never change files; "
                  "if the task needs a file written, say so instead.\n\n" + _BUDGET +
                  "\n\nOutput budget: at most 10 bullets, each `path:line` and one sentence, or exactly "
                  "the structured result asked for. If nothing matches, say so in one line.",
    },
    "make": {
        "model": "sonnet",
        "description": "Produces something: implements or changes code, writes and runs tests, reviews "
                       "a diff, or turns a goal into an implementation plan.",
        "prompt": "You make the change, tests, review or plan asked for, matching the project's existing "
                  "style.\n\n" + _BUDGET + "\n\nOutput budget: what you changed or concluded, as at most "
                  "10 one-line items (files touched, tests passed/failed, or findings as "
                  "`severity · path:line · problem · fix`). No code in the reply unless asked.",
    },
    "judge": {
        "model": "inherit",
        "tools": _READ_TOOLS,
        "description": "Hard reasoning that needs the strongest model: root-cause debugging, architecture "
                       "and design trade-offs. Investigates; does not edit.",
        "prompt": "You investigate hard problems: find root causes and weigh designs, with evidence from "
                  "the code. Never change files.\n\n" + _BUDGET + "\n\nOutput budget: conclusion first, "
                  "then at most 8 findings with `path:line` evidence, then the recommended next step.",
    },
}


def agent_definitions() -> dict[str, dict]:
    """``ROLES`` as Claude Code's ``--agents`` JSON, each prompt marked with
    its kind so the gateway can recognise its requests."""
    out = {}
    for name, r in ROLES.items():
        d = {"description": r["description"], "model": r["model"], "prompt": f"{r['prompt']}\n\naar-role: {name}"}
        if r.get("tools"):
            d["tools"] = list(r["tools"])
        out[name] = d
    return out


def agent_markdown(name: str) -> str:
    """One kind as a Claude Code agent file (``~/.claude/agents/<name>.md``)."""
    d = agent_definitions()[name]
    tools = f"tools: {', '.join(d['tools'])}\n" if d.get("tools") else ""
    return (f"---\nname: {name}\ndescription: {d['description']}\nmodel: {d['model']}\n{tools}---\n\n"
            f"{d['prompt']}\n")
