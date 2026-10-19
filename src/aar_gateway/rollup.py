"""Request-log summaries that merge.

``/stats`` needs totals over every request ever logged. With the log kept for
good (rotated into archives, never deleted) that cannot mean "load every
record": a log's summary is a small *state* built one record at a time
(:func:`accumulate`), two states of consecutive logs combine (:func:`merge`, the
older first), and :func:`finish` turns a state into the numbers the dashboard
shows. An archive's state is computed once and stored beside it.

A state is plain JSON (dicts, lists, numbers, strings) so it can be stored.
Sets are stored as dicts keyed by their members.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .pricing import prompt_tokens

TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
#: Claude Code sessions listed on the dashboard, newest first
SESSION_ROWS = 30


def _blank() -> dict:
    return {"requests": 0, "errors": 0, "rerouted": 0, "fallbacks": 0, **dict.fromkeys(TOKEN_KEYS, 0),
            "cost_usd": 0.0}


def new_state() -> dict:
    return {
        "total": {**_blank(), "cost_if_requested_usd": 0.0, "cost_if_strongest_usd": 0.0, "priced_requests": 0,
                  "turns_routed": 0, "quality_sum": 0, "best_quality_sum": 0, "subagent_requests": 0},
        "models": {}, "sessions": {}, "subagents": {}, "strongest": None,
    }


def expected_thread_expiry(record: Mapping) -> bool:
    """A 404 for a server-side thread the API has let go of: Claude Code
    replays the conversation itself, so it isn't a failure of the gateway."""
    return record.get("status") == 404 and "No thread state" in (record.get("error") or "")


def baseline_cost(r: Mapping, key: str) -> float:
    """A record's cost on a baseline model (``cost_if_requested_usd`` ...), or,
    when the baseline had no price, what the request really cost: unpriced work
    shows no saving rather than a negative one. (The totals instead skip the
    request, so their saving is only over fully priced traffic.)"""
    return r[key] if r.get(key) is not None else (r.get("cost_usd") or 0.0)


def accumulate(state: dict, r: Mapping) -> None:
    """Fold one logged request into ``state`` (requests in log order)."""
    total = state["total"]
    model = r.get("routed_model") or "unknown"
    row = state["models"].setdefault(model, _blank())
    for part in (total, row):
        part["requests"] += 1
        part["errors"] += r.get("status") != 200 and not expected_thread_expiry(r)
        part["rerouted"] += r.get("routed_model") != r.get("requested_model")
        part["fallbacks"] += bool(r.get("fallback"))
        for k in TOKEN_KEYS:
            part[k] += r.get(k) or 0
        part["cost_usd"] += r.get("cost_usd") or 0.0
    if None not in (r.get("cost_usd"), r.get("cost_if_requested_usd"), r.get("cost_if_strongest_usd")):
        total["priced_requests"] += 1
        total["cost_if_requested_usd"] += r["cost_if_requested_usd"]
        total["cost_if_strongest_usd"] += r["cost_if_strongest_usd"]
    if r.get("predicted_quality") is not None:
        total["turns_routed"] += 1
        total["quality_sum"] += r["predicted_quality"]
        total["best_quality_sum"] += r.get("best_predicted_quality") or r["predicted_quality"]
    if r.get("subagent"):
        total["subagent_requests"] += 1
    if r.get("strongest_model"):
        state["strongest"] = r["strongest_model"]

    sid = r.get("claude_session")
    sess = state["sessions"].setdefault(sid or "", {
        "session": sid, "first_ts": r.get("ts"), "last_ts": r.get("ts"), "requests": 0, "turns": 0,
        "subagent_spawns": {}, "rerouted": 0, "models": {}, "cost_usd": 0.0,
        "cost_if_requested_usd": 0.0, "cost_if_strongest_usd": 0.0})
    sess["last_ts"] = r.get("ts") or sess["last_ts"]
    sess["requests"] += 1
    sess["turns"] += bool(r.get("new_turn") and not r.get("subagent")
                          and not (r.get("reason") or "").startswith("side request"))
    if r.get("subagent"):
        sess["subagent_spawns"][r.get("conversation") or ""] = 1
    sess["rerouted"] += r.get("routed_model") != r.get("requested_model")
    sess["models"][model] = sess["models"].get(model, 0) + 1
    sess["cost_usd"] += r.get("cost_usd") or 0.0
    sess["cost_if_requested_usd"] += baseline_cost(r, "cost_if_requested_usd")
    sess["cost_if_strongest_usd"] += baseline_cost(r, "cost_if_strongest_usd")

    if r.get("subagent"):
        role = r.get("agent_role") or "unrecognised"
        agent = state["subagents"].setdefault(role, {
            "role": role, "source": r.get("role_source"), "conversations": {}, "requests": 0, "models": {},
            "cost_usd": 0.0, "cost_if_requested_usd": 0.0, "cache_read_input_tokens": 0, "prompt_tokens": 0,
            "latency_s": 0.0, "report_sum": 0, "report_n": 0})
        agent["conversations"][r.get("conversation") or ""] = 1
        agent["requests"] += 1
        agent["models"][model] = agent["models"].get(model, 0) + 1
        agent["cost_usd"] += r.get("cost_usd") or 0.0
        agent["cost_if_requested_usd"] += baseline_cost(r, "cost_if_requested_usd")
        agent["cache_read_input_tokens"] += r.get("cache_read_input_tokens") or 0
        agent["prompt_tokens"] += prompt_tokens(r)
        agent["latency_s"] += r.get("latency_s") or 0.0
        if r.get("report_tokens") is not None:
            agent["report_sum"] += r["report_tokens"]
            agent["report_n"] += 1


def _add_counts(into: dict, other: Mapping) -> None:
    for key, n in other.items():
        into[key] = into.get(key, 0) + n


def _merge_row(into: dict, other: Mapping, *, summed: Iterable[str]) -> None:
    for key in summed:
        into[key] += other[key]


def merge(older: Mapping, newer: Mapping) -> dict:
    """The state of ``older``'s requests followed by ``newer``'s."""
    out = _copy(older)
    for key in out["total"]:
        out["total"][key] += newer["total"][key]
    for name, row in newer["models"].items():
        mine = out["models"].setdefault(name, _blank())
        _merge_row(mine, row, summed=mine.keys())
    for sid, row in newer["sessions"].items():
        mine = out["sessions"].get(sid)
        if mine is None:
            out["sessions"][sid] = _copy(row)
            continue
        mine["last_ts"] = row["last_ts"] or mine["last_ts"]
        _merge_row(mine, row, summed=("requests", "turns", "rerouted", "cost_usd", "cost_if_requested_usd",
                                      "cost_if_strongest_usd"))
        mine["subagent_spawns"].update(row["subagent_spawns"])
        _add_counts(mine["models"], row["models"])
    for role, row in newer["subagents"].items():
        mine = out["subagents"].get(role)
        if mine is None:
            out["subagents"][role] = _copy(row)
            continue
        _merge_row(mine, row, summed=("requests", "cost_usd", "cost_if_requested_usd", "cache_read_input_tokens",
                                      "prompt_tokens", "latency_s", "report_sum", "report_n"))
        mine["conversations"].update(row["conversations"])
        _add_counts(mine["models"], row["models"])
    out["strongest"] = newer["strongest"] or older["strongest"]
    return out


def _copy(value):
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


def finish(state: Mapping, *, recent: list, log: str, default_strongest: str | None) -> dict:
    """The dashboard's numbers for ``state`` (``recent``: the newest requests, oldest first)."""
    t = dict(state["total"])
    n = t.pop("turns_routed")
    quality, best = t.pop("quality_sum"), t.pop("best_quality_sum")
    subagent_requests = t.pop("subagent_requests")
    total = {**t, "turns_routed": n, "predicted_quality": quality / n if n else None,
             "subagent_requests": subagent_requests, "best_predicted_quality": best / n if n else None}
    sessions = [{**_copy(row), "subagent_spawns": len(row["subagent_spawns"])} for row in state["sessions"].values()]
    sessions = sorted(sessions, key=lambda x: x["last_ts"] or "", reverse=True)[:SESSION_ROWS]
    agents = []
    for row in state["subagents"].values():
        prompt = row["prompt_tokens"]
        shown = {k: _copy(v) for k, v in row.items() if k not in ("conversations", "report_sum", "report_n")}
        agents.append({**shown,
                       "avg_report_tokens": round(row["report_sum"] / row["report_n"]) if row["report_n"] else None,
                       "spawns": len(row["conversations"]),
                       "cache_hit_rate": row["cache_read_input_tokens"] / prompt if prompt else None})
    return {"log": log, "strongest_model": state["strongest"] or default_strongest, "totals": total,
            "models": {k: dict(v) for k, v in state["models"].items()},
            "subagents": sorted(agents, key=lambda x: -x["cost_usd"]), "sessions": sessions,
            "recent": list(reversed(recent))}
