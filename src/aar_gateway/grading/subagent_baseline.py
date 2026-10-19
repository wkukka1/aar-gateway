"""What the gateway's subagent roles would have saved on subagents already
run: Claude Code's own transcripts of them
(``~/.claude/projects/*/<session>/subagents/agent-*.jsonl`` with a
``.meta.json`` naming the agent type).

    aar-gateway-subagents

Each subagent gets the role the gateway would give it (its agent type if it
is one of the roles, Explore as a built-in, else its task's keywords) and is
priced, message by message from the recorded token usage, on the model it
ran on and on the model its role would pick. Usage only, not quality: the
same tokens are assumed on the cheaper model, and whether its answers would
have been as good is what grading (``aar-gateway-grade``) measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from aar_gateway.config import iter_jsonl, load_settings
from aar_gateway.pricing import cost_usd
from aar_gateway.routing import price_for
from aar_gateway.subagents import identify, role_decision
from aar_gateway.translate import text_of

from .history import PROJECTS_DIR


def _first_task(records: list[dict]) -> str:
    for r in records:
        if r.get("type") == "user":
            return text_of((r.get("message") or {}).get("content"), sep=" ", typed=False)
    return ""


def subagent_runs(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    """One row per recorded subagent: type, task, and each assistant
    message's model and usage."""
    runs = []
    for path in sorted(projects_dir.glob("*/*/subagents/*.jsonl")):
        meta_path = path.with_name(path.stem + ".meta.json")
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            records = list(iter_jsonl(path))  # a corrupt line is skipped, not the whole run
        except (OSError, ValueError):
            continue
        messages = [(m.get("model"), m["usage"]) for r in records if r.get("type") == "assistant"
                    for m in [r.get("message") or {}] if m.get("usage")]
        if messages:
            cwd = next((r.get("cwd") for r in records if r.get("cwd")), None)
            runs.append({"path": str(path), "agent_type": meta.get("agentType"), "cwd": cwd,
                         "description": meta.get("description"), "task": _first_task(records),
                         "messages": messages})
    return runs


_ABS_PATH = re.compile(r"(/(?:private|Users|tmp|var|home)/[^\s'\"`)]+)")


def replayable(run: dict) -> bool:
    """Its folder still exists and so does every absolute path its task
    names -- a task whose input files were scratch files can't be redone."""
    if not (run.get("cwd") and os.path.isdir(run["cwd"])):
        return False
    return all(os.path.exists(p.rstrip(".,:;")) for p in _ABS_PATH.findall(run["task"]))


def subagent_turns(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    """Replayable subagent tasks in the shape of the gateway's
    ``turns.jsonl``, for ``aar-gateway-grade --from-subagents``."""
    out = {}
    for run in subagent_runs(projects_dir):
        if not run["task"].strip() or not replayable(run):
            continue
        h = hashlib.sha1(run["task"].encode()).hexdigest()[:16]
        out.setdefault(h, {"prompt_hash": h, "prompt": run["task"], "cwd": run["cwd"], "source": "subagent",
                           "subagent": True, "agent_role": role_of(run)["role"],
                           "ran_on": run["messages"][0][0]})
    return list(out.values())


def role_of(run: dict) -> dict:
    """The kind the gateway would give it -- the same rules, from its agent
    type and task."""
    return identify(run.get("agent_type"), run["task"])


def price_run(run: dict, settings: dict) -> dict:
    agent = role_of(run)
    ran_on = run["messages"][0][0]
    decision = role_decision(agent, {}, ran_on, settings, {}, 0) if agent["role"] else None
    tiered_model = (decision or {}).get("model") or ran_on
    actual = tiered = 0.0
    for model, usage in run["messages"]:
        actual += cost_usd(price_for(settings, model), usage) or 0.0
        tiered += cost_usd(price_for(settings, tiered_model if model == ran_on else model), usage) or 0.0
    return {**agent, "ran_on": ran_on, "tiered_model": tiered_model, "cost_usd": actual,
            "tiered_cost_usd": tiered, "description": run.get("description")}


def report(rows: list[dict]) -> str:
    by: dict[str, dict] = {}
    for r in rows:
        key = r["role"] or "(no role: general router)"
        row = by.setdefault(key, {"n": 0, "actual": 0.0, "tiered": 0.0, "to": set()})
        row["n"] += 1
        row["actual"] += r["cost_usd"]
        row["tiered"] += r["tiered_cost_usd"]
        row["to"].add(r["tiered_model"])
    lines = [f"{'role':28s} {'spawns':>6s} {'as run $':>9s} {'by role $':>9s} {'saved':>6s}  model by role"]
    for key, row in sorted(by.items(), key=lambda kv: -kv[1]["actual"]):
        saved = 1 - row["tiered"] / row["actual"] if row["actual"] else 0.0
        lines.append(f"{key:28s} {row['n']:6d} {row['actual']:9.2f} {row['tiered']:9.2f} {saved:6.0%}  "
                     + ", ".join(sorted(row["to"])))
    actual = sum(r["cost_usd"] for r in rows)
    tiered = sum(r["tiered_cost_usd"] for r in rows)
    lines.append(f"{'total':28s} {len(rows):6d} {actual:9.2f} {tiered:9.2f} "
                 f"{(1 - tiered / actual) if actual else 0.0:6.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aar-gateway-subagents", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="gateway config (models, prices, subagent roles)")
    ap.add_argument("--projects", default=str(PROJECTS_DIR))
    ap.add_argument("--json", action="store_true", help="one JSON row per subagent instead of the table")
    args = ap.parse_args(argv)
    settings = load_settings(args.config)
    rows = [price_run(run, settings) for run in subagent_runs(Path(args.projects))]
    if not rows:
        print("no subagent transcripts found", file=sys.stderr)
        return 1
    print("\n".join(json.dumps(r) for r in rows) if args.json else report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
