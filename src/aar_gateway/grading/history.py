"""The user's past Claude Code messages, from Claude Code's own local session
transcripts (``~/.claude/projects/*/*.jsonl``) -- grading material from day
one, instead of waiting for the gateway to capture new turns.

Only what the person typed: transcript lines marked ``origin.kind ==
"human"``, outside subagent side-chains. Dropped: tool results, Claude Code's
``<system-reminder>`` text, slash commands and local-command output, task
notifications, sessions run in temporary folders (scripts, the grader's own
replays), duplicates (a continued session repeats its history), and replies
too short to grade ("continue", "yes", "#").
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from aar_gateway.config import append_jsonl, iter_jsonl, read_jsonl
from aar_gateway.translate import REMINDER, text_of

PROJECTS_DIR = Path.home() / ".claude" / "projects"
MIN_CHARS = 20
_TEMP_PROJECTS = ("-private-tmp", "-private-var-folders", "-tmp-")


def _text(content) -> str:
    return text_of(content, sep="\n")


def _typed(record: dict) -> str | None:
    if record.get("type") != "user" or record.get("isSidechain") or record.get("isMeta"):
        return None
    if (record.get("origin") or {}).get("kind") != "human":
        return None
    text = REMINDER.sub("", _text((record.get("message") or {}).get("content"))).strip()
    if len(text) < MIN_CHARS or text.startswith(("/", "<command-", "<local-command", "<task-notification")):
        return None
    return text


def history_turns(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    """Rows in the gateway's ``turns.jsonl`` shape, oldest first, one per
    distinct message."""
    out: dict[str, dict] = {}
    for transcript in sorted(projects_dir.glob("*/*.jsonl")):
        if transcript.parent.name.startswith(_TEMP_PROJECTS):
            continue
        for record in iter_jsonl(transcript):
            text = _typed(record)
            if text is None:
                continue
            h = hashlib.sha1(text.encode()).hexdigest()[:16]
            out.setdefault(h, {"ts": record.get("timestamp"), "prompt_hash": h, "prompt": text,
                               "cwd": record.get("cwd"), "source": "history"})
    return sorted(out.values(), key=lambda r: r.get("ts") or "")


def merge_into(turns_file: Path, rows: Iterable[dict]) -> int:
    """Append the rows ``turns_file`` doesn't already have; returns how many."""
    have = {r["prompt_hash"] for r in read_jsonl(turns_file) if "prompt_hash" in r}
    new = [r for r in rows if r["prompt_hash"] not in have]
    append_jsonl(turns_file, *new)
    return len(new)


SELF_CONTAINED_PROMPT = """Below are numbered messages a developer typed to an AI coding assistant, each
taken out of its conversation. For each, decide: could a capable assistant understand what is being
asked and give a useful answer from this message alone (plus read access to the developer's
repository), without the earlier conversation? Follow-ups like "do it", "okay test it out", "yes",
or ones that refer to something only said earlier ("that", "the second option", "what you
proposed") are NOT self-contained.

Reply with ONLY a JSON array of the numbers of the self-contained messages, e.g. [1, 4, 7]."""


def mark_self_contained(rows: list[dict], model: str = "claude-haiku-5-5", batch: int = 40) -> list[dict]:
    """Adds ``self_contained`` (bool) to each row, asking a cheap model in
    batches -- grading a message nobody could answer out of context would
    only add noise. A batch whose reply can't be parsed is left unmarked."""
    from aar_gateway.claude_cli import ClaudeSubscriptionAdapter

    llm = ClaudeSubscriptionAdapter(model, timeout=600)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        listing = "\n\n".join(f"[{i + 1}] {r['prompt'][:1500]}" for i, r in enumerate(chunk))
        reply = llm.complete(f"{SELF_CONTAINED_PROMPT}\n\n{listing}",
                             extra_args=("--permission-mode", "plan", "--strict-mcp-config", "--safe-mode")).content
        found = re.search(r"\[[\d,\s]*\]", reply or "")
        if not found:
            continue
        keep = set(json.loads(found.group(0)))
        for i, r in enumerate(chunk):
            r["self_contained"] = (i + 1) in keep
    return rows
