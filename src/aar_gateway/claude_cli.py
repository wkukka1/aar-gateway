"""Run a prompt through the real Claude Code CLI (``claude -p``), authenticated by
``claude login`` -- a Pro/Max subscription, not a metered API key.

Shells out to the CLI rather than replaying its OAuth session: that token is
scoped to the real client. ``ANTHROPIC_BASE_URL`` is dropped from the CLI's
environment so a call made here goes straight to Anthropic, never back through
this gateway.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Prompt = str | Sequence[Mapping[str, str]]


@dataclass
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    meta: dict = field(default_factory=dict)


class ClaudeSubscriptionAdapter:
    """``model`` is the CLI's own ``--model`` value (``"opus"``, a full model id, ...).

    ``complete`` honours ``cwd`` (the directory the CLI runs in, i.e. what it can
    read) and ``extra_args`` (more CLI flags, e.g. ``["--permission-mode", "plan"]``)."""

    def __init__(self, model: str, *, cli: str = "claude", timeout: float = 120.0):
        if shutil.which(cli) is None:
            raise RuntimeError(f"{cli!r} CLI not found on PATH -- run `claude login` first")
        self.model = model
        self._cli = cli
        self._timeout = timeout

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        text = prompt if isinstance(prompt, str) else _flatten(prompt)
        # On stdin, never as an argument: one starting with "-" would be parsed as CLI flags
        # (and could lift the read-only mode the grader replays untrusted prompts under).
        cmd = [self._cli, "-p", "--output-format", "json", "--model", self.model,
               *generation_kwargs.get("extra_args", ())]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_BASE_URL"}
        # utf-8 explicitly: text=True alone uses the locale code page (cp1252 on Windows)
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self._timeout,
                              cwd=generation_kwargs.get("cwd"), env=env)
        if proc.returncode != 0:
            # with --output-format json the CLI reports its error (a usage limit, an auth
            # problem) on stdout, and stderr is often empty
            detail = proc.stderr.strip()
            try:
                detail = detail or str(json.loads(proc.stdout).get("result") or "")
            except (ValueError, AttributeError):
                detail = detail or proc.stdout.strip()[:300]
            raise RuntimeError(f"{self._cli} exited {proc.returncode}: {detail}")
        payload = json.loads(proc.stdout)
        usage = payload.get("usage") or {}
        return LLMResponse(content=payload.get("result", ""), input_tokens=usage.get("input_tokens"),
                           output_tokens=usage.get("output_tokens"),
                           meta={"subscription_cost_usd_equivalent": payload.get("total_cost_usd")})


def _flatten(messages: Sequence[Mapping[str, str]]) -> str:
    """Bare text for ``-p``, which has no structured chat input -- joins role-tagged turns."""
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
