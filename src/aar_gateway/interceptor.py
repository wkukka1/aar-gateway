"""The interceptor: Claude Code's requests in, routed, and out to the right model.

Claude Code pointed at this gateway (``ANTHROPIC_BASE_URL``) sends its real
requests here -- its own login, every tool definition, thinking settings,
``stream: true``. For each ``/v1/messages`` request:

1. **Decide the model, once per human turn.** A request whose last user
   message carries new human text starts a turn: the picker
   (:mod:`aar_gateway.routing`) may move it to another model. Requests
   inside the turn's tool loop (tool results only) stay on the turn's model,
   so the loop never changes model halfway through. A Claude Code subagent
   is a conversation of its own whose first message is its task; when its
   role is known (:mod:`aar_gateway.subagents`) the role picks its tier.
2. **Send it down the model's lane.**
   ``anthropic``: forwarded to Anthropic close to untouched -- client's own
   credentials, tools, thinking, beta headers; only ``model`` may change.
   ``openai``: translated to Chat Completions with tools
   (:mod:`aar_gateway.translate`), sent with that provider's key from the
   environment, translated back -- streaming included.
3. **Fail open.** If a rerouted request errors, it is retried on the model
   the client asked for; the router never leaves Claude Code worse off.
4. **Log it.** One JSON line per request: models, tokens, latency, observed
   cost, and two counterfactuals -- the cost on the model the client asked
   for, and on the strongest configured model -- with the token counts
   adjusted for what those models would have had cached. Never headers,
   credentials or prompt text.

These records are a stand-in for :class:`~router.tracing.traces.ExecutionTrace`,
which needs a full ``RoutingDecision``/``PromptSignals`` per request.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import archive, history, rollup, share
from .config import (
    DEFAULT_ROUTER,
    DEFAULT_SUBAGENTS,
    append_jsonl,
    atomic_write_text,
    iter_jsonl,
    load_settings,
    log_archives,
    log_path,
    overrides_path,
    read_json,
    read_jsonl,
    read_preferences,
    safe_append_jsonl,
    save_preferences,
)
from .config import thread_models_path as _thread_models_path
from .pricing import canonical_model, cost_usd, prompt_tokens
from .routing import ModelPicker, price_for, unavailable_reason
from .subagents import DEFAULT_ROLE_MODELS, role_decision, spawn_decision
from .subagents import detect as detect_subagent
from .translate import (
    REMINDER,
    OpenAIStreamTranslator,
    anthropic_to_openai,
    content_blocks,
    error_body,
    is_done_line,
    openai_to_anthropic,
    parse_sse_line,
    stream_error,
)

log = logging.getLogger(__name__)

_REQUEST_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding",
                        "transfer-encoding", "keep-alive"}
#: httpx decodes the upstream body, so its encoding/length no longer apply
_RESPONSE_HOP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection",
                         "keep-alive"}
_MAX_SESSIONS = 2000


def _relay_headers(upstream: httpx.Response) -> dict[str, str]:
    """The upstream response's headers minus the ones httpx already decoded."""
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _RESPONSE_HOP_HEADERS}


def _boot_settings() -> dict:
    """The settings before the server's lifespan calls :func:`configure` with the
    user's own: the defaults. A broken ``local-models.yaml`` or an unset
    ``api_key_env`` must not make ``import`` fail -- the lifespan loads the real
    config and reports that error where it can be read."""
    try:
        return load_settings(None)
    except Exception as exc:  # noqa: BLE001 -- yaml errors and ValueErrors alike
        log.warning("default gateway settings not loadable (%s); starting with none", type(exc).__name__)
        return {"models": {}, "strongest": None,
                "router": dict(DEFAULT_ROUTER), "subagents": dict(DEFAULT_SUBAGENTS)}


_settings: dict = _boot_settings()
_picker: ModelPicker | None = None
_client: httpx.AsyncClient | None = None
_sessions: OrderedDict[str, dict] = OrderedDict()
#: prefix signature (system prompt + tools) -> (last seen, fewest prompt tokens
#: seen with it): the shared part every conversation with that prefix reuses
_prefixes: dict[str, tuple[float, int]] = {}
#: (model, prefix signature) -> when that model last cached that prefix
_warm: dict[tuple[str, str], float] = {}
#: which model wrote each message (by its id): a request that continues a
#: server-side thread from a message goes to the model that wrote it
_message_models: OrderedDict[str, str] = OrderedDict()
_MAX_MESSAGES = 5000
#: the API rejects a thread continuation sent to another model than the
#: thread's ("differing field(s): model"), so who wrote what must survive a
#: restart: appended to ``thread-models.jsonl``, read back on first use
_message_models_loaded = False
_THREAD_MISMATCH = re.compile(r"differing field\(s\):[^.]*\bmodel\b", re.I)


def _lru_set(cache: OrderedDict, key, value, max_size: int) -> None:
    """Store ``key`` as the most recent entry, evicting the oldest past ``max_size``."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _load_thread_models() -> None:
    """Read ``thread-models.jsonl`` into memory once per process. Entries
    already in memory are newer than the file's and win."""
    global _message_models_loaded
    if _message_models_loaded:
        return
    _message_models_loaded = True
    persisted: dict[str, str] = {}
    try:
        for entry in read_jsonl(_thread_models_path())[-_MAX_MESSAGES:]:
            persisted[entry["m"]] = entry["model"]
    except (OSError, KeyError, TypeError):
        pass
    for message_id, model in persisted.items():
        _message_models.setdefault(message_id, model)


def _remember_message(message_id: str, model: str) -> None:
    _load_thread_models()  # compaction below rewrites the file from memory: memory must hold what the file does
    _lru_set(_message_models, message_id, model, _MAX_MESSAGES)
    path = _thread_models_path()
    try:
        append_jsonl(path, {"m": message_id, "model": model})
        if path.stat().st_size > _MAX_MESSAGES * 200:  # ~2x the entries kept: compact
            atomic_write_text(path, "".join(json.dumps({"m": k, "model": v}) + "\n" for k, v in _message_models.items()))
    except OSError:
        pass  # memory still has it; a lost entry only costs one retried request


def _message_model(message_id: str) -> str | None:
    if message_id not in _message_models:
        _load_thread_models()
    return _message_models.get(message_id)
#: running per-request token profile of all agent traffic -- the default for a
#: conversation that has none of its own yet
_global_profile: dict = {}
_PROFILE_KEYS = ("input_tokens", "cache_creation_input_tokens", "output_tokens")
_PROFILE_ALPHA = 0.3


def _update_profile(profile: dict, usage: Mapping) -> dict:
    """Exponential moving average of the tokens a request actually used --
    the measured profile the router prices on each candidate model."""
    out = dict(profile)
    for k in _PROFILE_KEYS:
        v = float(usage.get(k) or 0)
        out[k] = v if k not in profile else (1 - _PROFILE_ALPHA) * profile[k] + _PROFILE_ALPHA * v
    return out


#: the subagent role -> model map from the config file, before the user's
#: dashboard overrides are laid over it
_config_roles: dict = {}
ROUTING_MODES = ("auto", "off", "pin")

#: subagent task hash -> the decision the spawn hook got (``record_spawn``),
#: so the subagent's own requests are labelled with it
_spawns: OrderedDict[str, dict] = OrderedDict()
_MAX_SPAWNS = 500

#: Claude Code session id -> running totals, for its status line
_cc_sessions: OrderedDict[str, dict] = OrderedDict()
#: (model, client user-agent) -> why the upstream rejected that model for
#: that client (e.g. "Claude Code x.y does not support this model")
_blocked: OrderedDict[tuple[str, str], tuple[str, float]] = OrderedDict()
#: how many (model, client) blocks are remembered: the key includes a client-sent header
_MAX_BLOCKED = 500
#: a block is retried after this long: a 400 can be passing (a model briefly
#: unavailable to this client) and a permanent block silently ends all savings
BLOCK_SECONDS = 600.0
#: 400s about this one request, not the model -- never a reason to stop using it
_REQUEST_SPECIFIC = re.compile(r"prompt is too long|too many tokens|max_tokens|context (?:length|window)"
                               r"|exceeds|too large|invalid image|could not process image"
                               r"|thread|previous_message_id", re.I)


#: a message that says the last answer was wrong. Phrases only: a bare "wrong"
#: or "no" would catch questions about code that is wrong. Automated messages
#: (subagent hand-backs, grading prompts) are excluded in is_correction.
CORRECTION = re.compile(r"\b(?:that'?s|that is|this is|it'?s) (?:wrong|incorrect|not (?:right|correct|what i (?:asked|meant|wanted)))"
                        r"|\bnot (?:quite )?(?:right|correct)\b|\bnot (?:quite )?what i (?:asked|meant|wanted)\b"
                        r"|\btry again\b|\bredo (?:it|that|this)\b|\byou missed\b|\bstill (?:broken|failing|wrong)\b"
                        r"|\b(?:doesn'?t|didn'?t) work\b|^\s*no[,.!]", re.I)
#: after a correction of a cheaper model's answer, this many main-agent turns
#: stay on the model the user asked for
CORRECTION_HOLD_TURNS = 8


def is_correction(text: str | None) -> bool:
    if not text or len(text) > 400:  # a correction is short; long text is a new task
        return False
    if text.lstrip().startswith(("<", "Another Claude session", "You are grading")):
        return False
    return bool(CORRECTION.search(text))


#: why the router isn't loaded, when it failed at startup (shown on the dashboard)
_router_error: str | None = None


def _reset_state() -> None:
    """Every cache and running total the interceptor keeps, emptied: what
    :func:`configure` starts from (and what a test needs between cases)."""
    global _message_models_loaded
    for state in (_sessions, _cc_sessions, _global_profile, _prefixes, _warm, _blocked, _spawns, _message_models,
                  _mod_loops, _mod_routes):
        state.clear()
    _message_models_loaded = False
    with _records_lock:
        _reset_log_cache()
        _archive_cache.clear()
        _summary_cache.update(key=None, value=None)


def configure(settings: dict, picker: ModelPicker | None, router_error: str | None = None) -> None:
    global _settings, _picker, _router_error
    _settings, _picker, _router_error = settings, picker, router_error
    _reset_state()
    _compress_pending()
    for r in recent_records(ARCHIVE_TAIL):  # seed the default profile from past agent traffic
        side = (r.get("reason") or "").startswith("side request")
        if r.get("status") == 200 and r.get("lane") == "anthropic" and not side:
            _global_profile.update(_update_profile(_global_profile, r))
    saved = read_preferences()
    if saved.get("quality_margin") is not None:
        _settings["router"]["margin"] = saved["quality_margin"]
    _settings["routing"] = {"mode": saved.get("mode", "auto"), "pinned_model": saved.get("pinned_model"),
                            "allow_upgrade": bool(saved.get("allow_upgrade", False)),
                            "turn_smoothing": bool(saved.get("turn_smoothing", True))}
    _config_roles.clear()
    _config_roles.update(_settings["subagents"].get("roles") or {})
    _settings["subagents"]["roles"] = {**_config_roles, **(saved.get("role_models") or {})}


#: per-request override, e.g. ``ANTHROPIC_CUSTOM_HEADERS="x-aar-quality-margin: 0.1"``
MARGIN_HEADER = "x-aar-quality-margin"
#: named cost preferences for the dashboard: how much predicted quality a turn
#: may give up to reach a cheaper model
#: Calibrated on LLMRouterBench's held-out split (1,738 prompts, 13 models,
#: the shipped MLP router) against always-GPT-5 (0.596 quality, $28.3 per 1k
#: prompts): 0.01 -> 0.626 at $26.5; 0.05 -> 0.612 at $19.8; 0.08 -> 0.590 at $14.8.
PRESETS = {"best_quality": 0.01, "balanced": 0.05, "save_more": 0.08}


def preferences() -> dict:
    routing = _settings.get("routing") or {"mode": "auto", "pinned_model": None}
    overrides = read_preferences().get("role_models") or {}
    defaults = {**DEFAULT_ROLE_MODELS, **_config_roles}
    return {"quality_margin": _settings["router"]["margin"], "presets": PRESETS,
            "router_enabled": _picker is not None, "router_error": _router_error,
            "strongest_model": _settings.get("strongest"),
            "header": MARGIN_HEADER, "mode": routing["mode"], "pinned_model": routing["pinned_model"],
            "allow_upgrade": bool(routing.get("allow_upgrade")),
            "turn_smoothing": bool(routing.get("turn_smoothing", True)),
            "turn_smoothing_warning": None if routing.get("turn_smoothing", True) else TURN_SMOOTHING_WARNING,
            "models": list(_settings["models"]),
            # what the dropdowns show next to a model that isn't Claude
            "model_notes": {m: ("local" if "127.0.0.1" in (spec.get("base_url") or "") or "localhost" in
                                (spec.get("base_url") or "") else "other provider")
                            for m, spec in _settings["models"].items() if spec.get("lane") == "openai"},
            "roles": [{"role": r, "model": _settings["subagents"]["roles"].get(r, d), "default": d,
                       "overridden": r in overrides} for r, d in defaults.items()]}


def _task_key(task: str) -> str:
    return hashlib.sha1(REMINDER.sub("", task or "").strip().encode()).hexdigest()[:16]


def record_spawn(session_id: str, subagent_type: str | None, task: str,
                 asked_model: str | None = None) -> dict:
    """The model for a subagent Claude Code is about to spawn (asked by the
    ``PreToolUse`` hook, ``aar-gateway hook``). Kept, so the subagent's own
    requests are labelled with it when they arrive. Respects the dashboard:
    routing off or pinned leaves the spawn alone."""
    routing = _settings.get("routing") or {}
    if routing.get("mode") in ("off", "pin") or not _settings["subagents"].get("enabled", True):
        decision = {"role": None, "source": None, "why": None, "model": None, "alias": None,
                    "reason": f"routing {routing.get('mode')}: spawn left as asked"}
    else:
        session_model = (_cc_sessions.get(session_id) or {}).get("requested")
        decision = spawn_decision(subagent_type, task, session_model, _settings, asked_model)
    _lru_set(_spawns, _task_key(task), decision, _MAX_SPAWNS)
    return decision


_save_preferences = save_preferences


def _log_override(record: dict) -> None:
    safe_append_jsonl(overrides_path(), {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record})


def set_routing(mode: str, pinned_model: str | None = None) -> dict:
    """``auto``: the router decides; ``off``: every request keeps the model
    Claude Code asked for; ``pin``: every agent turn goes to
    ``pinned_model`` (it may be pricier than the one asked for -- the user
    chose it). Applies from the next turn; kept across restarts."""
    if mode not in ROUTING_MODES:
        raise ValueError(f"mode must be one of {ROUTING_MODES}, got {mode!r}")
    if mode == "pin" and canonical_model(pinned_model) not in _settings["models"]:
        raise ValueError(f"pin needs a configured model, got {pinned_model!r}")
    before = dict(_settings.get("routing") or {})
    pinned = canonical_model(pinned_model) if mode == "pin" else None
    _settings["routing"] = {**before, "mode": mode, "pinned_model": pinned}
    _save_preferences(mode=mode, pinned_model=pinned)
    _log_override({"kind": "routing", "from": before, "to": _settings["routing"]})
    return preferences()


TURN_SMOOTHING_WARNING = ("Turn smoothing is off: each message is judged on its own, so a conversation can "
                          "switch model back and forth between messages, and every switch makes the new model "
                          "re-read the whole conversation into its cache (seen as +1200% requests).")


def set_turn_smoothing(on: bool) -> dict:
    """Whether each turn's prediction is blended with the previous turn's
    (on by default: it stops a conversation flip-flopping between models)."""
    before = dict(_settings.get("routing") or {})
    _settings["routing"] = {**before, "turn_smoothing": bool(on)}
    _save_preferences(turn_smoothing=bool(on))
    _log_override({"kind": "turn_smoothing", "from": bool(before.get("turn_smoothing", True)), "to": bool(on)})
    return preferences()


def set_allow_upgrade(allow: bool) -> dict:
    """Whether the router may move a hard turn to a model pricier than the
    one Claude Code asked for (off by default: then routing only ever saves)."""
    before = dict(_settings.get("routing") or {})
    _settings["routing"] = {**before, "allow_upgrade": bool(allow)}
    _save_preferences(allow_upgrade=bool(allow))
    _log_override({"kind": "allow_upgrade", "from": bool(before.get("allow_upgrade")), "to": bool(allow)})
    return preferences()


def set_role_model(role: str, model: str | None) -> dict:
    """A subagent role's tier: a configured model, ``"session"`` (keep the
    session's model) or ``"default"`` (drop the override)."""
    defaults = {**DEFAULT_ROLE_MODELS, **_config_roles}
    if role not in defaults:
        raise ValueError(f"unknown subagent role {role!r}")
    if model not in ("session", "default") and canonical_model(model) not in _settings["models"]:
        raise ValueError(f"unknown model {model!r}")
    overrides = dict(read_preferences().get("role_models") or {})
    before = _settings["subagents"]["roles"].get(role, defaults[role])
    if model == "default":
        overrides.pop(role, None)
    else:
        overrides[role] = None if model == "session" else canonical_model(model)
    _settings["subagents"]["roles"] = {**_config_roles, **overrides}
    _save_preferences(role_models=overrides)
    _log_override({"kind": "role", "role": role, "from": before,
                   "to": _settings["subagents"]["roles"].get(role, defaults[role])})
    return preferences()


def set_quality_margin(margin: float) -> dict:
    """The user's cost preference, applied to every new turn from now on and
    kept across restarts. 0 = always the best predicted model."""
    margin = float(margin)
    if not 0.0 <= margin <= 1.0:
        raise ValueError(f"quality_margin must be in [0, 1], got {margin}")
    _settings["router"]["margin"] = margin
    _save_preferences(quality_margin=margin)
    return preferences()


def _request_margin(request: Request) -> float:
    asked = request.headers.get(MARGIN_HEADER)
    if asked is not None:
        try:
            return min(max(float(asked), 0.0), 1.0)
        except ValueError:
            pass
    return float(_settings["router"]["margin"])


def set_model_picker(picker) -> None:
    """``picker(body, ctx)`` may return a model id, a decision dict with a
    ``model`` key, or ``None``."""
    global _picker
    _picker = picker


def set_upstream_client(client: httpx.AsyncClient | None) -> None:
    global _client
    _client = client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _client


def _anthropic_url(path: str) -> str:
    return os.environ.get("AAR_GATEWAY_UPSTREAM", "https://api.anthropic.com").rstrip("/") + "/" + path


def has_client_credentials(request: Request) -> bool:
    return "authorization" in request.headers or "x-api-key" in request.headers


def lane_of(model: str | None) -> str:
    return (_settings["models"].get(canonical_model(model)) or {}).get("lane", "anthropic")


_CWD = re.compile(r"Primary working directory: ([^\n]+)")


def working_directory(body: Mapping) -> str | None:
    """Where Claude Code was run -- it states it in an environment block of
    the messages (or the system prompt). A replay of the turn runs there."""
    for block in content_blocks(body.get("system")) + [b for m in body.get("messages") or []
                                                 for b in content_blocks(m.get("content"))]:
        found = _CWD.search(block.get("text", "") if isinstance(block, Mapping) else "")
        if found:
            return found.group(1).strip()
    return None


#: rotate the request log past this size: it becomes a dated archive (``requests-<UTC>.jsonl.gz``),
#: and is never deleted -- the history is what the router is retrained and personalised from
MAX_LOG_BYTES = 50 * 1024 * 1024

def join_archivers(timeout: float | None = None) -> None:
    archive.join_background(timeout)


def _rotate_log(path: Path) -> Path:
    """Move the full log to a dated archive and start gzipping it; the next append starts a new log."""
    dest = archive.rotate_to_archive(path)
    archive.run_in_background(_gzip_archive, dest)
    return dest


def _gzip_archive(plain: Path) -> None:
    """Gzip a rotated log in one pass that also builds its rollup, so the archive is
    never read again to be summarised."""
    state, newest = rollup.new_state(), collections.deque(maxlen=ARCHIVE_TAIL)

    def seen(record: dict) -> None:
        rollup.accumulate(state, record)
        newest.append(record)

    def store(gz: Path) -> None:
        st = gz.stat()
        atomic_write_text(gz.with_name(gz.name + ".summary.json"),
                          json.dumps({"key": [st.st_size, st.st_mtime_ns], "state": state, "tail": list(newest)}))

    archive.gzip_archive(plain, each=seen, done=store)


def _compress_pending() -> None:
    """Finish what a stopped gateway left: rotated logs not yet gzipped."""
    for path in log_archives():
        if path.name.endswith(".jsonl"):
            archive.run_in_background(_gzip_archive, path)


def _append_record(record: dict) -> None:
    path = log_path()
    try:
        if path.stat().st_size > MAX_LOG_BYTES:
            _rotate_log(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # e.g. Windows, another program has the log open: append to it, rotate next time
        log.warning("could not rotate the request log %s: %s", path, type(exc).__name__)
    # the response was already served: losing one record is the lesser harm
    if safe_append_jsonl(path, record):
        _notify()


#: open dashboard streams (``GET /api/events``): (their event loop, queue)
_listeners: set = set()


def subscribe() -> asyncio.Queue:
    """A queue that gets an item each time a request is logged."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    _listeners.add((asyncio.get_running_loop(), queue))
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    for entry in [e for e in _listeners if e[1] is queue]:
        _listeners.discard(entry)


def _offer(queue: asyncio.Queue) -> None:
    if not queue.full():  # one pending "something changed" is enough
        queue.put_nowait(True)


def _notify() -> None:
    for loop, queue in list(_listeners):
        try:
            loop.call_soon_threadsafe(_offer, queue)
        except RuntimeError:  # that loop is gone
            _listeners.discard((loop, queue))


_LEADING_TAG = re.compile(r"\s*<([A-Za-z][\w-]{0,40})")


def turn_trigger(text: str | None) -> str | None:
    """What started a turn, coarsely, for the log: ``"typed"``, or the name
    of a leading tag such as ``command-name`` -- never the text itself. A
    message Claude Code writes that isn't yet in :data:`REMINDER` shows up
    here by its tag."""
    if text is None:
        return None
    found = _LEADING_TAG.match(text)
    return found.group(1) if found else "typed"


def human_text(body: Mapping) -> str | None:
    """The new human text that starts a turn, or ``None`` for a tool-loop
    continuation or a message Claude Code wrote itself. Trailing ``system``
    messages (Claude Code appends hook output after the user's message) are
    skipped, and Claude Code's own sections (:data:`REMINDER`) removed."""
    msgs = [m for m in body.get("messages") or [] if m.get("role") != "system"]
    if not msgs or msgs[-1].get("role") != "user":
        return None
    blocks = content_blocks(msgs[-1].get("content"))
    if any(b.get("type") == "tool_result" for b in blocks):
        return None
    texts = [REMINDER.sub("", b.get("text", "")).strip() for b in blocks if b.get("type") == "text"]
    text = "\n".join(t for t in texts if t)
    return text or None


def claude_session_id(request: Request, body: Mapping) -> str:
    """Claude Code's session id: its header, else the ``_session_`` suffix
    of ``metadata.user_id`` (the same place claude-code-router reads it)."""
    sid = request.headers.get("x-claude-code-session-id")
    if sid:
        return sid
    user = str((body.get("metadata") or {}).get("user_id") or "")
    return user.split("_session_", 1)[1] if "_session_" in user else ""


def _conversation_key(request: Request, body: Mapping) -> str:
    """Stable across one conversation (history is append-only) and distinct
    for side requests that share Claude Code's session id -- subagents, title
    generation -- whose first message differs."""
    first = content_blocks(((body.get("messages") or [{}])[0]).get("content"))
    seed = json.dumps([(b.get("type"), b.get("text") or b.get("id") or "") for b in first])[:4000]
    sid = request.headers.get("x-claude-code-session-id", "")
    return hashlib.sha1((sid + seed).encode()).hexdigest()[:16]


def prefix_signature(body: Mapping) -> str:
    """What a prompt cache is keyed on, roughly: system prompt and tools.
    Claude Code sends the same ones in every session, so a new session
    reuses a cache an earlier session wrote -- on the same model."""
    seed = json.dumps([body.get("system"), body.get("tools")], sort_keys=True, default=str)
    return hashlib.sha1(seed.encode()).hexdigest()[:16]


def _ttl() -> float:
    return 3600.0 if _settings["router"].get("cache_ttl", "1h") == "1h" else 300.0


def _shared_tokens(sig: str) -> int:
    seen = _prefixes.get(sig)
    return seen[1] if seen and time.monotonic() - seen[0] < _ttl() else 0


def _warm_usage(usage: Mapping, warm: int) -> dict:
    """``usage`` as it would have been on a model holding ``warm`` prefix
    tokens in cache: that much of what was written is a read instead."""
    out = dict(usage)
    read = usage.get("cache_read_input_tokens") or 0
    moved = min(max(min(warm, prompt_tokens(usage)) - read, 0), usage.get("cache_creation_input_tokens") or 0)
    if moved:
        out["cache_creation_input_tokens"] -= moved
        out["cache_read_input_tokens"] = read + moved
        split = dict(usage.get("cache_creation") or {})
        left = moved
        for k in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
            take = min(left, split.get(k) or 0)
            if take:
                split[k] -= take
                left -= take
        out["cache_creation"] = split
    return out


class _SSEUsageTap:
    """Token usage read off an Anthropic SSE stream as it passes through."""

    #: an event larger than this is junk (or a stream that never separates events):
    #: the buffer is dropped rather than grown for the rest of the stream
    MAX_EVENT_BYTES = 1_000_000

    def __init__(self):
        self._buf = b""
        self._cr = b""  # a trailing CR: the LF of a CRLF may arrive in the next chunk
        self.usage: dict = {}
        self.stop_reason: str | None = None
        self.error: str | None = None  # an error event mid-stream (overloaded, ...)
        self.message_id: str | None = None

    def feed(self, chunk: bytes) -> None:
        data = self._cr + chunk
        self._cr = b"\r" if data.endswith(b"\r") else b""
        if self._cr:
            data = data[:-1]
        self._buf += data.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buf:
            event, self._buf = self._buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if line.startswith(b"data:"):
                    self._read(line[5:].strip())
        if len(self._buf) > self.MAX_EVENT_BYTES:
            self._buf = b""

    def _read(self, data: bytes) -> None:
        try:
            payload = json.loads(data)
        except ValueError:
            return
        if payload.get("type") == "message_start":
            self.usage.update((payload.get("message") or {}).get("usage") or {})
            self.message_id = (payload.get("message") or {}).get("id")
        elif payload.get("type") == "message_delta":
            self.usage.update({k: v for k, v in (payload.get("usage") or {}).items() if v is not None})
            self.stop_reason = (payload.get("delta") or {}).get("stop_reason") or self.stop_reason
        elif payload.get("type") == "error":
            self.error = error_summary(json.dumps(payload).encode())


class _Turn:
    """Everything known about one ``/v1/messages`` request, for routing and
    for its log record."""

    def __init__(self, request: Request, raw: bytes, body: dict):
        self.request, self.raw, self.body = request, raw, body
        self.requested = body.get("model")
        self.target = self.requested
        self.key = _conversation_key(request, body)
        self.claude_session = claude_session_id(request, body)
        self.sig = prefix_signature(body)
        self.shared_before = _shared_tokens(self.sig)
        self.session = dict(_sessions.get(self.key) or {})
        self.human = human_text(body)
        self.stop_reason: str | None = None
        # a subagent's role comes from its task, which only its first request
        # carries as new text; later requests reuse what that one found
        self.agent = (self.session["agent"] if self.human is None and "agent" in self.session
                      else detect_subagent(body, self.human))
        spawned = _spawns.get(_task_key(self.human)) if self.agent and self.human else None
        if spawned and spawned.get("role"):  # the spawn hook already decided this one
            self.agent = {"role": spawned["role"], "source": "hook", "why": spawned.get("why")}
        self.decision: dict = {"reason": "client's choice"}
        self.fallback = False
        self.attempted: str | None = None
        self.error: str | None = None  # the provider's error type and message, when it refused
        self.message_id: str | None = None  # the id of the message this request produced
        self.correction = False  # the user's message corrects a cheaper model's answer
        self.hold_left = self.session.get("hold", 0)  # main-agent turns still held on the asked model
        self.t0 = time.perf_counter()

    def body_for(self, model: str) -> bytes:
        return self.raw if model == self.requested else json.dumps({**self.body, "model": model}).encode()

    def fall_back(self, status: int, detail: str) -> None:
        """The routed model refused (``status``): answer with the client's own
        model instead, and say why in the log."""
        self.fallback, self.attempted, self.target = True, self.target, self.requested
        self.decision = {**self.decision, "reason": f"fell back: {self.attempted} returned {status}: {detail[:160]}"}

    def finish(self, status: int, usage: Mapping | None, lane: str) -> None:
        """Baselines ("on the model you asked for", "on the strongest") assume
        that model had served all the traffic: it would hold this
        conversation's previous prefix, or at least the system prompt and
        tools any recent request with the same prefix cached."""
        usage = dict(usage or {})
        # a subagent's baseline is the session's own model: what it would have
        # run on with no routing, even when its tier was set at spawn time
        asked = self.requested
        if self.agent:
            asked = (_cc_sessions.get(self.claude_session) or {}).get("requested") or self.requested
        prev_tokens = self.session.get("prompt_tokens") or 0
        if canonical_model(self.target) == canonical_model(asked):
            # served on the model asked for: what happened is the baseline, unless this
            # request switched back after a reroute (the reroute cost that model its
            # cache). Other conversations' prefixes don't count: a cache they warmed
            # that this request still had to write was never shared.
            previous = self.session.get("model")
            warm = prev_tokens if previous and canonical_model(previous) != canonical_model(asked) else 0
        else:
            warm = max(prev_tokens, self.shared_before)
            if canonical_model(self.target) in _warm_tokens(self.sig):
                # the routed model held this prefix: what it actually read is how much was
                # really shared, and the model asked for would have read no more
                warm = min(warm, usage.get("cache_read_input_tokens") or 0)
        baseline = _warm_usage(usage, warm)
        self._remember_served(status, usage, lane)
        cost = cost_usd(price_for(_settings, self.target), usage)
        cost_asked = cost_usd(price_for(_settings, asked), baseline)
        self._tally_claude_session(status, cost, cost_asked, asked)
        record = self._log_record(status, usage, lane, asked, baseline, cost, cost_asked)
        _append_record(record)
        share.record(record, self.human)  # opt-in only; a background thread, no prompt text
        if self.body.get("tools") and not (record.get("reason") or "").startswith("side request"):
            # kept on this computer for personalization; never uploaded (share has its own record)
            history.record(record, self.human, working_directory(self.body))

    def _remember_served(self, status: int, usage: dict, lane: str) -> None:
        """What a served request teaches the router: the message's model, the
        cache it warmed, the conversation's token profile."""
        if status == 200 and self.message_id and lane == "anthropic":
            _remember_message(self.message_id, self.target)
        if status == 200 and usage:
            now, tokens = time.monotonic(), prompt_tokens(usage)
            if lane == "anthropic":
                _warm[(canonical_model(self.target), self.sig)] = now
            fewest = min(self.shared_before or tokens, tokens)
            _prefixes[self.sig] = (now, fewest)
            _forget_cold(now)
            profile = _update_profile(self.session.get("profile") or {}, usage)
            quality = self.decision.get("model_quality") or self.session.get("quality")
            _lru_set(_sessions, self.key, {"model": self.target, "prompt_tokens": tokens, "profile": profile,
                                           "agent": self.agent, "quality": quality, "hold": self.hold_left},
                     _MAX_SESSIONS)
            if self.claude_session and self.agent is None and self.body.get("tools"):
                # the session's main loop, for the mod's next routing question (route_text)
                main = f"{self.claude_session[:12]}:main"
                _lru_set(_mod_loops, main, {**(_mod_loops.get(main) or {}), "model": self.target,
                                            "prompt_tokens": tokens, "profile": profile}, _MAX_SESSIONS)
            if self.body.get("tools"):
                _global_profile.update(_update_profile(_global_profile, usage))

    def _tally_claude_session(self, status: int, cost: float | None, cost_asked: float | None,
                              asked: str | None) -> None:
        """Running totals for the Claude Code status line."""
        if self.claude_session and status == 200:
            tot = _cc_sessions.setdefault(self.claude_session, {"cost_usd": 0.0, "cost_if_requested_usd": 0.0,
                                                                "requests": 0, "rerouted": 0})
            tot["requests"] += 1
            tot["cost_usd"] += cost or 0.0
            tot["cost_if_requested_usd"] += cost_asked if cost_asked is not None else (cost or 0.0)
            if (self.body.get("tools") and not self.agent) or "model" not in tot:
                # the main agent's model: not side requests, not subagents
                tot.update(model=self.target, requested=self.requested)
            tot["rerouted"] += self.target != asked
            _lru_set(_cc_sessions, self.claude_session, tot, _MAX_SESSIONS)

    def _log_record(self, status: int, usage: dict, lane: str, asked: str | None, baseline: dict,
                    cost: float | None, cost_asked: float | None) -> dict:
        d = self.decision
        strongest = _settings.get("strongest")
        return {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "claude_session": self.claude_session[:12] or None,
            "conversation": self.key,
            "new_turn": self.human is not None,
            "turn_trigger": turn_trigger(self.human),
            "correction": self.correction,
            "subagent": self.agent is not None,
            "agent_role": (self.agent or {}).get("role"),
            "role_source": (self.agent or {}).get("source"),
            "baseline_model": asked if asked != self.requested else None,
            # a subagent's final answer -- all the main agent gets back
            "report_tokens": usage.get("output_tokens") if self.agent and self.stop_reason == "end_turn" else None,
            "requested_model": self.requested,
            "routed_model": self.target,
            "lane": lane,
            "reason": d.get("reason"),
            "fallback": self.fallback,
            "attempted_model": self.attempted,
            "error": self.error,
            # server-side conversation state: Claude Code may send only what's new
            # since a previous message ("continue") instead of the whole history
            "thread": thread_mode(self.body),
            "status": status,
            "latency_s": round(time.perf_counter() - self.t0, 3),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cost_usd": cost,
            "cost_if_requested_usd": cost_asked,
            "strongest_model": strongest,
            "cost_if_strongest_usd": cost_usd(price_for(_settings, strongest), baseline),
            "predicted_quality": d.get("predicted_quality"),
            "best_predicted_quality": d.get("best_predicted_quality"),
            "candidates": d.get("candidates"),
            "quality_margin": d.get("quality_margin"),
            "quality_source": d.get("quality_source"),
        }


def _forget_cold(now: float) -> None:
    """Drop cache entries past the cache's lifetime: they can never be warm
    again, and otherwise every new system prompt or tool set ever seen stays."""
    ttl = _ttl()
    for key in [k for k, t in _warm.items() if now - t >= ttl]:
        del _warm[key]
    for sig in [k for k, (t, _) in _prefixes.items() if now - t >= ttl]:
        del _prefixes[sig]


def _warm_tokens(sig: str) -> dict[str, int]:
    """Models that recently cached this prefix, with how many tokens of it."""
    shared, now = _shared_tokens(sig), time.monotonic()
    return {m: shared for (m, s), t in _warm.items() if s == sig and now - t < _ttl()} if shared else {}


def _remember_block(turn: _Turn, status: int, detail: str) -> None:
    """A 400 for the routed model, when the request is valid for the client's
    own: the model rejects this client. Remember it for ``BLOCK_SECONDS`` so
    the router stops choosing it (both lanes). A 400 about this one request
    (too long, bad image...) says nothing about the model."""
    if status == 400 and not _REQUEST_SPECIFIC.search(detail):
        _lru_set(_blocked, (turn.target, turn.request.headers.get("user-agent", "")),
                 (detail[:160], time.monotonic()), _MAX_BLOCKED)


def _blocked_for(turn: _Turn) -> dict[str, str]:
    ua = turn.request.headers.get("user-agent", "")
    now = time.monotonic()
    return {m: why for (m, agent), (why, at) in _blocked.items()
            if agent == ua and now - at < BLOCK_SECONDS}


def _est_prompt_tokens(turn: _Turn) -> int:
    return max(turn.session.get("prompt_tokens") or 0, len(turn.raw) // 4)


async def _decide(turn: _Turn) -> None:
    mod_route = _mod_route_for(turn) if turn.agent is None else None
    if mod_route:
        # the mod switched this turn's model inside Claude Code: keep it on every
        # request of the turn, and count savings against the model asked for
        turn.target, turn.requested = turn.requested, mod_route["asked"]
        turn.decision = {"reason": f"routed in Claude Code: {mod_route.get('reason') or 'router'}"
                         if turn.human is not None else "same turn as the previous request"}
        return
    previous = thread_previous(turn.body)
    if previous:
        # the API keeps this conversation server-side and this request only
        # adds to it: it stays with the model that wrote the message it
        # continues (or the one asked for), never moved mid-thread
        turn.target = _message_model(previous) or turn.requested
        turn.decision = {"reason": "continues a server-side thread: stays on its model"}
        return
    if turn.human is None and turn.session.get("model"):
        turn.target = turn.session["model"]
        turn.decision = {"reason": "same turn as the previous request"}
        return
    if turn.human is None:
        return
    routing = _settings.get("routing") or {}
    if routing.get("mode") == "off":
        turn.decision = {"reason": "routing off (dashboard): the model Claude Code asked for"}
        return
    if routing.get("mode") == "pin" and turn.body.get("tools"):
        pinned = routing["pinned_model"]
        why = unavailable_reason(pinned, turn.body, _settings, _blocked_for(turn), _est_prompt_tokens(turn))
        if why:
            turn.decision = {"reason": f"pinned {pinned} can't take this request ({why}); kept the client's model"}
        else:
            turn.target, turn.decision = pinned, {"reason": f"pinned to {pinned} (dashboard)"}
        return
    if turn.agent and turn.body.get("tools") and _settings["subagents"].get("enabled", True):
        by_role = role_decision(turn.agent, turn.body, turn.requested, _settings,
                                _blocked_for(turn), _est_prompt_tokens(turn))
        if by_role:
            turn.target, turn.decision = by_role["model"], by_role
            return
    if turn.agent is None and turn.human is not None:
        if is_correction(turn.human) and turn.session.get("model") not in (None, turn.requested):
            turn.correction, turn.hold_left = True, CORRECTION_HOLD_TURNS
        if turn.hold_left > 0:
            turn.target = turn.requested
            turn.decision = {"reason": "after a correction: staying on the model you asked for "
                                       f"({turn.hold_left} more turn{'s' if turn.hold_left != 1 else ''})"}
            turn.hold_left -= 1
            return
    if _picker is None:
        return
    if not turn.body.get("tools"):
        # Claude Code's side requests (titles, summaries) carry no tools and
        # cost little; rerouting them saves nothing and risks a model that
        # rejects their settings. Only the agent conversation is routed.
        turn.decision = {"reason": "side request (no tools), not routed"}
        return
    ctx = {"requested": turn.requested, "human_text": turn.human,
           "est_prompt_tokens": _est_prompt_tokens(turn),
           "last_model": turn.session.get("model"),
           "prev_quality": turn.session.get("quality"),
           "last_prompt_tokens": turn.session.get("prompt_tokens"),
           "quality_margin": _request_margin(turn.request),
           "allow_upgrade": bool(routing.get("allow_upgrade")),
           "turn_smoothing": bool(routing.get("turn_smoothing", True)),
           "usage_profile": turn.session.get("profile") or dict(_global_profile) or None,
           "warm_tokens": _warm_tokens(turn.sig),
           "blocked": _blocked_for(turn)}
    try:
        picked = await run_in_threadpool(_picker, turn.body, ctx)
    except Exception as exc:  # a router failure must never break the client
        turn.decision = {"reason": f"router error, kept client's model: {exc!r}"[:300]}
        return
    if isinstance(picked, str):
        picked = {"model": picked, "reason": "picker"}
    if picked and picked.get("model"):
        turn.target, turn.decision = picked["model"], picked


async def _send_anthropic(request: Request, path: str, content: bytes) -> httpx.Response:
    client = _get_client()
    upstream_request = client.build_request(
        request.method, _anthropic_url(path), params=request.query_params,
        headers={k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_HOP_HEADERS},
        content=content)
    return await client.send(upstream_request, stream=True)


async def _relay_anthropic(upstream: httpx.Response, on_done) -> Response:
    headers = _relay_headers(upstream)
    if upstream.headers.get("content-type", "").startswith("text/event-stream"):
        tap = _SSEUsageTap()

        async def relay():
            aborted = None
            try:
                async for chunk in upstream.aiter_bytes():
                    tap.feed(chunk)
                    yield chunk
            except httpx.HTTPError as exc:  # the connection dropped mid-stream
                aborted = type(exc).__name__
                yield stream_error("the upstream connection was lost")
            finally:
                await upstream.aclose()
                if aborted:
                    on_done(502, tap.usage, tap.stop_reason, f"stream aborted: {aborted}", tap.message_id)
                else:
                    on_done(upstream.status_code, tap.usage, tap.stop_reason, tap.error, tap.message_id)

        return StreamingResponse(relay(), status_code=upstream.status_code, headers=headers)

    content = await upstream.aread()
    await upstream.aclose()
    usage = stop_reason = error = message_id = None
    if upstream.status_code == 200:
        try:
            message = json.loads(content)
            usage, stop_reason, message_id = message.get("usage"), message.get("stop_reason"), message.get("id")
        except (ValueError, AttributeError):
            pass
    else:
        error = error_summary(content)
    on_done(upstream.status_code, usage, stop_reason, error, message_id)
    return Response(content, status_code=upstream.status_code, headers=headers)


def _unreachable(turn: _Turn | None, where: str, exc: Exception, lane: str) -> JSONResponse:
    """A network failure reaching the provider: an Anthropic-shaped 502
    (Claude Code retries those) instead of an unlogged crash."""
    if turn is not None:
        turn.decision = {**turn.decision, "reason": f"{where} unreachable: {type(exc).__name__}"}
        turn.finish(502, None, lane)
    return error_response(502, f"{where} unreachable: {exc!r}"[:300])


async def _via_anthropic(turn: _Turn) -> Response:
    try:
        return await _via_anthropic_unguarded(turn)
    except httpx.HTTPError as exc:
        return _unreachable(turn, "Anthropic API", exc, "anthropic")


async def _find_thread_model(turn: _Turn, upstream: httpx.Response, *, tried: set) -> httpx.Response:
    """A thread continuation sent to a model other than the thread's is
    rejected before anything runs ("differing field(s): model"). That
    happens when the gateway doesn't know the thread's model (it started
    while the gateway was down, say): try the other Claude models once each
    and remember the one that takes it. Anything else is returned as is."""
    detail = (await upstream.aread()).decode(errors="replace")
    if not _THREAD_MISMATCH.search(detail):
        return _replay(upstream, detail)
    await upstream.aclose()
    for model in [m for m, spec in _settings["models"].items() if spec.get("lane") == "anthropic" and m not in tried]:
        candidate = await _send_anthropic(turn.request, "v1/messages", turn.body_for(model))
        if candidate.status_code == 400:
            text = (await candidate.aread()).decode(errors="replace")
            if _THREAD_MISMATCH.search(text):
                await candidate.aclose()
                continue
            return _replay(candidate, text)
        turn.target = model
        turn.decision = {**turn.decision, "reason": f"continues a server-side thread on {model} (found by retrying)"}
        _remember_message(thread_previous(turn.body), model)
        return candidate
    return _replay(upstream, detail)


def _replay(upstream: httpx.Response, text: str) -> httpx.Response:
    """A response already read, as a fresh one that can be relayed."""
    headers = _relay_headers(upstream)
    return httpx.Response(upstream.status_code, headers=headers, content=text.encode())


async def _via_anthropic_unguarded(turn: _Turn) -> Response:
    upstream = await _send_anthropic(turn.request, "v1/messages", turn.body_for(turn.target))
    if upstream.status_code >= 400 and turn.target != turn.requested:
        detail = (await upstream.aread()).decode(errors="replace")[:300]
        await upstream.aclose()
        _remember_block(turn, upstream.status_code, detail)
        turn.error = f"{turn.target}: {error_summary(detail.encode())}"  # kept if the fallback succeeds
        turn.fall_back(upstream.status_code, detail)
        upstream = await _send_anthropic(turn.request, "v1/messages", turn.body_for(turn.requested))
        if upstream.status_code == 400 and thread_previous(turn.body):
            upstream = await _find_thread_model(turn, upstream, tried={turn.attempted, turn.requested})
    elif upstream.status_code == 400 and thread_previous(turn.body):
        upstream = await _find_thread_model(turn, upstream, tried={turn.target})

    def done(status: int, usage: Mapping | None, stop_reason: str | None = None, error: str | None = None,
             message_id: str | None = None) -> None:
        turn.stop_reason = stop_reason
        turn.error = error or turn.error
        turn.message_id = message_id
        turn.finish(status, usage, "anthropic")

    return await _relay_anthropic(upstream, done)


def thread_previous(body: Mapping) -> str | None:
    """The message id a server-side thread request continues from, if any."""
    thread = body.get("thread")
    return thread.get("previous_message_id") if isinstance(thread, Mapping) else None


def thread_mode(body: Mapping) -> str | None:
    """``create`` / ``continue`` / the type given, for a request that uses
    the API's server-side threads; ``None`` for a plain full-history request.
    Only the mode is logged, never ids or content."""
    thread = body.get("thread")
    if not isinstance(thread, Mapping):
        return None
    if thread.get("previous_message_id"):
        return "continue"
    return str(thread.get("type") or "thread")[:20]


def error_summary(content: bytes) -> str:
    """``type: message`` from a provider's error body, for the request log.
    Error messages name fields and limits, not conversation text; capped
    all the same."""
    try:
        err = json.loads(content).get("error") or {}
        text = f"{err.get('type', 'error')}: {err.get('message', '')}"
    except (ValueError, AttributeError):
        text = content.decode(errors="replace")
    return " ".join(text.split())[:240]


def error_response(status: int, message: str, kind: str = "api_error") -> JSONResponse:
    return JSONResponse(error_body(kind, message), status_code=status)


def _invalid_request(message: str) -> JSONResponse:
    return error_response(400, message, "invalid_request_error")


def router_status() -> dict:
    """What ``/health`` reports about the router: loaded or not and why, its kind and source, the
    models, and its calibration. Read from the interceptor's own state, which is the only copy."""
    rcfg = _settings.get("router") or {}
    status = {"enabled": _picker is not None, "error": _router_error,
              "kind": getattr(_picker, "router_kind", None),
              "source": rcfg.get("artifact"),
              "models": sorted(_settings.get("models") or {})}
    if _picker is not None and hasattr(_picker, "calibration_status"):
        status["calibration"] = _picker.calibration_status()
    return status


def _json_object(raw: bytes) -> dict | None:
    """``raw`` parsed as a JSON object, else ``None`` (not JSON, or not an object)."""
    try:
        body = json.loads(raw or b"{}")
    except ValueError:  # includes undecodable bytes
        return None
    return body if isinstance(body, dict) else None


async def _via_openai(turn: _Turn) -> Response:
    try:
        return await _via_openai_unguarded(turn)
    except httpx.HTTPError as exc:
        return _unreachable(turn, turn.target, exc, "openai")


async def _via_openai_unguarded(turn: _Turn) -> Response:
    spec = _settings["models"][canonical_model(turn.target)]
    key_env = spec.get("api_key_env")
    headers = {"content-type": "application/json"}
    if key_env:
        headers["authorization"] = f"Bearer {os.environ[key_env]}"
    try:
        payload = anthropic_to_openai(turn.body, spec.get("upstream_model", turn.target),
                                      max_tokens_param=spec.get("max_tokens_param", "max_tokens"))
    except (KeyError, TypeError, AttributeError) as exc:  # a message or block missing a field the API requires
        turn.error = f"untranslatable request: {type(exc).__name__}"
        turn.finish(400, None, "openai")
        return _invalid_request("the request has a message or content block that is missing a required field")
    client = _get_client()
    upstream = await client.send(client.build_request(
        "POST", spec["base_url"].rstrip("/") + "/chat/completions", headers=headers, json=payload),
        stream=True)
    if upstream.status_code >= 400:
        detail = (await upstream.aread()).decode(errors="replace")[:500]
        await upstream.aclose()
        if turn.target != turn.requested:
            _remember_block(turn, upstream.status_code, detail)
        if turn.target != turn.requested and lane_of(turn.requested) == "anthropic" \
                and has_client_credentials(turn.request):
            turn.fall_back(upstream.status_code, detail)
            return await _via_anthropic(turn)
        turn.error = error_summary(detail.encode())
        turn.finish(upstream.status_code, None, "openai")
        return error_response(upstream.status_code, f"{turn.target}: {detail}")

    if payload["stream"]:
        translator = OpenAIStreamTranslator(turn.target)

        async def relay():
            aborted = None
            try:
                async for line in upstream.aiter_lines():
                    if is_done_line(line):
                        translator.done = True
                    chunk = parse_sse_line(line)
                    if chunk is not None:
                        for event in translator.feed(chunk):
                            yield event
                for event in translator.finish():
                    yield event
            except httpx.HTTPError as exc:  # the connection dropped mid-stream
                aborted = type(exc).__name__
                for event in translator.finish():  # a model that had already finished is not an error
                    yield event
            finally:
                await upstream.aclose()
                if translator.truncated:
                    turn.error = f"stream aborted: {aborted}" if aborted else "stream ended early"
                    turn.finish(502, translator.usage, "openai")
                else:
                    turn.finish(200, translator.usage, "openai")

        return StreamingResponse(relay(), media_type="text/event-stream")

    message = openai_to_anthropic(json.loads(await upstream.aread()), turn.target)
    await upstream.aclose()
    turn.finish(200, message["usage"], "openai")
    return JSONResponse(message)


async def handle_messages(request: Request) -> Response | None:
    """``None`` means "not mine": no credentials to forward and the model is
    not on an ``openai`` lane -- the server answers 401."""
    raw = await request.body()
    body = _json_object(raw)
    if body is None:
        return _invalid_request("the request body must be a JSON object")
    turn = _Turn(request, raw, body)
    await _decide(turn)
    if lane_of(turn.target) == "openai":
        return await _via_openai(turn)
    if not has_client_credentials(request):
        return None
    return await _via_anthropic(turn)


async def forward(request: Request, path: str) -> Response:
    """Any other Anthropic endpoint Claude Code calls (``count_tokens`` ...).
    For a model on an ``openai`` lane there is no Anthropic endpoint to ask,
    so token counts are estimated at ~4 characters per token."""
    raw = await request.body()
    if path == "v1/messages/count_tokens" and raw:
        model = (_json_object(raw) or {}).get("model")  # unreadable: the upstream says so itself
        if lane_of(model) == "openai":
            return JSONResponse({"input_tokens": len(raw) // 4})
    try:
        upstream = await _send_anthropic(request, path, raw)
    except httpx.HTTPError as exc:
        return _unreachable(None, "Anthropic API", exc, "anthropic")
    return await _relay_anthropic(upstream, lambda *_: None)  # passed through, not logged


#: the live request log as parsed so far: a dashboard refresh parses only the
#: lines appended since the last one, not the whole file. ``state`` is its
#: rollup, ``folded`` how many of ``records`` it already counts.
_log_cache: dict = {"key": None, "offset": 0, "partial": b"", "records": [], "state": None, "folded": 0}
#: held while the cache above and ``_archive_cache`` are read or advanced: /stats runs in worker threads
_records_lock = threading.RLock()
#: the last ``summarize()``, valid while the logs it was computed from are unchanged
_summary_cache: dict = {"key": None, "value": None}
#: archived logs: path -> (their size and mtime, rollup state, newest records)
_archive_cache: dict = {}
#: how many of an archive's newest records are kept beside its rollup (for the dashboard's recent list
#: and for seeding the token profile after a restart)
ARCHIVE_TAIL = 500


def _reset_log_cache() -> None:
    _log_cache.update(key=None, offset=0, partial=b"", records=[], state=None, folded=0)


def _archive_info(path: Path) -> tuple[dict, list]:
    """``(rollup state, newest records)`` of one archived log. Computed once by streaming
    the archive and stored beside it (``<archive>.summary.json``), so an archive is never
    read again unless it changed."""
    st = path.stat()
    key = [st.st_size, st.st_mtime_ns]
    cached = _archive_cache.get(str(path))
    if cached and cached[0] == key:
        return cached[1], cached[2]
    side = path.with_name(path.name + ".summary.json")
    stored = read_json(side)
    if stored.get("key") == key and "state" in stored:
        state, tail = stored["state"], stored["tail"]
    else:
        state, newest = rollup.new_state(), collections.deque(maxlen=ARCHIVE_TAIL)
        for record in iter_jsonl(path):
            rollup.accumulate(state, record)
            newest.append(record)
        tail = list(newest)
        try:
            atomic_write_text(side, json.dumps({"key": key, "state": state, "tail": tail}))
        except OSError as exc:  # the summary is a cache: rebuilt next time
            log.warning("could not store %s: %s", side, type(exc).__name__)
    _archive_cache[str(path)] = (key, state, tail)
    return state, tail


def read_records() -> list[dict]:
    """Every logged request, oldest first: all archived logs and the live one. Reads
    everything: for tests and tools, not for the dashboard (see :func:`summarize`)."""
    with _records_lock:
        out: list[dict] = []
        for path in log_archives():
            out.extend(iter_jsonl(path))
        return out + _current_records()


def recent_records(n: int) -> list[dict]:
    """The newest ``n`` logged requests (fewer if the history is shorter)."""
    with _records_lock:
        out = _current_records()[-n:]
        for path in reversed(log_archives()):
            if len(out) >= n:
                break
            out = (_archive_info(path)[1] + out)[-n:]
        return list(out)


def _current_records() -> list[dict]:
    path = log_path()
    try:
        st = path.stat()
    except FileNotFoundError:
        _reset_log_cache()
        return []
    key = (str(path), st.st_ino)
    if _log_cache["key"] != key or st.st_size < _log_cache["offset"]:  # new, rotated or rewritten
        _reset_log_cache()
        _log_cache["key"] = key
    if st.st_size > _log_cache["offset"]:
        with path.open("rb") as f:
            f.seek(_log_cache["offset"])
            data = f.read()
        _log_cache["offset"] += len(data)
        lines = (_log_cache["partial"] + data).split(b"\n")
        _log_cache["partial"] = lines.pop()  # an unfinished last line waits for the rest
        for line in lines:
            if not line.strip():
                continue
            try:
                _log_cache["records"].append(json.loads(line))
            except ValueError:  # a line cut short by a crash: skip it, keep the rest
                log.warning("skipping an undecodable line of %s", path)
    return _log_cache["records"]


#: newest requests sent to the dashboard, which pages through them
RECENT_RECORDS = 200


def summarize(records: list[dict] | None = None) -> dict:
    """Totals and per-model rows for ``GET /stats`` and the dashboard: over ``records`` if
    given, else over every request ever logged. The archives are rollups computed once, so
    this costs the live log only, however long the history."""
    if records is not None:
        state = rollup.new_state()
        for record in records:
            rollup.accumulate(state, record)
        return rollup.finish(state, recent=records[-RECENT_RECORDS:], log=str(log_path()),
                             default_strongest=_settings.get("strongest"))
    with _records_lock:
        archives = [(p, *_archive_info(p)) for p in log_archives()]
        live = _current_records()
        key = (tuple((str(p), tuple(_archive_cache[str(p)][0])) for p, _, _ in archives), _log_cache["key"],
               _log_cache["offset"], _settings.get("strongest"))
        if _summary_cache["key"] == key:
            return _summary_cache["value"]
        if _log_cache["state"] is None or _log_cache["folded"] > len(live):
            _log_cache.update(state=rollup.new_state(), folded=0)
        for record in live[_log_cache["folded"]:]:
            rollup.accumulate(_log_cache["state"], record)
        _log_cache["folded"] = len(live)
        state = _log_cache["state"]
        for _, older, _ in reversed(archives):
            state = rollup.merge(older, state)
        recent = live[-RECENT_RECORDS:]
        for _, _, tail in reversed(archives):
            if len(recent) >= RECENT_RECORDS:
                break
            recent = (tail + recent)[-RECENT_RECORDS:]
        value = rollup.finish(state, recent=recent, log=str(log_path()), default_strongest=_settings.get("strongest"))
        _summary_cache.update(key=key, value=value)
        return value


def summarize_sessions(records: list[dict]) -> list[dict]:
    """Plan usage per Claude Code session, against the same work on the models it asked
    for and on the strongest (requests logged before session ids existed are one row)."""
    return summarize(records)["sessions"]


def summarize_subagents(records: list[dict]) -> list[dict]:
    """One row per subagent role: spawns, the models that answered, plan usage against the
    session's own model, and how much of the prompt was read from cache."""
    return summarize(records)["subagents"]


#: what the Claude Code mod chose for a session's turn (``/api/route``): the
#: proxy keeps that model instead of routing the request again, and counts the
#: saving against the model Claude Code asked for. Keyed by session id.
_mod_routes: dict[str, dict] = {}
MOD_ROUTE_SECONDS = 120.0


def route_text(text: str, requested: str, session_id: str = "", est_prompt_tokens: int = 0,
               blocked: list | None = None) -> dict:
    """The model the router picks for a prompt, for a caller that switches the
    model itself (the Claude Code mod), under the proxy's rules (_decide): the
    pinned model only when it fits, the correction hold, and the session's
    conversation so far -- the model holding its cache and its length (seen
    live: a 250k-token session moved to Haiku and back cost 10x what staying
    on Opus would have), the last turn's quality for smoothing, and the
    models that failed earlier in it. Never raises: ``{"model": None, ...}``
    means keep the model asked for."""
    routing = _settings.get("routing") or {}
    key = f"{session_id[:12]}:main"
    loop = dict(_mod_loops.get(key) or {})
    if loop.get("asked") and requested == loop.get("model") and requested != loop["asked"]:
        # a resumed session (claude -c) restarts on the model the mod last switched it to,
        # which Claude Code recorded: the person asked for the one before.
        # shortcut: a /model choice of the model already in use reads the same; rare
        requested = loop["asked"]
    if session_id:
        loop["asked"] = requested
        _lru_set(_mod_loops, key, loop, _MAX_SESSIONS)
    keep = {"model": None, "requested": requested}
    est = max(est_prompt_tokens, loop.get("prompt_tokens") or 0)
    failed = {str(m): "failed earlier in this session" for m in (blocked or [])}
    body = {"model": requested, "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
    if routing.get("mode") == "off":
        return {**keep, "reason": "routing off (dashboard)"}
    if routing.get("mode") == "pin":
        pinned = routing["pinned_model"]
        why = unavailable_reason(pinned, body, _settings, failed, est)
        pick = ({"model": None, "reason": f"pinned {pinned} can't take this request ({why})"} if why
                else {"model": pinned, "reason": "pinned (dashboard)"})
    elif text and is_correction(text) and loop.get("model") not in (None, requested) or loop.get("hold", 0) > 0:
        # after "that's wrong" on a rerouted answer: the model asked for, for a few turns
        hold = CORRECTION_HOLD_TURNS if is_correction(text) else loop["hold"]
        _lru_set(_mod_loops, key, {**loop, "hold": hold - 1}, _MAX_SESSIONS)
        pick = {"model": requested, "reason": "after a correction: staying on the model you asked for "
                                              f"({hold - 1} more turn{'s' if hold - 1 != 1 else ''})"}
    elif _picker is None or not text.strip():
        return {**keep, "reason": "no router loaded" if _picker is None else "no prompt text"}
    else:
        ctx = {"requested": requested, "human_text": text, "est_prompt_tokens": est,
               "last_model": loop.get("model"), "prev_quality": loop.get("quality"),
               "last_prompt_tokens": loop.get("prompt_tokens"), "quality_margin": None,
               "allow_upgrade": bool(routing.get("allow_upgrade")),
               "usage_profile": loop.get("profile") or dict(_global_profile) or None,
               "turn_smoothing": bool(routing.get("turn_smoothing", True)),
               "warm_tokens": {}, "blocked": failed}
        try:
            picked = _picker(body, ctx)
        except Exception as exc:  # noqa: BLE001 -- a router failure keeps the asked model
            return {**keep, "reason": f"router error: {type(exc).__name__}"}
        if isinstance(picked, str):
            picked = {"model": picked, "reason": "picker"}
        picked = picked or {}
        if picked.get("model_quality"):
            _lru_set(_mod_loops, key, {**loop, "quality": picked["model_quality"]}, _MAX_SESSIONS)
        pick = {"model": picked.get("model"), "reason": picked.get("reason") or "router",
                "predicted_quality": picked.get("predicted_quality")}
    if pick["model"] and session_id:
        _mod_routes[session_id[:12]] = {"asked": requested, "model": pick["model"], "at": time.monotonic(),
                                        "reason": pick["reason"]}
    return {**pick, "requested": requested}


def _mod_route_for(turn: _Turn) -> dict | None:
    """The mod's choice for this request's session, if it names this model and is
    recent: the turn's first request or any later one (each use keeps it fresh,
    so a long tool loop stays counted against the model asked for)."""
    route = _mod_routes.get((turn.claude_session or "")[:12])
    if route and route["model"] == turn.requested and time.monotonic() - route["at"] < MOD_ROUTE_SECONDS:
        route["at"] = time.monotonic()
        return route
    return None


#: per (session, loop): the model and prompt size of its previous reported request,
#: for the baseline of the next (as the proxy keeps per conversation)
_mod_loops: OrderedDict[str, dict] = OrderedDict()


def _mod_reason(session: str, report: Mapping, routed: str, asked: str) -> str:
    if report.get("agent_id") or not report.get("new_turn"):
        return "same turn as the previous request"
    route = _mod_routes.get(session)
    return f"routed in Claude Code: {route.get('reason') or 'router'}" if route else (
        "routed in Claude Code" if routed != asked else "as asked")


def record_from_mod(report: Mapping) -> dict:
    """Log a request the Claude Code mod saw but the proxy didn't (a session
    that talks to Anthropic directly while the mod routes): priced the way the
    proxy prices its own. The baseline is the model asked for, with this
    loop's previous prompt in its cache when the previous request ran there."""
    usage = {k: int(report.get(k) or 0) for k in
             ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    asked, routed = str(report.get("requested_model") or ""), str(report.get("routed_model") or "")
    session = str(report.get("session_id") or "")[:12]
    loop_key = f"{session}:{report.get('agent_id') or 'main'}"
    loop = _mod_loops.get(loop_key) or {}
    if canonical_model(routed) == canonical_model(asked):
        warm = loop.get("prompt_tokens", 0) if loop.get("model") and canonical_model(loop["model"]) != canonical_model(asked) else 0
    else:
        warm = loop.get("prompt_tokens", 0) if loop.get("model") and canonical_model(loop["model"]) == canonical_model(asked) else 0
    baseline = _warm_usage(usage, warm)
    _lru_set(_mod_loops, loop_key, {**loop, "model": routed, "prompt_tokens": prompt_tokens(usage),
                                    "profile": _update_profile(loop.get("profile") or {}, usage)}, _MAX_SESSIONS)
    _global_profile.update(_update_profile(_global_profile, usage))
    strongest = _settings.get("strongest")
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "claude_session": session or None, "conversation": loop_key,
        "new_turn": bool(report.get("new_turn")), "turn_trigger": None, "correction": False,
        "subagent": bool(report.get("agent_id")), "agent_role": report.get("agent_role"), "role_source": "mod",
        "baseline_model": None, "report_tokens": None,
        "requested_model": asked, "routed_model": routed, "lane": "anthropic",
        "reason": str(report.get("reason") or _mod_reason(session, report, routed, asked))[:120],
        "fallback": False, "attempted_model": None, "error": None, "thread": None, "status": 200,
        "latency_s": None, **usage,
        "cost_usd": cost_usd(price_for(_settings, routed), usage),
        "cost_if_requested_usd": cost_usd(price_for(_settings, asked), baseline),
        "strongest_model": strongest, "cost_if_strongest_usd": cost_usd(price_for(_settings, strongest), baseline),
        "predicted_quality": None, "best_predicted_quality": None, "candidates": None,
        "quality_margin": None, "quality_source": "mod",
    }
    _append_record(record)
    return {"logged": True, "cost_usd": record["cost_usd"], "cost_if_requested_usd": record["cost_if_requested_usd"]}


def session_requests(session_id: str, limit: int = 40) -> dict:
    """One Claude Code session's recent requests for the mod's pane: models,
    why, cache use and cost against the model asked for. No prompt text."""
    key = (session_id or "")[:12]
    mine = [r for r in read_records() if key and r.get("claude_session") == key]
    fields = ("ts", "requested_model", "routed_model", "reason", "status", "subagent", "agent_role",
              "cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens", "output_tokens",
              "cost_usd", "cost_if_requested_usd", "correction", "thread")
    priced = [r for r in mine if r.get("cost_usd") is not None and r.get("cost_if_requested_usd")]
    cost, base = sum(r["cost_usd"] for r in priced), sum(r["cost_if_requested_usd"] for r in priced)
    return {"session": key, "requests": len(mine),
            "saved_pct": round(100 * (1 - cost / base), 1) if base else None,
            "recent": [{k: r.get(k) for k in fields} for r in mine[-max(1, min(limit, 200)):]],
            "turns": session_turns(mine)[-max(1, min(limit, 200)):]}


def session_turns(records: list[dict]) -> list[dict]:
    """A session's requests grouped into the turns the person typed: a turn
    starts with a main-loop request carrying new prompt text; its tool loop,
    side requests and subagents join it. Times are this machine's local time."""
    turns: list[dict] = []
    for r in records:
        side = str(r.get("reason") or "").startswith("side request")
        if not turns or (r.get("new_turn") and not r.get("subagent") and not side):
            try:
                at = datetime.fromisoformat(r["ts"]).astimezone().strftime("%H:%M")
            except (KeyError, TypeError, ValueError):
                at = ""
            turns.append({"time": at, "requested_model": r.get("requested_model"), "models": [],
                          "reason": r.get("reason"), "requests": 0, "side_requests": 0, "errors": 0,
                          "cache_read": 0, "cache_written": 0, "fresh": 0,
                          "cost_usd": 0.0, "cost_if_requested_usd": 0.0, "subagents": []})
        t = turns[-1]
        t["errors"] += r.get("status") != 200
        if r.get("cost_usd") is not None and r.get("cost_if_requested_usd"):
            t["cost_usd"] += r["cost_usd"]
            t["cost_if_requested_usd"] += r["cost_if_requested_usd"]
        if r.get("subagent"):
            role = r.get("agent_role") or "subagent"
            sub = next((s for s in t["subagents"] if s["role"] == role and s["model"] == r.get("routed_model")), None)
            if sub is None:
                t["subagents"].append(sub := {"role": role, "model": r.get("routed_model"), "requests": 0})
            sub["requests"] += 1
        elif side:
            t["side_requests"] += 1
        else:
            t["requests"] += 1
            t["cache_read"] += r.get("cache_read_input_tokens") or 0
            t["cache_written"] += r.get("cache_creation_input_tokens") or 0
            t["fresh"] += r.get("input_tokens") or 0
            if r.get("routed_model") and r["routed_model"] not in t["models"]:
                t["models"].append(r["routed_model"])
    for t in turns:
        base = t["cost_if_requested_usd"]
        t["saved_pct"] = round(100 * (1 - t["cost_usd"] / base)) if base else None
    return turns


def session_summary(session_id: str) -> dict:
    """What Claude Code's status line shows for one session: the model that
    answered the last turn, and how much less plan usage the session took
    than without the router. Usage is scaled by API price -- the relative cost
    of the same work on each model -- since on a subscription the dollars
    themselves are never billed."""
    tot = _cc_sessions.get(session_id)
    if not tot:
        return {"line": "aar router · waiting for the first turn"}
    model = tot.get("model") or "?"
    how = "routed" if model != tot.get("requested") else "as asked"
    base = tot["cost_if_requested_usd"]
    if base > 0:
        pct = 1 - tot["cost_usd"] / base
        usage = f"{pct:.0%} less plan usage" if pct >= 0 else f"{-pct:.0%} more plan usage"
    else:
        usage = "no usage yet"
    return {**tot, "plan_usage_saved": (1 - tot["cost_usd"] / base) if base > 0 else None,
            "line": f"aar router · {model} ({how}) · {usage}"}
