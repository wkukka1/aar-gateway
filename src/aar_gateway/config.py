"""Gateway settings: the ``interceptor:`` section of a gateway config (YAML)
merged over the defaults below -- which models Claude Code's requests may be
routed to, how each is reached, its prices, and the router's knobs. See
``gateway.example.yaml``. No secrets live in the YAML: an ``openai``
lane model names the environment variable its key is read from.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

#: what the interceptor may route Claude Code's requests to when no config
#: says otherwise. ``proxy`` is the router's pool model whose predicted quality
#: stands in for this one -- the defaults are LLMRouterBench's 2025 pool
#: (the LLMRouterBench pool): no public dataset scores today's Claude
#: tiers on shared prompts, so Opus reads GPT-5 (the pool's strongest), Sonnet
#: reads Claude Sonnet 4 (the same family, one generation back) and Haiku reads
#: Gemini 2.5 Flash (the pool's fast tier).
DEFAULT_MODELS: dict[str, dict] = {
    "claude-opus-5-5": {"lane": "anthropic", "proxy": "gpt-5", "context_window": 1_000_000,
                        "rejects": ["thinking_disabled"]},
    "claude-opus-5": {"lane": "anthropic", "proxy": "gpt-5", "context_window": 1_000_000},
    "claude-sonnet-5-5": {"lane": "anthropic", "proxy": "claude-sonnet-4", "context_window": 1_000_000,
                          "rejects": ["thinking_disabled"]},
    "claude-sonnet-5": {"lane": "anthropic", "proxy": "claude-sonnet-4", "context_window": 1_000_000},
    "claude-haiku-5-5": {"lane": "anthropic", "proxy": "gemini-2.5-flash", "context_window": 1_000_000},
    "claude-haiku-4-5": {"lane": "anthropic", "proxy": "gemini-2.5-flash", "context_window": 200_000,
                         "rejects": ["effort"]},
}
#: the router that serves savings: a turn priced by cost weight (lambda) is scored by it. It scores
#: the Claude tiers directly, so it needs no proxies and no stand-ins; its BERT encoder is
#: ``aar-gateway``'s ``encoders/bert-base-uncased-mean`` (see docs/gateway.md).
BUNDLED_SAVINGS_ROUTER = Path(__file__).with_name("routers") / "manifold-nirt-claude-judged.npz"
DEFAULT_ROUTER = {"enabled": False, "artifact": None, "margin": 0.05, "cost_weight": None,
                  "savings_artifact": str(BUNDLED_SAVINGS_ROUTER),
                  "expected_output_tokens": 800, "cache_ttl": "1h", "horizon_requests": 20,
                  "switch_margin": 0.05, "turn_smoothing": 0.5}
#: Claude Code subagents (``aar_gateway.subagents``): ``roles`` maps a role
#: to a model (``null`` keeps the session's), merged over that module's
#: defaults; ``enabled: false`` routes subagents like any other turn.
DEFAULT_SUBAGENTS = {"enabled": True, "roles": {}}


def log_path() -> Path:
    """The request log: ``AAR_GATEWAY_LOG`` (``~`` expanded), else
    ``~/.aar-gateway/requests.jsonl``. Every other gateway file lives beside it."""
    log = os.environ.get("AAR_GATEWAY_LOG")
    return Path(log).expanduser() if log else Path.home() / ".aar-gateway" / "requests.jsonl"


def gateway_home() -> Path:
    """Where the gateway keeps its files: the request log's folder."""
    return log_path().parent


def pid_file() -> Path:
    return gateway_home() / "gateway.pid"


def thread_models_path() -> Path:
    """Which model wrote each message id, so a server-side thread continues on it after a restart."""
    return gateway_home() / "thread-models.jsonl"


def overrides_path() -> Path:
    """Every manual routing change from the dashboard, one JSON line each: labelled data on
    where the router's or a role's default was wrong."""
    return gateway_home() / "overrides.jsonl"


def models_dir() -> Path:
    """Where the router's text encoder is cached."""
    return gateway_home() / "models"


def encoders_dir() -> Path:
    """Local ONNX text encoders a router file names by folder (``encoder.model_path``),
    e.g. the BERT encoder of a NIRT file; fastembed's own downloads stay in ``models``."""
    return gateway_home() / "encoders"


def routers_dir() -> Path:
    """Router files added with ``aar-gateway router add`` (``router use`` picks one)."""
    return gateway_home() / "routers"


def graded_path() -> Path:
    """The user's graded turns (``aar-gateway-grade`` writes, the router reads)."""
    return gateway_home() / "graded.jsonl"


def config_path() -> Path | None:
    """The gateway config in use: ``AAR_GATEWAY_CONFIG``, else
    ``~/.aar-gateway/config.yaml`` if it exists, else none (the defaults)."""
    named = os.environ.get("AAR_GATEWAY_CONFIG")
    if named:
        return Path(named).expanduser()
    default = gateway_home() / "config.yaml"
    return default if default.exists() else None


def local_models_path() -> Path:
    return gateway_home() / "local-models.yaml"


def local_models() -> dict[str, dict]:
    """Models added with ``aar-gateway local add`` -- ``{}`` when none."""
    path = local_models_path()
    if not path.exists():
        return {}
    return {str(k): dict(v) for k, v in ((yaml.safe_load(path.read_text()) or {}).get("models") or {}).items()}


def load_settings(config_path: str | Path | None = None) -> dict:
    """The interceptor's settings: the ``interceptor:`` section of a gateway
    config merged over the defaults above. Fails at load time, not on a
    request, for a bad lane, an ``openai`` lane without ``base_url``, or an
    ``api_key_env`` that isn't set."""
    from .pricing import CLAUDE_PRICES

    spec = (yaml.safe_load(Path(config_path).read_text()) or {}) if config_path else {}
    section = spec.get("interceptor") or {}
    models = {k: dict(v) for k, v in (section.get("models") or DEFAULT_MODELS).items()}
    # opt-in local models (`aar-gateway local add`), kept in a file of their own
    # so the main config is never rewritten; they never replace a configured model
    for model_id, m in local_models().items():
        models.setdefault(model_id, dict(m))
    for model_id, m in models.items():
        lane = m.setdefault("lane", "anthropic")
        if lane not in ("anthropic", "openai"):
            raise ValueError(f"gateway config: model {model_id!r} has unknown lane {lane!r}")
        if lane == "openai" and not m.get("base_url"):
            raise ValueError(f"gateway config: openai-lane model {model_id!r} needs base_url")
        key_env = m.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            raise ValueError(f"gateway config: model {model_id!r} names api_key_env={key_env!r} "
                             "but it is not set in the environment")
        if "prices" not in m and model_id in CLAUDE_PRICES:
            m["prices"] = CLAUDE_PRICES[model_id]
    # The "always the strongest" baseline: the config's choice, else the first
    # priced Claude model listed (the defaults list Opus 5.5 first). Not the
    # priciest: Opus 5 costs more than the newer, stronger Opus 5.5, and
    # measuring against it overstated the saving.
    anthropic_priced = [k for k, m in models.items() if m["lane"] == "anthropic" and m.get("prices")]
    strongest = section.get("strongest") or next(iter(anthropic_priced), None)
    return {"models": models, "strongest": strongest,
            "router": {**DEFAULT_ROUTER, **(section.get("router") or {})},
            "subagents": {**DEFAULT_SUBAGENTS, **(section.get("subagents") or {})}}


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` in one step: a reader sees the old file or the new
    one, never half of it. Each writer gets its own temporary file, so two
    writing at once can't mix their content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        for attempt in range(100):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows won't replace a file another program has open (Claude
                # Code reading its settings, say): it is free again moments later
                if os.name != "nt" or attempt == 99:
                    raise
                time.sleep(0.02)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> list[dict]:
    """Every record in a jsonl file, oldest first. A missing file is empty; an
    unreadable one raises. A line that does not decode (cut short by a crash or
    rotation, or half-written) is skipped and logged by position, never by
    content: one bad line must not cost the rest of the file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    records = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            log.warning("skipping an undecodable line %d of %s", n, path)
    return records


@contextlib.contextmanager
def locked_file(path: Path, *, timeout: float = 30.0):
    """Hold an exclusive lock on ``path`` across processes (the gateway and
    the CLI both rewrite the share queue). The lock lives on a sidecar
    ``<name>.lock`` file, so replacing ``path`` itself does not release it.
    Every writer of ``path`` must take it; readers need not (replacement is
    atomic)."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with open(lock, "a+b") as f:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {lock} within {timeout:.0f}s") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def safe_append_jsonl(path: Path, *records: dict) -> bool:
    """:func:`append_jsonl` for records that must never cost a request: a full
    disk or a locked file is logged (the exception type only) and ``False`` is
    returned; the caller carries on."""
    try:
        append_jsonl(path, *records)
    except OSError as exc:
        log.warning("could not write %s: %s", path, type(exc).__name__)
        return False
    return True


def append_jsonl(path: Path, *records: dict) -> None:
    """Append ``records`` to a jsonl file, one line each, creating its folder.
    Raises ``OSError`` if the write fails; callers choose whether that matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_json(path: Path, default: dict | None = None, *, strict: bool = False) -> dict:
    """A small json object file. Missing is ``default``. A file that is not a
    json object is logged (its path, never its content) and is ``default`` too,
    unless ``strict``, which raises ``ValueError`` so a caller about to rewrite
    the file can set the unreadable one aside instead of erasing it."""
    default = {} if default is None else default
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("not a json object")
    except ValueError:
        log.warning("%s is not a readable json object", path)
        if strict:
            raise
        return default
    return value


def log_archives() -> list[Path]:
    """The rotated request logs, oldest first (see :func:`aar_gateway.archive.archives_of`)
    including the single ``<log>.1`` older versions kept."""
    from .archive import archives_of

    return archives_of(log_path(), legacy=True)


def iter_jsonl(path: Path):
    """The records of a jsonl file (gzip too, by its ``.gz`` name), one at a time, oldest
    first: for logs too large to hold. An undecodable line is skipped and logged by position."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                log.warning("skipping an undecodable line %d of %s", n, path)


def preferences_path() -> Path:
    """The user's saved choices (routing mode, quality margin, subagent models, history on/off)."""
    return gateway_home() / "preferences.json"


def read_preferences() -> dict:
    try:
        return read_json(preferences_path())
    except OSError:
        return {}


def save_preferences(**changes) -> None:
    """Merge ``changes`` into the preferences file. An unreadable file (cut short,
    hand-edited) is kept aside as ``preferences.json.corrupt`` rather than erased."""
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = read_json(path, strict=True)
    except ValueError:
        path.replace(path.with_name(path.name + ".corrupt"))
        current = {}
    atomic_write_text(path, json.dumps({**current, **changes}))
