"""Opt-in usage sharing: anonymous routing records, to improve the router.

Off unless the user turns it on (``aar-gateway setup`` asks; ``aar-gateway
share on|off``). Then each message the user types adds one record to
``~/.aar-gateway/share/queue.jsonl``:

* the models asked for and used, and why (a category, not free text);
* tokens, cost, the cost on the model asked for, latency, status and the
  error *type*; whether the message corrected a cheaper model's answer; the
  thread mode; the prompt's length, rounded to a power of two;
* the prompt's embedding, compressed: the 384-number MiniLM vector projected
  onto 64 fixed random directions (:data:`PROJECTION_SEED`) and rounded to
  small integers. Coarse on purpose: a full embedding of a known model can
  be turned back into much of a short text.

Never: the prompt, error messages, file paths, session or conversation ids,
times finer than the day. The queue stays on this computer, never trimmed
(``aar-gateway share show`` prints it), until a collection endpoint is set
(``aar-gateway share endpoint``); then it is posted there as one gzip'd jsonl
body once a day, or after :data:`UPLOAD_AFTER` records, with a random install
id (not tied to the person) and the record format version. The collection
server holds the credentials for the project's dataset; nothing on this
computer can write to it directly. A failed upload keeps the records for next
time and never affects routing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .config import atomic_write_text, gateway_home, locked_file, read_json, read_jsonl

log = logging.getLogger(__name__)

#: the compressed embedding: this many dimensions, from this fixed projection
DIMS = 64
PROJECTION_SEED = 20261017
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: where records go, and when. No address is built in: until one is set (``aar-gateway share
#: endpoint`` or AAR_GATEWAY_SHARE_ENDPOINT) the queue stays on this computer.
DEFAULT_ENDPOINT: str | None = None
#: the record format version sent with every upload
SCHEMA = 1
UPLOAD_AFTER = 100
UPLOAD_EVERY_S = 24 * 3600
#: why a model was used, as categories (the gateway's reasons are free text)
_REASONS = (("continues a server-side thread", "thread"), ("same turn", "same_turn"),
            ("pinned", "pinned"), ("routing off", "routing_off"), ("after a correction", "after_correction"),
            ("side request", "side_request"), ("fell back", "fallback"), ("router error", "router_error"),
            ("client's choice", "client_choice"), ("subagent", "subagent"))

#: records waiting for the worker (each holds the user's message text in memory)
MAX_PENDING = 200
#: one upload gets this long; the records stay queued for the next attempt
UPLOAD_TIMEOUT_S = 120.0

_lock = threading.Lock()
_worker: ThreadPoolExecutor | None = None
_pending = 0
#: ``(queue file size, records)`` after our last append: valid while the size adds up
_queue_state: dict = {"size": None, "count": 0}
_encoder = None
_projection = None
_consent: dict = {"mtime": None, "on": False}


def share_dir() -> Path:
    return gateway_home() / "share"


def consent_path() -> Path:
    return share_dir() / "consent.json"


def queue_path() -> Path:
    return share_dir() / "queue.jsonl"


def endpoint_path() -> Path:
    return share_dir() / "endpoint"


def _valid_endpoint(url: str) -> bool:
    """HTTPS, or plain http only to this computer (a collection server under test)."""
    from urllib.parse import urlparse

    parts = urlparse(url.strip())
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"))


def endpoint() -> str | None:
    """Where records are sent: ``AAR_GATEWAY_SHARE_ENDPOINT``, else the one saved by
    ``aar-gateway share endpoint``, else :data:`DEFAULT_ENDPOINT`. An address that is
    not HTTPS (or local) is ignored, never posted to; with none, records stay on this computer."""
    for candidate in (os.environ.get("AAR_GATEWAY_SHARE_ENDPOINT"), _saved_endpoint(), DEFAULT_ENDPOINT):
        if candidate and _valid_endpoint(candidate):
            return candidate.strip()
        if candidate:
            log.warning("ignoring a usage-sharing endpoint that is not https")
    return None


def _saved_endpoint() -> str | None:
    try:
        return endpoint_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def set_endpoint(url: str) -> None:
    if not _valid_endpoint(url):
        raise ValueError("the usage-sharing endpoint must be an https:// address")
    atomic_write_text(endpoint_path(), url.strip())


def state_path() -> Path:
    return share_dir() / "state.json"


def state() -> dict:
    try:
        return read_json(state_path())
    except OSError:
        return {}


def _install_id() -> str:
    path = share_dir() / "install-id"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        import uuid

        value = uuid.uuid4().hex[:16]
        atomic_write_text(path, value)
        return value


def _call_with_timeout(fn, seconds: float):
    """``fn()``'s result, or ``TimeoutError`` after ``seconds`` (the call, which
    cannot be interrupted, is left to finish on its own daemon thread)."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=run, name="aar-share-upload", daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError("upload timed out")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _post(url: str, body: bytes, headers: dict) -> int:
    """POST ``body``; the HTTP status (an error status is returned, a network failure raises)."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=UPLOAD_TIMEOUT_S) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def upload(*, post=None) -> dict:
    """Send the queued records to the collection server as one gzip'd jsonl body; on a 2xx
    they leave the queue. ``{"sent": n}``, or ``{"error": ...}`` / ``{"skipped": why}``.
    ``post(url, body, headers) -> status`` is for tests."""
    import gzip

    url = endpoint()
    if not url:
        return {"skipped": "no collection endpoint (aar-gateway share endpoint <https url>)"}
    with _lock:
        path = queue_path()
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    lines = [line for line in lines if line.strip()]
    if not lines:
        return {"skipped": "nothing queued"}
    now = datetime.now(UTC)
    st = state()
    headers = {"content-type": "application/x-ndjson", "content-encoding": "gzip", "x-aar-schema": str(SCHEMA),
               "x-aar-install": _install_id(), "user-agent": f"aar-gateway/{_version()}"}
    body = gzip.compress("".join(lines).encode("utf-8"))
    try:
        status = _call_with_timeout(lambda: (post or _post)(url, body, headers), UPLOAD_TIMEOUT_S)
        if not 200 <= status < 300:
            st.update(last_error=f"http_{status}", last_attempt=now.isoformat(timespec="seconds"))
            atomic_write_text(state_path(), json.dumps(st))
            return {"error": f"http_{status}"}
    except Exception as exc:  # noqa: BLE001 -- kept for next time; the type only, never the message
        st.update(last_error=type(exc).__name__, last_attempt=now.isoformat(timespec="seconds"))
        atomic_write_text(state_path(), json.dumps(st))
        return {"error": type(exc).__name__}
    with _lock, locked_file(path):  # drop exactly what was sent; anything written meanwhile stays
        current = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
        sent = set(lines)
        remaining = current[len(lines):] if current[:len(lines)] == lines else [x for x in current if x not in sent]
        atomic_write_text(path, "".join(remaining))
    st.update(last_upload=now.isoformat(timespec="seconds"), sent_total=st.get("sent_total", 0) + len(lines),
              last_error=None, last_attempt=now.isoformat(timespec="seconds"))
    atomic_write_text(state_path(), json.dumps(st))
    return {"sent": len(lines)}


def _upload_due(n_queued: int) -> bool:
    if not endpoint() or n_queued == 0:
        return False
    if n_queued >= UPLOAD_AFTER:
        return True
    last = state().get("last_attempt")
    if not last:
        return True
    return (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds() >= UPLOAD_EVERY_S


def enabled() -> bool:
    """The user's choice, re-read when the consent file changes (the CLI
    writes it; the gateway runs in another process)."""
    path = consent_path()
    try:
        st = path.stat()
    except OSError:
        return False
    stamp = (st.st_mtime_ns, st.st_size)  # a coarse clock can give a rewrite the old mtime; the size rarely matches too
    if stamp != _consent["mtime"]:
        try:
            on = bool(read_json(path).get("share_usage"))
        except OSError:
            on = False
        _consent.update(mtime=stamp, on=on)
    return _consent["on"]


def set_enabled(on: bool) -> None:
    atomic_write_text(consent_path(), json.dumps({"share_usage": on,
                                                  "changed": datetime.now(UTC).date().isoformat()}))


def reason_category(reason: str | None) -> str:
    text = (reason or "").lower()
    return next((cat for key, cat in _REASONS if key in text), "router")


def _compress(vector) -> list[int]:
    import numpy as np

    global _projection
    if _projection is None:
        rng = np.random.default_rng(PROJECTION_SEED)
        _projection = rng.normal(0, 1 / math.sqrt(DIMS), (len(vector), DIMS))
    z = np.asarray(vector, dtype=float) @ _projection
    z = z / (np.linalg.norm(z) or 1.0)
    return [int(v) for v in np.clip(np.round(z * 127), -127, 127)]


def _embed(text: str) -> list[int] | None:
    global _encoder
    try:
        if _encoder is None:
            from ._core.encoder import EncoderConfig, TextEncoder, light_backend

            _encoder = TextEncoder(EncoderConfig(name="share", model_name=EMBEDDING_MODEL, backend=light_backend()))
        return _compress(_encoder.encode([text])[0])
    except Exception:  # noqa: BLE001 -- a missing encoder means no embedding, never a failed record
        return None


_NAME = re.compile(r"^[A-Za-z0-9_.:/\[\]+\-]{1,80}$")   # the collection server's formats (ingest.schema)
_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def _name(value) -> str | None:
    """A model id or version in the server's format, else ``None``: a name with spaces or
    free text is not sent."""
    return value if isinstance(value, str) and _NAME.match(value) else None


def _token(value) -> str | None:
    """A short category in the server's format; anything else that is not empty becomes ``"other"``."""
    if not value:
        return None
    return value if isinstance(value, str) and _TOKEN.match(value) else "other"


def build_record(record: dict, human: str | None, *, embed: bool = True) -> dict:
    """The shareable part of a request-log record (see the module docstring).
    ``embed=False`` leaves the embedding out (the slow part) but keeps the rest."""
    error = record.get("error") or ""
    return {
        "v": 1,
        "day": (record.get("ts") or datetime.now(UTC).isoformat())[:10],
        "gateway": _name(_version()) or "dev",
        "requested_model": _name(record.get("requested_model")),
        "routed_model": _name(record.get("routed_model")),
        "decision": reason_category(record.get("reason")),
        "subagent_role": _token(record.get("agent_role")),
        "correction": bool(record.get("correction")),
        "thread": _token(record.get("thread")),
        "status": record.get("status"),
        "error_type": _token(error.split(":", 1)[0].strip()) if error else None,
        "latency_s": record.get("latency_s"),
        **{k: record.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                      "cache_creation_input_tokens", "cost_usd", "cost_if_requested_usd")},
        "prompt_chars": 2 ** round(math.log2(len(human))) if human else None,
        "embedding_model": f"{EMBEDDING_MODEL}/rp{DIMS}-int8",
        "embedding": _embed(human) if human and embed else None,
    }


def record(log_record: dict, human: str | None) -> None:
    """Queue one record when sharing is on: a message the user typed only.
    Runs on a background thread: never slows the request."""
    if not human or log_record.get("subagent") or not enabled():
        return
    if reason_category(log_record.get("reason")) == "side_request":
        return  # Claude Code's own background requests (titles, summaries): not the user's messages
    global _worker, _pending
    with _lock:
        backed_up = _pending >= MAX_PENDING
        if not backed_up:
            _pending += 1
        if _worker is None:
            _worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aar-share")
        worker = _worker
    if backed_up:
        # keep the record, not the message text waiting to be embedded: the embedding is the slow part
        log.warning("usage sharing is backed up (%d waiting): keeping a record without its embedding", MAX_PENDING)
        try:
            _queue_line(json.dumps(build_record(log_record, human, embed=False)) + "\n")
        except Exception as exc:  # noqa: BLE001 -- sharing must never affect routing
            log.warning("usage sharing failed: %s", type(exc).__name__)
        return
    worker.submit(_append, dict(log_record), human)


def _append(log_record: dict, human: str) -> None:
    global _pending
    try:
        n = _queue_line(json.dumps(build_record(log_record, human)) + "\n")
        if _upload_due(n):
            upload()
    except Exception as exc:  # noqa: BLE001 -- sharing must never affect routing
        log.warning("usage sharing failed: %s", type(exc).__name__)  # the type only: a message can echo a token
    finally:
        with _lock:
            _pending -= 1


def _queue_line(line: str) -> int:
    """Append one record to the queue; returns its length. Records leave the queue
    only when uploaded (the on-device history is the lasting copy). Locked across processes: the CLI's upload rewrites the same file."""
    path = queue_path()
    with _lock, locked_file(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as f:  # bytes: the size then grows by exactly len(line) on every OS
            f.write(line.encode("utf-8"))
        size = path.stat().st_size
        known = _queue_state["size"]
        if known is not None and size == known + len(line.encode("utf-8")):
            count = _queue_state["count"] + 1
        else:  # first append, a trim, or another process wrote: count the file
            count = _count_lines(path)
        _queue_state.update(size=size, count=count)
        return count


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def queued() -> list[dict]:
    return read_jsonl(queue_path())


def shutdown(timeout: float | None = 10) -> None:
    """Write what is pending, then stop the worker (the server's shutdown)."""
    global _worker
    worker, _worker = _worker, None
    if worker is not None:
        worker.shutdown(wait=True, cancel_futures=False)


def flush(timeout: float | None = None) -> None:
    """Wait for queued records to be written (tests, shutdown)."""
    if _worker is not None:
        _worker.submit(lambda: None).result(timeout=timeout)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("agentic-llm-router")
    except Exception:  # noqa: BLE001
        return os.environ.get("AAR_GATEWAY_VERSION", "dev")
