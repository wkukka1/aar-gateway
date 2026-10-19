"""The on-device history: every message the user typed, with the route it got and how it went.

Kept for personalization and for grading (``aar-gateway-grade`` reads these rows), on
this computer only: nothing here is ever uploaded -- the opt-in usage sharing in
:mod:`aar_gateway.share` has its own record, built without the text. Nothing is
deleted by rotation: at :data:`MAX_HISTORY_BYTES` the file becomes a dated gzip
archive. ``aar-gateway history delete`` is the only thing that removes data.

A row is the request-log record of the turn's first request plus the message
(``prompt``, its ``prompt_hash``) and the folder it ran in (``cwd``) -- the same
fields the grader already reads from ``turns.jsonl``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from . import archive
from .config import gateway_home, iter_jsonl, read_preferences, safe_append_jsonl, save_preferences

log = logging.getLogger(__name__)

#: rotate the history into a gzip archive past this size
MAX_HISTORY_BYTES = 200 * 1024 * 1024
#: the preference that turns it off (``aar-gateway history off``); on when absent
PREFERENCE = "keep_history"

_state: dict = {"stamp": None, "on": True}


def history_dir() -> Path:
    return gateway_home() / "history"


def history_path() -> Path:
    return history_dir() / "history.jsonl"


def archives() -> list[Path]:
    return archive.archives_of(history_path())


def enabled() -> bool:
    """The user's choice (default on), re-read when the preferences file changes (the CLI
    writes it; the gateway runs in another process)."""
    from .config import preferences_path

    path = preferences_path()
    try:
        st = path.stat()
    except OSError:
        return True
    stamp = (st.st_mtime_ns, st.st_size)
    if stamp != _state["stamp"]:
        try:
            on = bool(read_preferences().get(PREFERENCE, True))
        except OSError:
            on = True
        _state.update(stamp=stamp, on=on)
    return _state["on"]


def set_enabled(on: bool) -> None:
    save_preferences(**{PREFERENCE: bool(on)})
    _state["stamp"] = None


def build_row(record: Mapping, prompt: str, cwd: str | None) -> dict:
    """``record`` (the turn's request-log record) with the message and where it ran."""
    return {**record, "prompt": prompt, "prompt_hash": hashlib.sha1(prompt.encode()).hexdigest()[:16], "cwd": cwd}


def record(log_record: Mapping, prompt: str | None, cwd: str | None) -> None:
    """Keep one user turn. Never raises: losing a row must not cost the request."""
    if not prompt or not enabled():
        return
    try:
        append(build_row(log_record, prompt, cwd))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not keep the history row: %s", type(exc).__name__)


def append(row: Mapping) -> bool:
    path = history_path()
    if not path.exists():
        _create_private(path)
    try:
        if path.stat().st_size > MAX_HISTORY_BYTES:
            archive.run_in_background(archive.gzip_archive, archive.rotate_to_archive(path))
            _create_private(path)
    except OSError as exc:  # e.g. another program has it open: append to it, rotate next time
        log.warning("could not rotate %s: %s", path, type(exc).__name__)
    return safe_append_jsonl(path, dict(row))


def _create_private(path: Path) -> None:
    """The folder and file exist and, where the OS has modes, only this user can read them."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(path.parent, 0o700)
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
    except OSError as exc:
        log.warning("could not create %s: %s", path, type(exc).__name__)


def read(n: int | None = None) -> list[dict]:
    """The newest ``n`` rows, newest first (all rows, oldest first, when ``n`` is None)."""
    if n is None:
        out: list[dict] = []
        for path in archives():
            out.extend(iter_jsonl(path))
        if history_path().exists():
            out.extend(iter_jsonl(history_path()))
        return out
    out = list(iter_jsonl(history_path())) if history_path().exists() else []
    for path in reversed(archives()):
        if len(out) >= n:
            break
        out = list(iter_jsonl(path)) + out
    return list(reversed(out[-n:]))


def stats(*, rows: bool = True) -> dict:
    """Size and (unless ``rows=False``, which skips reading the files) the row count."""
    files = [*archives(), *([history_path()] if history_path().exists() else [])]
    count = None
    if rows:
        count = 0
        for path in files:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rb") as f:
                count += sum(1 for line in f if line.strip())
    return {"rows": count, "bytes": sum(p.stat().st_size for p in files), "files": len(files),
            "enabled": enabled(), "path": str(history_path())}


def export(dest: Path) -> int:
    """Write every row, oldest first, to one plain jsonl file; the number written."""
    n = 0
    with Path(dest).open("w", encoding="utf-8") as out:
        for row in read():
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def delete() -> int:
    """Remove the whole history (live file, archives, their summaries); the number of rows removed."""
    archive.join_background(30)
    removed = stats()["rows"]
    for path in history_dir().glob("*") if history_dir().exists() else []:
        if path.is_file():
            path.unlink()
    return removed
