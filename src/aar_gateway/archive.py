"""Rotating a jsonl file into dated, gzipped archives that are never deleted.

The request log and the on-device history both keep everything: at a size limit the
live file moves to ``<stem>-<UTC time>.jsonl`` and is gzipped in the background.
The gzip is complete only once renamed from ``.part``; until then the plain file *is*
the archive (:func:`archives_of` ignores a plain file once its gzip exists), so a
crash at any point loses nothing and the next start finishes the job.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: threads gzipping a rotated file (joined by tests; daemon threads, so exit never waits for one)
_threads: list[threading.Thread] = []


def run_in_background(target: Callable, *args) -> None:
    thread = threading.Thread(target=target, args=args, name="aar-archive", daemon=True)
    _threads.append(thread)
    thread.start()


def join_background(timeout: float | None = None) -> None:
    for thread in list(_threads):
        thread.join(timeout)
    _threads[:] = [t for t in _threads if t.is_alive()]


def archives_of(path: Path, *, legacy: bool = False) -> list[Path]:
    """The archives of ``path``, oldest first: ``<stem>-<UTC time>.jsonl`` and its
    ``.jsonl.gz`` (plus, with ``legacy``, the single ``<name>.1`` older versions kept).
    Unfinished ``.part`` files and the summaries stored beside archives are not archives."""
    dated = sorted((p for p in path.parent.glob(f"{path.stem}-*") if p.name.endswith((".jsonl", ".jsonl.gz"))),
                   key=lambda p: p.name.removesuffix(".gz").removesuffix(".jsonl"))  # rotation order
    done = {p.name for p in dated}
    # a plain archive whose gzip exists is a compression that had not yet removed it: count the gzip only
    dated = [p for p in dated if not (p.name.endswith(".jsonl") and p.name + ".gz" in done)]
    old = path.with_name(path.name + ".1")
    return ([old] if legacy and old.exists() else []) + dated


def rotate_to_archive(path: Path) -> Path:
    """Move ``path`` to a new dated name (never over an existing file). The caller then
    gzips it (:func:`gzip_archive`); the next append to ``path`` starts a new file."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")  # microseconds: names sort in rotation order
    dest = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    n = 0
    while dest.exists() or dest.with_name(dest.name + ".gz").exists():
        n += 1
        dest = path.with_name(f"{path.stem}-{stamp}-{n}{path.suffix}")
    path.replace(dest)
    return dest


def gzip_archive(plain: Path, *, each: Callable[[dict], None] | None = None,
                 done: Callable[[Path], None] | None = None) -> Path | None:
    """Gzip a rotated file, calling ``each(record)`` for every record on the way (one
    pass, so a rollup is built without reading it again), then ``done(gz)``, then remove
    the plain file. ``None`` if it could not (disk full ...): the plain file stays and
    is retried at the next start."""
    gz = plain.with_name(plain.name + ".gz")
    part = gz.with_name(gz.name + ".part")
    try:
        with plain.open("rb") as src, gzip.open(part, "wb") as dst:
            for line in src:
                dst.write(line)
                if each is not None:
                    try:
                        each(json.loads(line))
                    except ValueError:
                        continue
        os.replace(part, gz)
        if done is not None:
            done(gz)
        plain.unlink()
    except OSError as exc:
        log.warning("could not compress %s: %s", plain, type(exc).__name__)
        return None
    return gz
