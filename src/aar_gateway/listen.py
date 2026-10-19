"""Where the gateway listens: ``AAR_GATEWAY_HOST`` / ``AAR_GATEWAY_PORT``, read and checked in one place.

Standard library only (the Claude Code hook and status line import it and must start fast).
"""

from __future__ import annotations

import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


def listen_address() -> tuple[str, int]:
    """``(host, port)`` from the environment. ``ValueError`` names the variable when the port is not 1-65535."""
    host = os.environ.get("AAR_GATEWAY_HOST") or DEFAULT_HOST
    raw = os.environ.get("AAR_GATEWAY_PORT") or str(DEFAULT_PORT)
    try:
        port = int(raw)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError(f"AAR_GATEWAY_PORT must be a number from 1 to 65535, got {raw!r}") from None
    return host, port


def base_url() -> str:
    host, port = listen_address()
    return f"http://{host}:{port}"
