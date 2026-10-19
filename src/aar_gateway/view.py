"""A read-only dashboard for other devices (a phone on your Tailscale network).

``aar-gateway view`` serves the dashboard and its read-only data on one
address you choose, usually this computer's Tailscale address. It is a
separate, minimal app: it loads no router, writes no pid file, and changes
nothing. Only these requests are answered:

    GET /dashboard, /stats, /health, /api/preferences

Everything else is refused: any POST (so routing settings can't be changed
from the phone), any other path, and any request whose Host header isn't
the address it was started on or this computer. The dashboard's controls
will show an error on the phone view; the numbers are what it's for.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from . import interceptor
from .config import load_settings

READ_ONLY = {("GET", "/dashboard"), ("GET", "/stats"), ("GET", "/health"), ("GET", "/api/preferences"),
             ("HEAD", "/dashboard"), ("HEAD", "/stats"), ("HEAD", "/health"), ("HEAD", "/api/preferences")}
_LOCAL = {"127.0.0.1", "localhost", "[::1]", "::1"}


def _hostname(netloc: str) -> str:
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc.split("]", 1)[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def build_app(view_host: str, config_path: str | None = None) -> FastAPI:
    """The app for a view on ``view_host``. ``config_path``: the gateway config,
    for the strongest model's name and prices (no router is loaded)."""
    allowed = _LOCAL | {view_host.lower()}
    interceptor.configure(load_settings(config_path), None, router_error="read-only view: no router loaded")
    app = FastAPI(title="AAR gateway view")

    @app.middleware("http")
    async def only_read_only(request: Request, call_next):
        host = _hostname(request.headers.get("host", ""))
        if host not in allowed:
            return PlainTextResponse("refused: not this address", status_code=403)
        if (request.method, request.url.path) not in READ_ONLY:
            return PlainTextResponse("refused: read-only view", status_code=405)
        return await call_next(request)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/stats")
    async def stats():
        return interceptor.summarize()

    @app.get("/health")
    async def health():
        return {"status": "ok", "view": True}

    @app.get("/api/preferences")
    async def preferences():
        return interceptor.preferences()

    return app
