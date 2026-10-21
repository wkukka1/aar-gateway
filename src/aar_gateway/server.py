"""FastAPI process for :mod:`aar_gateway` -- see that package's docstring
for what this exposes and why, and ``docs/gateway.md`` for the walkthrough.
The command line (``aar-gateway``) is :mod:`aar_gateway.cli`.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from . import interceptor, listen, share
from .config import graded_path, models_dir, pid_file


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load eagerly rather than on the first request -- a bad config (missing
    API key, unknown lane, router run that won't load) should fail the
    process at boot, in the logs, not surface on someone's first real call."""
    from .config import atomic_write_text, config_path, load_settings
    from .routing import build_picker

    settings = load_settings(config_path())
    # the text encoder's model, kept with the gateway's files: fastembed's
    # default is the system temp folder, which macOS clears, and a missing
    # model means a download before the gateway can answer
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(models_dir()))
    use_graded = settings["router"].get("use_graded", True)
    router_error = None
    try:
        picker = build_picker(settings, graded=_graded_predictor(settings),
                              graded_file=graded_path() if use_graded else None)
    except Exception as exc:  # noqa: BLE001 -- no router must not mean no Claude Code
        # The router (its data, encoder or run) failed to load: forward
        # everything as asked instead of refusing to start, and say why.
        picker = None
        router_error = f"{type(exc).__name__}: {exc}"[:400]
        print(f"router not loaded, forwarding requests unrouted: {router_error}", file=sys.stderr)
    interceptor.configure(settings, picker, router_error=router_error)
    pid_file = _pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pid_file, str(os.getpid()))
    try:
        yield
    finally:
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()
        share.shutdown()  # write the records still waiting
        await interceptor.aclose()


def _pid_file() -> Path:
    return pid_file()


def _graded_predictor(settings: dict):
    """The user's graded turns (``aar-gateway-grade``), when the router is on --
    built here, not in :mod:`.routing`, so tests can inject their own."""
    rcfg = settings["router"]
    if not rcfg.get("enabled") or not rcfg.get("use_graded", True):
        return None
    from ._core.encoder import EncoderConfig, TextEncoder, light_backend
    from .graded import graded_predictor

    encoder = TextEncoder(EncoderConfig(name="gateway-graded", backend=light_backend()))
    return graded_predictor(graded_path(), encoder.encode,
                            min_prompts=int(rcfg.get("min_graded_prompts", 20)))


app = FastAPI(title="AAR gateway", lifespan=_lifespan)

#: names this gateway answers to; anything else in ``Host`` is a DNS-rebinding
#: page (a site whose domain resolves to 127.0.0.1). ``AAR_GATEWAY_ALLOWED_HOSTS``
#: adds more, comma-separated, e.g. when serving on a LAN address.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


def _allowed_hosts() -> set[str]:
    extra = os.environ.get("AAR_GATEWAY_ALLOWED_HOSTS", "")
    names = _LOCAL_HOSTS | {h.strip().lower() for h in extra.split(",") if h.strip()}
    try:
        bound = listen.listen_address()[0].lower()
    except ValueError:
        return names
    if bound not in ("0.0.0.0", "::"):  # a wildcard says nothing about which names clients use
        names |= {bound, f"[{bound}]" if ":" in bound and not bound.startswith("[") else bound}
    return names


def _hostname(netloc: str) -> str:
    netloc = netloc.strip().lower()
    if netloc.startswith("["):  # [::1]:8787
        return netloc.split("]", 1)[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


@app.middleware("http")
async def _local_callers_only(request: Request, call_next):
    """Claude Code, the CLI and the dashboard -- never a web page. A page in
    the browser can send requests to localhost but can't make them look
    same-origin: its ``Origin`` header names its own site (rejected unless it
    is this gateway's own dashboard), and a DNS-rebinding page arrives with
    its own domain in ``Host``. Without this, any site the user visits could
    switch routing off or spend an ``openai``-lane API key through the
    gateway."""
    host = request.headers.get("host", "")
    if _hostname(host) not in _allowed_hosts():
        return interceptor.error_response(403, f"host {host!r} is not this gateway", "forbidden")
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/").lower() not in (f"http://{host.lower()}", f"https://{host.lower()}"):
        return interceptor.error_response(403, f"requests from {origin} are not accepted", "forbidden")
    return await call_next(request)


@app.get("/api/events")
async def events(request: Request):
    """Server-sent events for the dashboard: ``update`` the moment a request
    is logged, a comment every 15 s to keep the connection open."""
    import asyncio

    from fastapi.responses import StreamingResponse

    async def stream():
        # subscribed here, not before the response: a client that drops before
        # the body starts never runs this generator, so nothing would unsubscribe
        queue = interceptor.subscribe()
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    await asyncio.wait_for(queue.get(), timeout=15)
                    yield "event: update\ndata: {}\n\n"
                except TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            interceptor.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache"})


@app.get("/health")
async def health():
    return {"status": "ok", "pid": os.getpid(), "router": interceptor.router_status()}


@app.post("/api/shutdown")
async def shutdown():
    """Ask this server to exit cleanly (``aar-gateway stop`` on Windows, which has
    no SIGTERM: killing the process skips the lifespan cleanup). Callable only
    as the CLI is: the Host/Origin middleware refuses a web page."""
    running = getattr(app.state, "uvicorn", None)
    if running is None:
        raise HTTPException(503, "not running under `aar-gateway serve`")
    running.should_exit = True
    return {"status": "stopping"}


@app.get("/stats")
def stats():  # plain def: FastAPI runs it in a worker thread, off the event loop
    return interceptor.summarize()


@app.get("/api/statusline")
def statusline(session_id: str = ""):
    return interceptor.session_summary(session_id)


async def _json_object(request: Request) -> dict:
    """The request's JSON body, which must be an object: else a 400, not a 500."""
    try:
        body = await request.json()
    except ValueError as exc:  # includes undecodable bytes
        raise HTTPException(400, "send a JSON object") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "send a JSON object")
    return body


@app.post("/api/mod-record")
async def mod_record(request: Request):
    """A request the Claude Code mod saw that didn't come through this proxy:
    its token usage, priced and logged like the proxy's own."""
    body = await _json_object(request)
    return await run_in_threadpool(interceptor.record_from_mod, body)


@app.get("/api/session-requests")
def session_requests(session_id: str = "", limit: int = 40):  # plain def: reads the log off the event loop
    """One session's recent requests, for the Claude Code mod's pane."""
    return interceptor.session_requests(session_id, limit)


@app.post("/api/route")
async def route(request: Request):
    """Which model the router picks for a prompt, for the Claude Code mod
    that switches models itself (``{"model": null}``: keep the one asked for)."""
    body = await _json_object(request)
    try:
        est = int(body.get("est_prompt_tokens") or 0)
    except (TypeError, ValueError):
        est = 0
    return await run_in_threadpool(interceptor.route_text, str(body.get("text") or ""),
                                   str(body.get("model") or ""), str(body.get("session_id") or ""), est)


@app.post("/api/subagent-spawn")
async def subagent_spawn(request: Request):
    """Asked by the ``PreToolUse`` hook (``aar-gateway hook``) before Claude
    Code spawns a subagent: which model it should run on."""
    body = await _json_object(request)
    return interceptor.record_spawn(str(body.get("session_id") or ""), body.get("subagent_type"),
                                    str(body.get("prompt") or ""), body.get("model"))


@app.get("/api/preferences")
async def get_preferences():
    return interceptor.preferences()


@app.post("/api/preferences")
async def update_preferences(request: Request):
    body = await _json_object(request)
    try:
        if "mode" in body:
            return interceptor.set_routing(body["mode"], body.get("pinned_model"))
        if "turn_smoothing" in body:
            return interceptor.set_turn_smoothing(bool(body["turn_smoothing"]))
        if "allow_upgrade" in body:
            return interceptor.set_allow_upgrade(bool(body["allow_upgrade"]))
        if "role" in body:
            return interceptor.set_role_model(body["role"], body.get("model"))
    except (TypeError, ValueError) as exc:  # TypeError: a list or object where a name or number belongs
        raise HTTPException(400, str(exc)) from exc
    margin = body.get("quality_margin")
    try:
        if body.get("preset") is not None:
            if body["preset"] not in interceptor.PRESETS:
                raise HTTPException(400, f"unknown preset {body['preset']!r}")
            margin = interceptor.PRESETS[body["preset"]]
        if margin is None:
            raise HTTPException(400, "send quality_margin or preset")
        return interceptor.set_quality_margin(margin)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


def _no_credentials() -> JSONResponse:
    return interceptor.error_response(
        401, "no credentials: the gateway forwards the client's own Anthropic login "
             "(run Claude Code through it), or name a model on an `openai` lane", "authentication_error")


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    routed = await interceptor.handle_messages(request)
    return routed if routed is not None else _no_credentials()


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def anthropic_passthrough(path: str, request: Request):
    """The rest of the Anthropic API Claude Code calls besides
    ``/v1/messages`` (e.g. ``/v1/messages/count_tokens``) -- forwarded on the
    client's own credentials; registered last so the routes above win."""
    if not interceptor.has_client_credentials(request):
        return _no_credentials()
    return await interceptor.forward(request, f"v1/{path}")


def main(argv: list[str] | None = None) -> int:
    """The command line lives in :mod:`aar_gateway.cli`; kept here because
    Claude Code sessions started before it moved call
    ``python -m aar_gateway.server hook`` / ``statusline``."""
    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
