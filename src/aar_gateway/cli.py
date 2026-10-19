"""``aar-gateway``: run Claude Code through the gateway, and manage the
gateway process.

    aar-gateway code [claude args...]   Claude Code through the gateway (starts it if needed)
    aar-gateway start | stop | restart  the gateway in the background
    aar-gateway status                  running? router loaded? where are its files?
    aar-gateway logs [-n N] [-f]        the gateway's own log
    aar-gateway setup [--router FILE|URL]   first-time setup: the router + a config
    aar-gateway doctor                  check the setup, with a fix for each problem
    aar-gateway local [add|remove] M    optional: a local model through Ollama
    aar-gateway history [on|off|show|export|delete]   the messages kept on this computer
    aar-gateway config                  which config is in use
    aar-gateway dashboard               open the dashboard
    aar-gateway service install|uninstall|status   start at login (launchd / systemd)
    aar-gateway serve                   the gateway in the foreground
    aar-gateway agents [--install DIR]  the subagent kinds as --agents JSON / agent files

``hook`` and ``statusline`` are called by Claude Code itself. They import
nothing but the standard library: the hook runs on every subagent spawn and
must not wait for the web server's imports.

Environment: ``AAR_GATEWAY_HOST`` / ``AAR_GATEWAY_PORT`` (default
``127.0.0.1:8787``), ``AAR_GATEWAY_CONFIG`` (default
``~/.aar-gateway/config.yaml`` when it exists), ``AAR_GATEWAY_LOG``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import listen

SERVICE_LABEL = "com.aar.gateway"
#: the router ``aar-gateway setup`` installs when not given one: a trained
#: network (weights only, 0.4 MB) shipped inside the package.
#: ``AAR_GATEWAY_ROUTER_URL`` points setup at another file or URL instead.
BUNDLED_ROUTER = Path(__file__).with_name("routers") / "llmrouterbench-mlp.npz"
WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------- #
# where things are                                                            #
# --------------------------------------------------------------------------- #
def _home() -> Path:
    from .config import gateway_home

    return gateway_home()


def _log_path() -> Path:
    from .config import log_path

    return log_path()


def _config_path() -> Path | None:
    from .config import config_path

    return config_path()


def _pid_file() -> Path:
    from .config import pid_file

    return pid_file()


def _log_file() -> Path:
    return _home() / "gateway.log"


def _listen() -> tuple[str, int]:
    """Where the gateway listens, or a one-line exit for a bad AAR_GATEWAY_PORT."""
    try:
        return listen.listen_address()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def _base_url() -> str:
    host, port = _listen()
    return f"http://{host}:{port}"


def _src() -> str:
    return str(Path(__file__).resolve().parents[1])


def _src_env() -> dict:
    """This package importable in a child process even when site-packages
    ``.pth`` files are skipped -- macOS marks them hidden in iCloud-synced
    folders, and Python 3.13+ ignores hidden ``.pth`` files."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_src(), os.environ.get("PYTHONPATH")]))}


def _get(url: str, timeout: float = 1.0) -> dict | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return None


def _healthy(url: str) -> bool:
    return _get(url + "/health") is not None


def _running_pid() -> int | None:
    """The gateway's pid. The gateway names itself at ``/health``, and that
    answer wins over the pid file. If nothing answers, the file's pid counts
    only while that process really is a gateway (it may be hung): a crash
    leaves a stale file, and its pid can since belong to an unrelated process
    that ``stop`` must never signal. (On Windows ``os.kill(pid, 0)`` would
    *terminate* the process, so liveness is never probed that way.)"""
    answering = (_get(f"{_base_url()}/health") or {}).get("pid")
    if isinstance(answering, int):
        return answering
    try:
        filed = int(_pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    return filed if _is_gateway_process(filed) else None


def _is_gateway_process(pid: int) -> bool:
    """Whether ``pid`` is a running ``aar_gateway`` process, by its command
    line (``ps``; Windows has no such cheap probe and relies on ``/health``)."""
    if WINDOWS:
        return False
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "aar_gateway" in out


# --------------------------------------------------------------------------- #
# process management                                                          #
# --------------------------------------------------------------------------- #
def _ensure_running(url: str, *, quiet: bool = False, timeout: float = 90) -> bool:
    """Start the gateway in the background if nothing answers at ``url``,
    waiting up to ``timeout`` seconds for it. ``True`` if this call started it."""
    if _healthy(url):
        return False
    if _service_running():
        # the login service is starting it (at login the router takes a few
        # seconds to load): a second one here would only fail on the port
        if _wait_healthy(url, min(30, timeout)):
            return False
    elif _service_start():
        # started by the login service, so it restarts the gateway if it crashes
        if not quiet:
            print(f"starting the gateway at {url} (login service) ...", file=sys.stderr)
        if _wait_healthy(url, timeout):
            return True
    log = _log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    _rotate_log(log)
    # detached, so it outlives this command and the terminal it ran in
    detach = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
              if WINDOWS else {"start_new_session": True})
    with log.open("ab") as out:
        proc = subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "serve"],
                                stdout=out, stderr=out, stdin=subprocess.DEVNULL, env=_src_env(), **detach)
    if not quiet:
        print(f"starting the gateway at {url} (log: {log}) ...", file=sys.stderr)
    if _wait_healthy(url, timeout, proc):
        return True
    if proc.poll() is not None and not _healthy(url):
        raise SystemExit(f"the gateway exited while starting (code {proc.returncode}) -- see `aar-gateway logs`")
    raise SystemExit(f"the gateway did not come up within {timeout:.0f}s -- see `aar-gateway logs`")


def _wait_healthy(url: str, seconds: float, proc: subprocess.Popen | None = None) -> bool:
    """Whether ``url`` answers within ``seconds``; gives up early once
    ``proc`` (the gateway being started) has exited."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _healthy(url):
            return True
        if proc is not None and proc.poll() is not None:
            return _healthy(url)  # another gateway may have taken the port meanwhile
        time.sleep(0.5)
    return False


def _request_shutdown() -> bool:
    """Ask the running gateway to exit by itself; whether it said it would."""
    import urllib.request

    request = urllib.request.Request(_base_url() + "/api/shutdown", data=b"{}", method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as resp:
            return resp.status == 200
    except OSError:
        return False


def _stop() -> bool:
    """Stops the gateway this machine is running, however it was started.
    ``True`` if one was stopped."""
    import signal

    pid = _running_pid()
    if pid is None:
        _pid_file().unlink(missing_ok=True)
        print("the gateway is not running")
        return False
    # Windows has no SIGTERM -- os.kill there is TerminateProcess, which skips the
    # gateway's cleanup (open streams unlogged, the share queue unflushed) -- so
    # it is asked to exit itself first, and only killed if it does not.
    if not (WINDOWS and _request_shutdown() and _wait_exit(pid, 10)):
        os.kill(pid, signal.SIGTERM)  # on Windows: TerminateProcess
        # gone, not just quiet: a gateway closes its port first, then waits for
        # open connections (the dashboard's live stream never ends by itself)
        if not _wait_exit(pid, 10) and not WINDOWS:
            os.kill(pid, signal.SIGKILL)
            _wait_exit(pid, 5)
    if _healthy(_base_url()):
        print(f"sent stop to pid {pid}; something is still answering on {_base_url()}")
        return False
    _pid_file().unlink(missing_ok=True)  # a terminated process can't remove its own
    print(f"stopped the gateway (pid {pid})")
    return True


def _restart() -> None:
    """Stop the gateway (if one runs) and start it again, quietly."""
    _stop()
    _ensure_running(_base_url(), quiet=True)


def _wait_exit(pid: int, seconds: float) -> bool:
    """Whether ``pid`` exited within ``seconds``. Windows has no signal-free
    probe (see :func:`_running_pid`); there the port going quiet is the sign."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if WINDOWS:
            if not _healthy(_base_url()):
                return True
        else:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:  # our own child: reap it
                    return True
            except ChildProcessError:
                pass  # not our child: the usual case
            try:
                os.kill(pid, 0)
            except OSError:
                return True
        time.sleep(0.2)
    return False


def _status(as_json: bool = False) -> int:
    url = _base_url()
    health = _get(url + "/health")
    pid = _running_pid()
    cfg = _config_path()
    info = {"running": health is not None, "url": url, "pid": pid, "config": str(cfg) if cfg else None,
            "log": str(_log_file()), "requests_log": str(_log_path()),
            "dashboard": url + "/dashboard"}
    if health:
        router = health.get("router") or {}
        info["router"] = {"enabled": router.get("enabled"), "error": router.get("error"),
                          "kind": router.get("kind")}
        stats = _get(url + "/stats", timeout=3) or {}
        totals = stats.get("totals") or {}
        info["requests"] = totals.get("requests")
        base = totals.get("cost_if_requested_usd") or 0
        info["plan_usage_saved"] = round(1 - totals.get("cost_usd", 0) / base, 3) if base else None
        from . import history

        info["history"] = history.stats(rows=False)
        if pid and _pid_file().exists():  # the pid can come from /health with no file
            info["up_since"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(_pid_file().stat().st_mtime))
    if as_json:
        print(json.dumps(info, indent=2))
        return 0 if health else 1
    if not health:
        print(f"● stopped        nothing answers at {url}")
        print(f"  config         {info['config'] or 'none (defaults; router off)'}")
        print(f"  log            {info['log']}")
        print("  start it with  aar-gateway start   (or aar-gateway code)")
        return 1
    router = info["router"]
    state = ("on" if router["enabled"] else f"FAILED TO LOAD -- {router['error']}" if router["error"]
             else "off (enable it in the config)")
    saved = info.get("plan_usage_saved")
    print(f"● running        {url}" + (f"  (pid {pid}, since {info.get('up_since')})" if pid else ""))
    print(f"  router         {state}")
    print(f"  config         {info['config'] or 'none (defaults)'}")
    print(f"  requests       {info.get('requests', 0)}" +
          (f"  ·  {saved:.0%} less plan usage than without the router" if saved is not None else ""))
    print(f"  dashboard      {info['dashboard']}")
    print(f"  logs           {info['log']}")
    return 0


def _text(raw: bytes) -> str:
    """Log bytes as text; Windows writes \\r\\n, which print() would double up."""
    return raw.decode(errors="replace").replace("\r\n", "\n")


def _tail(path: Path, lines: int, block: int = 8192) -> list[bytes]:
    """The last ``lines`` lines of ``path`` (newlines kept), reading blocks from
    the end: the daemon's log can be large and ``logs`` wants only the end."""
    if lines <= 0:
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        data = b""
        while end > 0 and data.count(b"\n") <= lines:
            start = max(0, end - block)
            f.seek(start)
            data = f.read(end - start) + data
            end = start
    return data.splitlines(keepends=True)[-lines:]


#: the daemon's stdout/stderr file is set aside past this size when a gateway starts
MAX_GATEWAY_LOG_BYTES = 10 * 1024 * 1024


def _rotate_log(log: Path, max_bytes: int = MAX_GATEWAY_LOG_BYTES) -> None:
    """Keep one previous file (``.1``) so ``gateway.log`` cannot grow for ever
    across restarts. (A running daemon holds the file as its stdout, so it can
    only be moved aside between runs.)"""
    try:
        if log.stat().st_size > max_bytes:
            os.replace(log, log.with_name(log.name + ".1"))
    except OSError:
        pass  # missing, or held open elsewhere: append to it as before


def _logs(lines: int, follow: bool) -> int:
    log = _log_file()
    if not log.exists():
        print(f"no log yet ({log})")
        return 1
    sys.stdout.write(_text(b"".join(_tail(log, lines))))
    if not follow:
        return 0
    with log.open("rb") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = f.readline()
                if chunk:
                    sys.stdout.write(_text(chunk))
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            return 0


# --------------------------------------------------------------------------- #
# local models (opt-in)                                                       #
# --------------------------------------------------------------------------- #
OLLAMA_URL = os.environ.get("AAR_GATEWAY_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_INSTALL = {
    "darwin": "brew install ollama && brew services start ollama\n"
              "  (or download the app from https://ollama.com/download and open it)",
    "linux": "curl -fsSL https://ollama.com/install.sh | sh",
    "win32": "download and run the installer from https://ollama.com/download",
}


def _ollama_models() -> list[str] | None:
    """Models Ollama has pulled, or ``None`` when Ollama isn't running."""
    tags = _get(f"{OLLAMA_URL}/api/tags", timeout=2)
    return None if tags is None else [m.get("name", "") for m in tags.get("models") or []]


def _ollama_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def _write_file(path: Path, text: str) -> None:
    """Replace ``path`` atomically (config is imported lazily: it pulls in yaml)."""
    from .config import atomic_write_text

    atomic_write_text(path, text)


def _write_local_models(models: dict) -> None:
    import yaml

    from .config import local_models_path

    path = local_models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not models:
        path.unlink(missing_ok=True)
        return
    _write_file(path, "# Local models added with `aar-gateway local add` (Ollama). Remove one with\n"
                      "# `aar-gateway local remove <model>`, or delete this file to remove them all.\n"
                      + yaml.safe_dump({"models": models}, sort_keys=False))


def _local(argv: list[str]) -> int:
    """``aar-gateway local [add|remove] [model]``: opt-in local models via Ollama."""
    from .config import local_models

    action = argv[0] if argv else "status"
    configured = local_models()
    pulled = _ollama_models()
    installed = shutil.which("ollama") is not None or pulled is not None
    if action == "status":
        print(f"Ollama: {'running' if pulled is not None else 'installed, not running' if installed else 'not installed'}")
        if pulled:
            print("  models it has: " + ", ".join(pulled))
        print("local models in the gateway: " + (", ".join(configured) if configured else "none"))
        if not installed:
            print(f"\nto install Ollama: {OLLAMA_INSTALL.get(sys.platform, OLLAMA_INSTALL['linux'])}")
        print("\nadd one: aar-gateway local add llama3.1   (good at tool calls; 4.9 GB)")
        return 0
    if action not in ("add", "remove") or len(argv) < 2:
        raise SystemExit("usage: aar-gateway local [add|remove] <model>   (e.g. llama3.1)")
    name = _ollama_name(argv[1])
    if action == "remove":
        if name not in configured:
            print(f"{name} isn't a local model in the gateway")
            return 1
        del configured[name]
        _write_local_models(configured)
        print(f"removed {name} from the gateway (Ollama still has it: `ollama rm {name}` frees the disk)")
    else:
        if pulled is None:
            raise SystemExit("Ollama isn't running. " + (
                "Start it (open the Ollama app, or run `ollama serve`)." if installed else
                f"Install it first:\n  {OLLAMA_INSTALL.get(sys.platform, OLLAMA_INSTALL['linux'])}"))
        if name not in pulled:
            ollama = shutil.which("ollama")
            if ollama is None:
                raise SystemExit(f"Ollama doesn't have {name} yet and its command isn't on PATH: run `ollama pull {name}`")
            print(f"downloading {name} with Ollama (one-time) ...")
            if subprocess.run([ollama, "pull", name]).returncode != 0:
                raise SystemExit(f"`ollama pull {name}` failed")
        configured[name] = {"lane": "openai", "base_url": f"{OLLAMA_URL}/v1", "proxy": None,
                            "context_window": 32768, "max_tools": 64, "prices": {"input": 0, "output": 0}}
        _write_local_models(configured)
        print(f"added {name}. The router never picks it on its own; use it by\n"
              f"  - pinning it on the dashboard (Advanced settings -> Pin a model), or\n"
              f"  - aar-gateway code --model {name}")
    if _healthy(_base_url()):
        _restart()
        print("restarted the gateway")
    return 0


# --------------------------------------------------------------------------- #
# setup                                                                       #
# --------------------------------------------------------------------------- #
CONFIG_TEMPLATE = """# AAR gateway config, written by `aar-gateway setup`.
# Every option is explained in gateway.example.yaml in the repository
# (https://github.com/wkukka1/aar-gateway) and in docs/gateway.md.
interceptor:
  router:
    enabled: {enabled}
    artifact: {artifact}      # the router file
    margin: 0.05              # cost preference; change it from the dashboard
"""


def _fetch_router(source: str, target: Path) -> dict:
    """Copy or download a router file to ``target``, checking it loads.
    Returns its header."""
    from ._core.artifacts import load_router_file

    tmp = target.with_suffix(".part")
    if source.startswith(("http://", "https://")):
        import urllib.request

        print(f"downloading the router from {source} ...")
        with urllib.request.urlopen(source, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise SystemExit(f"no router file at {path}")
        shutil.copyfile(path, tmp)
    try:
        payload, header = load_router_file(tmp)
    except Exception as exc:  # noqa: BLE001 -- anything unreadable is not a router file
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{source} is not a router file this gateway can load: {exc}") from exc
    tmp.replace(target)
    kind = "trained network" if header.get("format") == "mlp" else f"kNN, k={header.get('k')}"
    print(f"router installed: {target} ({kind}, {len(payload.model_ids)} models"
          + (f", {header['dataset']}" if header.get("dataset") else "") + ")")
    return header


def _setup(router: str | None, force: bool, enable: bool | None = False, share: bool | None = False) -> int:
    """Everything a new user needs: the router file and a config that uses it."""
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    router_file = home / "router.npz"
    source = router or os.environ.get("AAR_GATEWAY_ROUTER_URL") or (
        str(BUNDLED_ROUTER) if BUNDLED_ROUTER.exists() else "")
    if source:
        _fetch_router(source, router_file)
    elif router_file.exists():
        print(f"router already installed: {router_file}")
    else:
        print("no router file to install yet: the gateway will forward every request to the model you\n"
              "asked for until you run `aar-gateway setup --router <file or URL>`.")
    config = home / "config.yaml"
    if config.exists() and not force:
        print(f"kept your config: {config}  (--force rewrites it)")
    else:
        _write_file(config, CONFIG_TEMPLATE.format(enabled="true" if router_file.exists() else "false",
                                                   artifact=json.dumps(str(router_file))))
        print(f"wrote {config}")
    if _healthy(_base_url()):
        print("restarting the running gateway so it picks this up ...")
        _restart()
    print()
    if auto_enabled():
        print("plain `claude` already goes through the gateway (`aar-gateway disable` to undo)")
    elif enable is None and sys.stdin.isatty():
        # opt in: only a yes changes Claude Code's settings
        answer = input("Also use the gateway in every Claude Code session, so plain `claude` saves too?\n"
                       "(otherwise start Claude Code with `aar-gateway code`) [y/N] ")
        enable = answer.strip().lower() in ("y", "yes")
    if enable and not auto_enabled():
        try:
            _enable()
        except SystemExit as exc:  # another base URL already set: say so, finish setup
            print(exc)
    elif not auto_enabled():
        print("plain `claude` is unchanged: start Claude Code with `aar-gateway code`\n"
              "(or run `aar-gateway enable` once to use the gateway in every session)")
    from . import share as sharing

    print()
    print(HISTORY_NOTE)
    if share is None and sys.stdin.isatty() and not sharing.consent_path().exists():
        print()
        share = input(SHARE_QUESTION).strip().lower() in ("y", "yes")  # opt in: only a yes
    if share is not None:
        _share(["on" if share else "off"])
    print()
    return _doctor()


# --------------------------------------------------------------------------- #
# doctor                                                                      #
# --------------------------------------------------------------------------- #
def _doctor() -> int:
    """Checks, each with a fix: the claude CLI, the config, the router, the
    port, the log folder."""
    problems = 0

    def report(ok: bool | None, what: str, detail: str = "", fix: str = "") -> None:
        nonlocal problems
        mark = {True: "✓", False: "✗", None: "!"}[ok]
        problems += ok is False
        print(f"{mark} {what}" + (f": {detail}" if detail else ""))
        if fix and ok is not True:
            print(f"    fix: {fix}")

    claude = shutil.which("claude")
    if claude:
        version = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        report(True, "Claude Code installed", version)
    else:
        report(False, "Claude Code installed", "`claude` is not on PATH",
               "install Claude Code (https://claude.com/claude-code) and run `claude` once to log in")

    cfg = _config_path()
    settings = None
    try:
        from .config import load_settings

        settings = load_settings(cfg)
        report(True if cfg else None, "config", str(cfg) if cfg else "none found, using the defaults (router off)",
               f"put one at {_home() / 'config.yaml'} (see gateway.example.yaml in the repository) or set AAR_GATEWAY_CONFIG")
    except Exception as exc:  # noqa: BLE001 -- doctor reports whatever is wrong
        report(False, "config", f"{cfg}: {exc}", "fix the error above, or move the file away to use the defaults")

    artifact = (settings or {}).get("router", {}).get("artifact")
    if artifact:
        path = Path(artifact).expanduser()
        report(path.exists(), "router file", str(path) if path.exists() else f"{path} is missing",
               "aar-gateway setup --router <file or URL>")

    url = _base_url()
    health = _get(url + "/health")
    if health:
        router = health.get("router") or {}
        if router.get("enabled"):
            report(True, "router", f"{router.get('kind')} loaded from {router.get('source')}")
        elif router.get("error"):
            report(False, "router", router["error"],
                   "check router.artifact in the config points at a router file (`aar-gateway setup` installs one), then `aar-gateway restart`")
        else:
            report(None, "router", "off: requests go through unrouted", "set interceptor.router.enabled: true")
        report(True, "gateway running", url)
    else:
        if settings and settings["router"].get("enabled"):
            report(None, "router", "enabled in the config; it loads when the gateway starts",
                   "run `aar-gateway start`, then `aar-gateway doctor` again to check it loaded")
        import socket

        with socket.socket() as sock:
            sock.settimeout(0.5)
            taken = sock.connect_ex(_listen()) == 0
        if taken:
            report(False, "port", f"something else is listening on {url}",
                   "stop it, or pick another port with AAR_GATEWAY_PORT")
        else:
            report(None, "gateway running", "not running", "aar-gateway start")

    home = _home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        report(True, "log folder writable", str(home))
    except OSError as exc:
        report(False, "log folder writable", f"{home}: {exc}", "set AAR_GATEWAY_LOG to a writable path")

    if _suspended_file().exists():
        report(False, "every Claude Code session",
               "the gateway stopped answering, so sessions talk to Anthropic directly (no routing)",
               "aar-gateway start  -- routing comes back on with the next message")
    elif not auto_enabled():
        report(None, "every Claude Code session", "off: only `aar-gateway code` uses the gateway",
               "aar-gateway enable")
    elif _healthy(url):
        report(True, "every Claude Code session", "plain `claude` goes through the gateway")
    else:
        report(False, "every Claude Code session", "on, but the gateway isn't running, so `claude` can't connect",
               "aar-gateway start  (or `aar-gateway disable` to talk to Anthropic directly)")

    from . import share as sharing

    if not sharing.enabled():
        report(True, "usage sharing", "off (optional; `aar-gateway share` explains)")
    elif not sharing.endpoint():
        report(True, "usage sharing", f"on, {len(sharing.queued())} record(s) kept locally; no collection endpoint yet")
    else:
        last = sharing.state().get("last_error")
        report(not last, "usage sharing", f"on, sent to {sharing.endpoint()}" + (f"; last upload failed ({last})" if last else ""),
               "check the address: aar-gateway share endpoint <https url>, then aar-gateway share upload")
    print("\nall good" if problems == 0 else f"\n{problems} problem(s) to fix")
    return 0 if problems == 0 else 1


# --------------------------------------------------------------------------- #
# start at login                                                              #
# --------------------------------------------------------------------------- #
def _service_env() -> dict[str, str]:
    env = {"PYTHONPATH": _src()}
    cfg = _config_path()
    if cfg:
        env["AAR_GATEWAY_CONFIG"] = str(cfg.resolve())
    for key in ("AAR_GATEWAY_HOST", "AAR_GATEWAY_PORT", "AAR_GATEWAY_LOG", "AAR_GATEWAY_ALLOWED_HOSTS"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def launchd_plist() -> str:
    """macOS: a LaunchAgent that starts the gateway at login and restarts it
    if it crashes -- but not after ``aar-gateway stop`` (a clean exit)."""
    import plistlib

    log = str(_log_file())
    return plistlib.dumps({
        "Label": SERVICE_LABEL,
        "ProgramArguments": [sys.executable, "-m", "aar_gateway.cli", "serve"],
        "EnvironmentVariables": _service_env(),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }).decode()


def systemd_unit() -> str:
    """Linux: a user unit, restarted on failure but not after a clean stop."""
    env = "\n".join(f"Environment={shlex.quote(f'{k}={v}')}" for k, v in _service_env().items())
    return (f"[Unit]\nDescription=AAR gateway (routes Claude Code)\n\n[Service]\n"
            f"ExecStart={shlex.quote(sys.executable)} -m aar_gateway.cli serve\n{env}\n"
            f"Restart=on-failure\n\n[Install]\nWantedBy=default.target\n")


def _service_paths() -> tuple[str, Path]:
    if sys.platform == "darwin":
        return "launchd", Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"
    if sys.platform.startswith("linux"):
        return "systemd", Path.home() / ".config" / "systemd" / "user" / "aar-gateway.service"
    raise SystemExit(f"start-at-login isn't supported on {sys.platform}; run `aar-gateway start` instead")


def _service_serves_this_port(path: Path) -> bool:
    """Whether the installed login service runs the gateway this command
    talks to: the same port (a second gateway on another port, for a test
    or a second setup, must not wait on or start the service's)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    # launchd: <key>AAR_GATEWAY_PORT</key><string>N</string>; systemd: AAR_GATEWAY_PORT=N
    found = re.search(r"AAR_GATEWAY_PORT\D{0,40}?(\d+)", text)
    return (found.group(1) if found else "8787") == os.environ.get("AAR_GATEWAY_PORT", "8787")


def _service_start() -> bool:
    """Asks launchd/systemd to start the gateway, if its service is installed."""
    try:
        kind, path = _service_paths()
    except SystemExit:
        return False
    if not _service_serves_this_port(path):
        return False
    cmd = (["launchctl", "kickstart", f"gui/{os.getuid()}/{SERVICE_LABEL}"] if kind == "launchd"
           else ["systemctl", "--user", "start", "aar-gateway.service"])
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _service_running() -> bool:
    """Whether launchd/systemd has the gateway's process up (it may still be loading)."""
    try:
        kind, path = _service_paths()
    except SystemExit:  # no login service on this OS
        return False
    if not _service_serves_this_port(path):
        return False
    cmd = (["launchctl", "print", f"gui/{os.getuid()}/{SERVICE_LABEL}"] if kind == "launchd"
           else ["systemctl", "--user", "is-active", "aar-gateway.service"])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "state = running" in out if kind == "launchd" else out.strip() == "active"


def _service(action: str, run=subprocess.run, *, at_next_login: bool = False) -> int:
    """``at_next_login``: register it without starting it now, so a gateway
    that is already serving keeps serving -- no gap for sessions using it."""
    kind, path = _service_paths()
    if action == "status":
        print(f"{'installed' if path.exists() else 'not installed'} ({path})")
        return 0 if path.exists() else 1
    if action == "install":
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_file(path, launchd_plist() if kind == "launchd" else systemd_unit())
        if at_next_login:  # launchd loads ~/Library/LaunchAgents at login by itself
            if kind == "systemd":
                run(["systemctl", "--user", "daemon-reload"], check=True)
                run(["systemctl", "--user", "enable", "aar-gateway.service"], check=True)
            print(f"installed: the gateway starts at login ({path}); the running one keeps serving")
            return 0
        if _running_pid():
            _stop()  # the service starts its own
        if kind == "launchd":
            job = f"gui/{os.getuid()}/{SERVICE_LABEL}"
            run(["launchctl", "bootout", job], capture_output=True)
            # bootout returns before the old job is gone, and loading it
            # again meanwhile fails ("Bootstrap failed: 5")
            for _ in range(60):
                if run(["launchctl", "print", job], capture_output=True).returncode != 0:
                    break
                time.sleep(0.5)
            try:
                run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
            except subprocess.CalledProcessError:
                raise SystemExit(f"launchd wouldn't load {path}; try again in a few seconds, "
                                 f"or `aar-gateway start` for now") from None
        else:
            run(["systemctl", "--user", "daemon-reload"], check=True)
            run(["systemctl", "--user", "enable", "--now", "aar-gateway.service"], check=True)
        if not _wait_healthy(_base_url(), 90):
            print("the service didn't answer within 90s -- see `aar-gateway logs`")
        print(f"installed: the gateway now starts at login ({path})")
        return 0
    if action == "uninstall":
        if not path.exists():
            print("not installed")
            return 0
        if kind == "launchd":
            run(["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE_LABEL}"], capture_output=True)
        else:
            run(["systemctl", "--user", "disable", "--now", "aar-gateway.service"], capture_output=True)
        path.unlink()
        print(f"uninstalled: removed {path}; the gateway no longer starts at login")
        return 0
    raise SystemExit(f"unknown service action {action!r}")


# --------------------------------------------------------------------------- #
# Claude Code                                                                 #
# --------------------------------------------------------------------------- #
def _self_command() -> str:
    """How Claude Code calls back into this package (status line, hook): this
    Python, ``-m aar_gateway.cli``. Installed (uv, pip) that's enough; from
    a source checkout on macOS/Linux, PYTHONPATH points at it too -- Windows
    shells don't take the ``VAR=value cmd`` form, and there the package is
    expected to be installed."""
    if WINDOWS:
        return subprocess.list2cmdline([sys.executable, "-m", "aar_gateway.cli"])
    command = f"{shlex.quote(sys.executable)} -m aar_gateway.cli"
    if "site-packages" in _src():
        return command
    return f"PYTHONPATH={shlex.quote(_src())} {command}"


def _user_has_statusline() -> bool:
    for path in (Path.home() / ".claude" / "settings.json", Path(".claude") / "settings.json",
                 Path(".claude") / "settings.local.json"):
        try:
            if json.loads(path.read_text()).get("statusLine"):
                return True
        except (OSError, ValueError):
            continue
    return False


def _claude_command(url: str, claude_args: list[str]) -> tuple[list[str], dict]:
    """``claude`` pointed at the gateway. Two settings matter:

    * ``ENABLE_TOOL_SEARCH=true`` -- for any base URL that isn't Anthropic's,
      Claude Code stops loading MCP tools on demand and sends every tool
      schema with every request (~245k tokens instead of ~40k with many MCP
      servers: several times the cost, and compaction every turn). The
      gateway forwards tool-search requests unchanged, so turn it back on.
    * a status line showing which model actually answered -- the model
      itself can't say, it is told it is whichever model was asked for.
      Skipped when the user already has a status line, or with
      ``AAR_GATEWAY_STATUSLINE=0``.

    It also installs the subagent hook (``AAR_GATEWAY_HOOK=0`` to skip) and
    gives Claude Code the subagent kinds (``--agents``,
    :data:`~aar_gateway.subagents.ROLES`) -- unless the user passes
    ``--agents`` themselves, or with ``AAR_GATEWAY_AGENTS=0``.
    """
    env = {"ENABLE_TOOL_SEARCH": "true", **os.environ, "ANTHROPIC_BASE_URL": url}
    args = ["claude"]
    run = _self_command()
    settings: dict = {}
    enabled = auto_enabled()  # the user settings already have the hook and status line
    if not enabled and os.environ.get("AAR_GATEWAY_STATUSLINE", "1") != "0" and not _user_has_statusline():
        settings["statusLine"] = {"type": "command", "command": f"{run} statusline"}
    if not enabled and os.environ.get("AAR_GATEWAY_HOOK", "1") != "0":
        # sets each subagent's model as Claude Code spawns it (see aar_gateway.subagents)
        settings["hooks"] = {"PreToolUse": [{"matcher": "Agent|Task",
                                             "hooks": [{"type": "command", "command": f"{run} hook",
                                                        "timeout": 5}]}]}
    if settings:
        args += ["--settings", json.dumps(settings)]
    if os.environ.get("AAR_GATEWAY_AGENTS", "1") != "0" and "--agents" not in claude_args:
        from .subagents import agent_definitions

        args += ["--agents", json.dumps(agent_definitions())]
    return args + claude_args, env


# --------------------------------------------------------------------------- #
# plain `claude`, every session (enable / disable)                            #
# --------------------------------------------------------------------------- #
def _claude_settings_path() -> Path:
    """Claude Code's user settings (``AAR_GATEWAY_CLAUDE_SETTINGS``: another
    settings file, such as a test project's)."""
    other = os.environ.get("AAR_GATEWAY_CLAUDE_SETTINGS")
    return Path(other).expanduser() if other else Path.home() / ".claude" / "settings.json"


def _suspended_file() -> Path:
    """Present while sessions bypass a gateway that isn't answering (:func:`_check`)."""
    return _home() / "bypassed"


#: the module a command we wrote runs; ``router.gateway.cli`` is what installs
#: from before this package was split out of agentic-llm-router wrote
_OWN_COMMANDS = ("aar_gateway.cli", "router.gateway.cli")


def _is_ours(command: str) -> bool:
    return any(name in command for name in _OWN_COMMANDS)


def _ours(entry: dict) -> bool:
    """A hook entry this package wrote: its command runs ``aar_gateway.cli``."""
    return any(_is_ours(h.get("command", "")) for h in entry.get("hooks", []))


def _read_claude_settings(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError as exc:
        raise SystemExit(f"{path} isn't valid JSON ({exc}); fix it, then run this again") from None


def _write_claude_settings(path: Path, settings: dict) -> None:
    from .config import atomic_write_text

    atomic_write_text(path, json.dumps(settings, indent=2) + "\n")


@contextlib.contextmanager
def _settings_lock(path: Path, *, wait: float = 10.0):
    """Holds a lock on Claude Code's settings file while it is read, changed
    and written: every session's hooks change it, and without the lock two of
    them at once can each write back what they read, losing the other's
    change. An OS file lock (:func:`aar_gateway.config.locked_file`): a
    process that dies releases it, and a slow live holder is never robbed of it."""
    from .config import locked_file

    try:
        with locked_file(path, timeout=wait):
            yield
    except TimeoutError:
        raise SystemExit(f"{path} is held by another aar-gateway command; try again") from None


def auto_enabled() -> bool:
    """Whether plain ``claude`` goes through this gateway (``aar-gateway enable``)."""
    try:
        settings = _read_claude_settings(_claude_settings_path())
    except SystemExit:
        return False
    return ((settings.get("env") or {}).get("ANTHROPIC_BASE_URL") == _base_url()
            or _suspended_file().exists())


def _set_base_url(on: bool) -> None:
    """Under :func:`_settings_lock` only."""
    path = _claude_settings_path()
    settings = _read_claude_settings(path)
    env = settings.setdefault("env", {})
    if on:
        env["ANTHROPIC_BASE_URL"] = _base_url()
    elif env.get("ANTHROPIC_BASE_URL") == _base_url():
        del env["ANTHROPIC_BASE_URL"]
    _write_claude_settings(path, settings)


def _check(session_start: bool) -> None:
    """Claude Code's SessionStart and UserPromptSubmit hook, once enabled:
    the gateway must answer before a message goes out, or Claude Code talks
    to Anthropic directly until it does.

    Claude Code rereads its settings while it runs, so taking the gateway's
    address out of them sends the session's next request straight to
    Anthropic: a gateway that won't start costs the routing, never the
    session. Starting it is tried first (it comes back within a second or
    two when the login service has it). Once it answers again, the address
    goes back in. Prints only a one-line notice for the user, when that
    changes: anything else a SessionStart hook prints joins the conversation.
    """
    url, bypassed = _base_url(), _suspended_file()
    notice = None
    if not _healthy(url):
        if bypassed.exists():
            # already talking to Anthropic: try a start without holding the message
            subprocess.Popen([sys.executable, "-m", "aar_gateway.cli", "start", "--quiet"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             env=_src_env(), **({"creationflags": subprocess.DETACHED_PROCESS} if WINDOWS
                                                else {"start_new_session": True}))
            return
        try:
            _ensure_running(url, quiet=True, timeout=60 if session_start else 15)
        except SystemExit:
            pass
    # The switch itself happens under the settings lock, the address and the
    # marker always changing together: two hooks switching opposite ways at
    # once can then never leave the address out with no marker to bring it
    # back. Health is checked before taking the lock (a refused connection
    # takes ~2 s on Windows; holding the lock for it would queue every other
    # session); a stale reading costs at most one switch the next check undoes.
    healthy = _healthy(url)
    switched_off = False
    try:
        with _settings_lock(_claude_settings_path()):
            if healthy and bypassed.exists():
                _set_base_url(True)
                bypassed.unlink(missing_ok=True)
                notice = "AAR gateway is answering again: routing is back on."
            elif not healthy and not bypassed.exists():
                bypassed.parent.mkdir(parents=True, exist_ok=True)
                _write_file(bypassed, url)
                _set_base_url(False)
                switched_off = True
    except SystemExit:
        return  # another hook held the lock too long; the next message checks again
    if switched_off:
        # Claude Code rereads its settings a moment after they change; the
        # message waits for that, or it would still go to the gateway's address
        time.sleep(float(os.environ.get("AAR_GATEWAY_SETTINGS_SETTLE", "3")))
        notice = ("AAR gateway isn't answering, so Claude Code is talking to Anthropic directly (no routing) "
                  "until it's back. `aar-gateway doctor` says why.")
    if notice:
        print(json.dumps({"systemMessage": notice}))


def _enable(force: bool = False) -> int:
    """Every ``claude`` session goes through the gateway, no command needed:
    Claude Code's user settings get the gateway's address, a SessionStart
    hook that starts the gateway when it isn't running, the subagent hook and
    the status line; the subagent kinds go to ``~/.claude/agents``; on macOS
    and Linux the gateway also starts at login. Only what's missing is added,
    and ``aar-gateway disable`` takes exactly that out again."""
    import tempfile

    # the environment's own folder: its `python` is often a link to one elsewhere
    here = Path(sys.prefix).resolve()
    if not force and any(here.is_relative_to(Path(t).resolve()) for t in (tempfile.gettempdir(), "/tmp")):
        raise SystemExit(f"this copy of the gateway is in a temporary folder ({sys.prefix}), which the\n"
                         f"system deletes: every `claude` session would then fail to connect. Install it with\n"
                         f"`uv tool install` and run `aar-gateway enable` from there (or pass --force).")
    path = _claude_settings_path()
    url = _base_url()
    with _settings_lock(path):
        _enable_settings(path, url, force)
    print(f"plain `claude` now goes through the gateway in every session ({path})")
    _agents(["--install"])
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            _service("install", at_next_login=_healthy(url))
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"couldn't register the login service ({exc}); each session starts the gateway instead")
    else:
        _ensure_running(url, quiet=True)
    print("undo with `aar-gateway disable`")
    return 0


def _enable_settings(path: Path, url: str, force: bool) -> None:
    """:func:`_enable`'s change to Claude Code's settings (under the lock)."""
    settings = _read_claude_settings(path)
    run = _self_command()
    env = settings.setdefault("env", {})
    current = env.get("ANTHROPIC_BASE_URL")
    if current and current != url and not force:
        raise SystemExit(f"Claude Code already sends requests to {current} (ANTHROPIC_BASE_URL in {path}).\n"
                         f"Run `aar-gateway enable --force` to point it at the gateway instead.")
    env["ANTHROPIC_BASE_URL"] = url
    _suspended_file().unlink(missing_ok=True)  # the next message checks the gateway afresh
    env.setdefault("ENABLE_TOOL_SEARCH", "true")  # see _claude_command
    hooks = settings.setdefault("hooks", {})
    wanted = {
        # before a session and before each message: the gateway answers, or
        # the session talks to Anthropic directly until it does (_check)
        "SessionStart": {"hooks": [{"type": "command", "command": f"{run} check --session", "timeout": 90}]},
        "UserPromptSubmit": {"hooks": [{"type": "command", "command": f"{run} check", "timeout": 30}]},
        # and before the request after each tool call: a turn makes many
        "PostToolUse": {"hooks": [{"type": "command", "command": f"{run} check", "timeout": 30}]},
        "PreToolUse": {"matcher": "Agent|Task",
                       "hooks": [{"type": "command", "command": f"{run} hook", "timeout": 5}]},
    }
    for event, entry in wanted.items():
        hooks[event] = [e for e in hooks.get(event, []) if not _ours(e)] + [entry]
    # the user's own status line stays; ours is rewritten (this install may have moved)
    if not settings.get("statusLine") or _is_ours(settings["statusLine"].get("command", "")):
        settings["statusLine"] = {"type": "command", "command": f"{run} statusline"}
    if path.exists() and not path.with_name("settings.json.aar-backup").exists():
        shutil.copy2(path, path.with_name("settings.json.aar-backup"))
    _write_claude_settings(path, settings)


def _disable() -> int:
    """Takes out what :func:`_enable` added; the rest of the settings stay."""
    path = _claude_settings_path()
    with _settings_lock(path):
        _disable_settings(path)
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _service("uninstall")
    print("plain `claude` talks to Anthropic directly again; `aar-gateway code` still uses the gateway")
    return 0


def _disable_settings(path: Path) -> None:
    settings = _read_claude_settings(path)
    env = settings.get("env") or {}
    if env.get("ANTHROPIC_BASE_URL") == _base_url() or _suspended_file().exists():
        if env.get("ANTHROPIC_BASE_URL") == _base_url():
            del env["ANTHROPIC_BASE_URL"]
        if env.get("ENABLE_TOOL_SEARCH") == "true":
            del env["ENABLE_TOOL_SEARCH"]
    _suspended_file().unlink(missing_ok=True)
    if not env:
        settings.pop("env", None)
    hooks = settings.get("hooks") or {}
    for event in list(hooks):
        hooks[event] = [e for e in hooks[event] if not _ours(e)]
        if not hooks[event]:
            del hooks[event]
    if not hooks:
        settings.pop("hooks", None)
    if _is_ours((settings.get("statusLine") or {}).get("command", "")):
        del settings["statusLine"]
    if path.exists():
        _write_claude_settings(path, settings)


# --------------------------------------------------------------------------- #
# the on-device history                                                       #
# --------------------------------------------------------------------------- #
HISTORY_NOTE = ("Keeping each message you type, with the model it got and how it went, on this computer only\n"
                "(personalization and `aar-gateway-grade`; never uploaded). `aar-gateway history off` stops it.")


def _size(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


def _history(argv: list[str]) -> int:
    """``aar-gateway history [on|off|status|show [N]|export FILE|delete [--yes]]``."""
    from . import history

    action = argv[0] if argv else "status"
    if action == "on":
        history.set_enabled(True)
        print(f"history on: your messages are kept in {history.history_dir()} (this computer only)")
        return 0
    if action == "off":
        history.set_enabled(False)
        print("history off: nothing new is kept. What is already there stays until `aar-gateway history delete`.")
        return 0
    if action == "status":
        info = history.stats()
        print(f"history      {'on' if info['enabled'] else 'off'}")
        print(f"kept         {info['rows']} message(s), {_size(info['bytes'])} in {info['files']} file(s)")
        print(f"where        {info['path']}  (plus gzip archives beside it)")
        print("never uploaded; `aar-gateway share` is a separate, anonymous record without any text")
        return 0
    if action == "show":
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 5
        for row in history.read(n):
            text = " ".join(str(row.get("prompt", "")).split())
            print(f"{row.get('ts', '')}  {row.get('routed_model') or '?'}  status {row.get('status')}  "
                  f"{text[:160]}{'...' if len(text) > 160 else ''}")
        return 0
    if action == "export":
        if len(argv) < 2:
            raise SystemExit("usage: aar-gateway history export <file>")
        dest = Path(argv[1]).expanduser()
        print(f"wrote {history.export(dest)} message(s) to {dest}")
        return 0
    if action == "delete":
        if "--yes" not in argv[1:]:
            if not sys.stdin.isatty():
                raise SystemExit("history delete removes every kept message; run it in a terminal, or add --yes")
            if input("Delete every message the history has kept? This cannot be undone. [y/N] ").strip().lower() not in (
                    "y", "yes"):
                print("kept")
                return 0
        print(f"deleted {history.delete()} message(s)")
        return 0
    raise SystemExit(f"unknown history action {action!r}: on, off, status, show [N], export <file> or delete")


# --------------------------------------------------------------------------- #
# opt-in usage sharing                                                        #
# --------------------------------------------------------------------------- #
SHARE_QUESTION = ("Share anonymous usage data to help improve routing? Never your prompts, file paths or ids:\n"
                  "which models answered, cost, tokens, and a compressed embedding of each message.\n"
                  "It is sent to the project's collection server, to retrain the router, once one is set up;\n"
                  "until then it stays on this computer (`aar-gateway share show` shows it). [y/N] ")


def _share(argv: list[str]) -> int:
    """``aar-gateway share [on|off|status|show|delete]``."""
    from . import share

    action = argv[0] if argv else "status"
    if action == "on":
        share.set_enabled(True)
        dest = (f"sent to {share.endpoint()} once a day" if share.endpoint()
                else "kept on this computer until a collection endpoint is set (`aar-gateway share endpoint <https url>`)")
        print(f"sharing on: each message you type adds an anonymous record (no prompt text), {dest}.\n"
              "`aar-gateway share show` prints them; `aar-gateway share off` stops it.")
        return 0
    if action == "off":
        share.set_enabled(False)
        print(f"sharing off: no new records and nothing more is sent. Records not yet sent stay in\n"
              f"{share.queue_path()} until `aar-gateway share delete`.")
        return 0
    if action == "delete":
        with share.locked_file(share.queue_path()):
            share.queue_path().unlink(missing_ok=True)
        print("deleted every kept record")
        return 0
    if action == "show":
        rows = share.queued()
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 3
        for row in rows[-n:]:
            emb = row.get("embedding")
            print(json.dumps({**row, "embedding": f"{len(emb)} small integers" if emb else None}, indent=2))
        print(f"{len(rows)} record(s) kept in {share.queue_path()}")
        return 0
    if action == "endpoint":
        if len(argv) < 2:
            raise SystemExit("usage: aar-gateway share endpoint <https address of the collection server>")
        try:
            share.set_endpoint(argv[1])
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(f"collection endpoint saved; records go to {share.endpoint()} once a day")
        return 0
    if action == "token":
        raise SystemExit("uploads no longer use a Hugging Face token: records go to the project's collection "
                         "server. Set its address with `aar-gateway share endpoint <https url>`.")
    if action == "upload":
        result = share.upload()
        if "sent" in result:
            print(f"sent {result['sent']} record(s) to {share.endpoint()}")
            return 0
        print(result.get("skipped") or f"upload failed ({result['error']}); the records are kept for next time")
        return 0 if "skipped" in result else 1
    if action == "status":
        st = share.state()
        print(f"sharing      {'on' if share.enabled() else 'off'}")
        print(f"kept         {len(share.queued())} record(s) waiting in {share.queue_path()}")
        print(f"sent to      {share.endpoint()}" if share.endpoint()
              else "sent to      nowhere yet (no collection endpoint: aar-gateway share endpoint <https url>)")
        if st.get("sent_total"):
            print(f"sent so far  {st['sent_total']} record(s), last {st.get('last_upload')}")
        if st.get("last_error"):
            print(f"last upload  failed ({st['last_error']}); retried later")
        return 0
    raise SystemExit(f"unknown share action {action!r}: on, off, status, show [N], delete, endpoint or upload")


def _agents(argv: list[str]) -> None:
    """``aar-gateway agents``: print the kinds as ``--agents`` JSON;
    ``--install [DIR]``: write them as agent files (default
    ``~/.claude/agents``) so plain ``claude`` has them too, without the
    gateway. Existing files are left alone."""
    from .subagents import ROLES, agent_definitions, agent_markdown

    if argv[:1] != ["--install"]:
        print(json.dumps(agent_definitions(), indent=2))
        return
    target = Path(argv[1]).expanduser() if len(argv) > 1 else Path.home() / ".claude" / "agents"
    target.mkdir(parents=True, exist_ok=True)
    for name in ROLES:
        path = target / f"{name}.md"
        if path.exists():
            print(f"kept   {path} (already exists)")
            continue
        _write_file(path, agent_markdown(name))
        print(f"wrote  {path}")


def _hook() -> None:
    """Claude Code's ``PreToolUse`` hook for the Agent tool: asks the gateway
    which model the subagent should run on and returns the call with
    ``model`` set. Anything else -- another tool, the gateway down, no
    decision -- prints nothing, which leaves the call exactly as it was."""
    import urllib.request

    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return
    if event.get("tool_name") not in ("Agent", "Task"):
        return
    call = dict(event.get("tool_input") or {})
    payload = {"session_id": event.get("session_id"), "subagent_type": call.get("subagent_type"),
               "prompt": call.get("prompt"), "model": call.get("model")}
    try:
        req = urllib.request.Request(f"{_base_url()}/api/subagent-spawn", data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            alias = json.loads(resp.read()).get("alias")
    except (OSError, ValueError):
        return
    if alias:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                 "updatedInput": {**call, "model": alias}}}))


def _statusline() -> None:
    """Reads Claude Code's status-line JSON on stdin, prints one line."""
    import urllib.parse

    try:
        session = json.loads(sys.stdin.read() or "{}").get("session_id", "")
    except ValueError:
        session = ""
    info = _get(f"{_base_url()}/api/statusline?{urllib.parse.urlencode({'session_id': session})}")
    print(info.get("line", "aar gateway") if info else "aar gateway: not running")


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    url = _base_url()
    # Claude Code's own calls, and `code` (whose arguments belong to `claude`):
    # handled before argparse, with no heavy imports
    if argv[:1] == ["code"]:
        _ensure_running(url)
        args, env = _claude_command(url, argv[1:])
        claude = shutil.which("claude", path=env.get("PATH"))
        if claude is None:
            raise SystemExit("`claude` is not on PATH -- install Claude Code first (see `aar-gateway doctor`)")
        if WINDOWS:  # no exec() there: run it and pass its exit code on
            return subprocess.call([claude, *args[1:]], env=env)
        os.execve(claude, [claude, *args[1:]], env)
    if argv[:1] == ["hook"]:
        _hook()
        return 0
    if argv[:1] == ["check"]:
        try:
            _check(session_start="--session" in argv)
        except Exception as exc:  # noqa: BLE001 -- never stand between the user and their message
            print(f"aar-gateway check: {exc}", file=sys.stderr)
        return 0
    if argv[:1] == ["statusline"]:
        _statusline()
        return 0
    if argv[:1] == ["agents"]:
        _agents(argv[1:])
        return 0
    if argv[:1] == ["local"]:
        return _local(argv[1:])
    if argv[:1] == ["share"]:
        return _share(argv[1:])
    if argv[:1] == ["history"]:
        return _history(argv[1:])

    parser = argparse.ArgumentParser(prog="aar-gateway", description="Route Claude Code through the AAR gateway.")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("code", help="run Claude Code through the gateway, starting it if needed "
                                "(further arguments go to `claude`)")
    start = sub.add_parser("start", help="start the gateway in the background")
    start.add_argument("--quiet", action="store_true", help="print nothing (Claude Code's SessionStart hook)")
    sub.add_parser("stop", help="stop the gateway")
    sub.add_parser("restart", help="stop, then start the gateway")
    status = sub.add_parser("status", help="is it running, is the router loaded, where are its files")
    status.add_argument("--json", action="store_true")
    logs = sub.add_parser("logs", help="the gateway's log")
    logs.add_argument("-n", type=int, default=50, help="lines to show (default 50)")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing new lines")
    setup = sub.add_parser("setup", help="install the router and write a config (first-time setup)")
    setup.add_argument("--router", help="a router file or URL (default: the one shipped with the package)")
    setup.add_argument("--force", action="store_true", help="rewrite an existing config")
    every = setup.add_mutually_exclusive_group()
    every.add_argument("--yes", action="store_true", help="use the gateway in every `claude` session without asking")
    every.add_argument("--no-enable", action="store_true",
                       help="leave plain `claude` alone (use `aar-gateway code` instead)")
    sharing = setup.add_mutually_exclusive_group()
    sharing.add_argument("--share", action="store_true", help="opt in to anonymous usage sharing without asking")
    sharing.add_argument("--no-share", action="store_true", help="don't share usage data, and don't ask")
    enable = sub.add_parser("enable", help="every `claude` session goes through the gateway, no command needed")
    enable.add_argument("--force", action="store_true", help="replace another ANTHROPIC_BASE_URL in your settings")
    sub.add_parser("disable", help="undo `enable`: plain `claude` talks to Anthropic directly again")
    sub.add_parser("doctor", help="check the setup, with a fix for each problem")
    sub.add_parser("local", help="optional local models via Ollama: local [add|remove] <model>")
    sub.add_parser("share", help="opt-in anonymous usage sharing: share [on|off|status|show|delete|token|upload]")
    sub.add_parser("config", help="which config file is in use")
    sub.add_parser("dashboard", help="open the dashboard in a browser")
    service = sub.add_parser("service", help="start the gateway at login (launchd on macOS, systemd on Linux)")
    service.add_argument("action", choices=("install", "uninstall", "status"))
    sub.add_parser("serve", help="run the gateway in the foreground")
    sub.add_parser("agents", help="the subagent kinds as --agents JSON; --install [DIR] writes agent files")
    sub.add_parser("statusline", help="Claude Code's status-line command (reads JSON on stdin)")
    sub.add_parser("hook", help="Claude Code's PreToolUse hook for subagents (reads JSON on stdin)")
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "start":
        if args.quiet:
            try:
                _ensure_running(url, quiet=True)
            except SystemExit:
                pass  # Claude Code's next request reports the gateway being down
            return 0
        started = _ensure_running(url)
        print(("started" if started else "already running") + f": {url}  (dashboard {url}/dashboard)")
        return 0
    if args.command == "stop":
        _stop()
        return 0
    if args.command == "restart":
        _restart()
        print(f"restarted: {url}")
        return 0
    if args.command == "status":
        return _status(as_json=args.json)
    if args.command == "logs":
        return _logs(args.n, args.follow)
    if args.command == "doctor":
        return _doctor()
    if args.command == "setup":
        return _setup(args.router, args.force, enable=True if args.yes else False if args.no_enable else None,
                      share=True if args.share else False if args.no_share else None)
    if args.command == "enable":
        return _enable(force=args.force)
    if args.command == "disable":
        return _disable()
    if args.command == "config":
        cfg = _config_path()
        print(cfg if cfg else f"none -- using the defaults (router off). Put a config at "
                              f"{_home() / 'config.yaml'} or set AAR_GATEWAY_CONFIG.")
        return 0 if cfg else 1
    if args.command == "dashboard":
        import webbrowser

        _ensure_running(url)
        print(url + "/dashboard")
        webbrowser.open(url + "/dashboard")
        return 0
    if args.command == "service":
        return _service(args.action)

    import uvicorn  # serve

    from .server import app

    if not WINDOWS:
        import signal

        # uvicorn shuts down on SIGTERM, then re-raises it. Dying of a signal
        # counts as a crash to launchd, which would restart a gateway that
        # `aar-gateway stop` just stopped: a stop is a clean exit
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    import socket

    host, port = _listen()
    config = uvicorn.Config(app, host=host, port=port,
                            timeout_graceful_shutdown=5)  # open dashboard streams never close by themselves
    # the port first, before anything loads: a second gateway started at the
    # same moment exits here, without touching the running one's pid file,
    # and requests arriving while the router loads wait instead of being refused.
    # Windows: SO_REUSEADDR there lets a second socket take a port in use, so
    # the exclusive option is used instead (uvicorn's own bind sets REUSEADDR)
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    if WINDOWS:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"can't listen on {host}:{port} ({exc}): another gateway may be running -- "
                         f"see `aar-gateway status`") from None
    running = uvicorn.Server(config)
    app.state.uvicorn = running  # POST /api/shutdown asks it to exit
    running.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    sys.exit(main())
