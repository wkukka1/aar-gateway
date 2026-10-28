# Aptic for Claude Code

Use less of your Claude plan with the same login, tools and skills. For each
message you type, Aptic picks the cheapest Claude model that should answer
it about as well as the model you chose, and switches the turn to it from inside
Claude Code. Subagents get a model by the kind of work they do.

## Install

In Claude Code:

```
/plugin install aptic --marketplace wkukka1/aar-gateway
```

## What you get

* **Routing per turn**, by a small trained router that runs on your computer.
  Turn it off in `/config` (the plugin's "Route in Claude Code" setting).
* **A pane** (`/aptic-pane`): this session's saving against always using the
  model you asked for, the mix of models its requests ran on, its requests,
  cache hit and turns, and all sessions' plan usage; then a table of turns
  (newest or oldest first) with each turn's time, model, decision, requests
  and cache hit, and any subagents. A chevron collapses it to the two savings
  and a requests-per-turn chart; the pane remembers your choice.
* **A line under each answer** with the model that answered and its cache use,
  and the totals on the status line. `/aptic` prints them.

## Requirements

* Claude Code 2.1.287 or later.
* [uv](https://docs.astral.sh/uv/). The plugin uses it to run the router
  (`uvx`), the first time downloading it in a few seconds. Without uv the
  plugin says so and Claude Code works as normal.

## What it does on your machine

Nothing else is installed and no settings are changed. If the router isn't
answering, every request goes to the model you asked for.

### Programs it runs

Only these, and only when needed:

* `uvx --from "aar-gateway @ git+https://github.com/wkukka1/aar-gateway@main" aar-gateway setup --no-enable --no-share --no-mod`,
  once, when `~/.aar-gateway/config.yaml` doesn't exist yet: writes that config
  file and the router file, and changes nothing else.
* `uvx --from "…same…" aar-gateway start --quiet`, when the router isn't
  answering: starts it as a background process on `127.0.0.1:8787`.
  If `uvx` isn't on `PATH`, it tries `~/.local/bin/uvx`, where uv installs itself.
* Your system's URL opener, with the fixed address
  `http://127.0.0.1:8787/dashboard`, only when you press the pane's dashboard
  button: `rundll32 url.dll,FileProtocolHandler` on Windows, `open` on macOS,
  `xdg-open` on Linux.

### What it fetches and sends, and where

* The plugin itself makes requests to one host only: the router on your own
  computer, `http://127.0.0.1:8787`. It never contacts any other host. It
  takes no commands from the router: the router's answers are only model
  names and numbers to show.
* uv downloads the router package from GitHub (`github.com/wkukka1/aar-gateway`)
  and its dependencies from PyPI, the first time `uvx` runs it.
* What goes to the local router:
  * `/api/route`: the message you typed, the model you chose, the session id,
    and whether the session was resumed. The router picks this turn's model.
  * `/api/subagent-spawn`: a subagent's type, its prompt and its model. The
    router picks the subagent's model.
  * `/api/mod-record`: each request's model and token counts (input, output,
    cache read, cache write), for the savings numbers. No prompt or answer text.
  * `/health`, `/stats`, `/api/session-requests`: reads only, for the pane and
    the status line.
* For the plugin's requests the router keeps no prompt text; it logs models and
  token counts in `~/.aar-gateway`. (If you also turn on its proxy with
  `aar-gateway enable`, the proxy keeps an on-device history of the messages
  you type in `~/.aar-gateway/history`, never uploaded; `aar-gateway history off`
  stops it and `aar-gateway history delete` removes it.)
* The plugin starts the router with usage sharing off (`--no-share`), so
  nothing is uploaded.

### What it reads

* The conversation: the text of each message you type (to route it) and each
  request's token usage (for the savings).
* Environment variables `HOME` / `USERPROFILE` (to find `~/.aar-gateway` and
  `uvx`), `OS` (to pick the URL opener) and `ANTHROPIC_BASE_URL` (to tell
  whether this session already goes through the router's proxy). None of
  them is sent anywhere.

### Hooks and what they do with what they see

* `turn.start`: asks the local router for this turn's model and switches the
  turn to it. Nothing else about the turn changes.
* `agent.spawn`: replaces only the subagent's `model` with the one the router
  picks. Its tools, prompt and permission mode stay as Claude Code set them,
  and your permission rules and prompts still decide every tool use. The
  plugin answers no permission checks.
* `turn.step`, `turn.complete`: read token usage and add one line under the
  answer naming the model that answered.
* Turn routing off in `/config` ("Route in Claude Code") and `turn.start` and
  `agent.spawn` pass everything on unchanged.

Source and tests: https://github.com/wkukka1/aar-gateway

The measurements behind the numbers:
https://github.com/wkukka1/agentic-llm-router

License: Apache-2.0.
