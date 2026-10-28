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

* Starts the router (a local process on `127.0.0.1:8787`) when it isn't running,
  with `uvx`, from this repository; first run writes `~/.aar-gateway/config.yaml`
  and the router file. Nothing else is installed and no settings are changed.
* Sends your typed message to that local process to pick a model, and reports
  each request's token counts to it so it can show savings. Nothing leaves your
  computer; no prompt text is stored.
* If the router isn't answering, every request goes to the model you asked for.

Source and tests: https://github.com/wkukka1/aar-gateway

The measurements behind the numbers:
https://github.com/wkukka1/agentic-llm-router

License: Apache-2.0.
