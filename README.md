# aar-gateway

A local gateway between Claude Code and the model providers. Claude Code keeps its login and tools;
at every new message a small trained router picks the cheapest model that should handle the turn, and a
dashboard shows what that cost against two baselines.

```
Claude Code ──► gateway (127.0.0.1:8787) ──► router picks a model for this turn
                    │                          ├─ Claude model → Anthropic, on your own login
                    └─► request log/dashboard  └─ other model  → its API / Ollama, tools translated
```

## Install

```bash
uv tool install git+https://github.com/wkukka1/aar-gateway
aar-gateway setup        # installs the bundled router file and writes ~/.aar-gateway/config.yaml
aar-gateway code         # starts the gateway if needed, then runs Claude Code through it
aar-gateway dashboard    # http://127.0.0.1:8787/dashboard
aar-gateway doctor       # if anything looks wrong
```

No clone needed: `uvx --from git+https://github.com/wkukka1/aar-gateway aar-gateway setup` runs it once.
Python 3.11+. The router is served with numpy and ONNX (`fastembed`); there is no PyTorch.

All options are in [`gateway.example.yaml`](src/aar_gateway/gateway.example.yaml); the full guide is
[docs/gateway.md](docs/gateway.md).

## Also installed

- `aar-gateway-grade`: replay your own Claude Code turns on each tier and grade them, so routing can use your
  work instead of benchmark stand-ins (uses the `claude` CLI and your login).
- `aar-gateway-subagents`: price what Claude Code's subagents cost against the routed alternative.

## Development

```bash
uv sync --extra dev
uv run pytest
```

The router file's training pipeline lives in
[agentic-llm-router](https://github.com/wkukka1/agentic-llm-router); this repo only serves router files.
