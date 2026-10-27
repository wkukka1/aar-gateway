# aar-gateway

A local gateway between Claude Code and the model providers. Claude Code keeps its login and tools;
at every new message a small trained router picks the model that should handle the turn, and a
dashboard shows what that cost against two baselines. Two routers do the picking: an MLP for better
quality, and a manifold router for the most savings ([which one answers](#which-router-answers)).

```
Claude Code ──► gateway (127.0.0.1:8787) ──► router picks a model for this turn
                    │                          ├─ Claude model → Anthropic, on your own login
                    └─► request log/dashboard  └─ other model  → its API / Ollama, tools translated
```

## Install

```bash
uv tool install git+https://github.com/wkukka1/aar-gateway
aar-gateway setup        # installs the bundled router file and writes ~/.aar-gateway/config.yaml
aar-gateway router list  # the bundled routers; `router use <name>` switches the quality (MLP) router
aar-gateway start        # turns the gateway on: every Claude Code session goes through it
aar-gateway stop         # turns it off again: Claude Code talks to Anthropic directly
aar-gateway dashboard    # http://127.0.0.1:8787/dashboard
aar-gateway doctor       # if anything looks wrong
```

No clone needed: `uvx --from git+https://github.com/wkukka1/aar-gateway aar-gateway setup` runs it once.
Python 3.11+. The router is served with numpy and ONNX (`fastembed`); there is no PyTorch.

All options are in [`gateway.example.yaml`](src/aar_gateway/gateway.example.yaml); the full guide is
[docs/gateway.md](docs/gateway.md).

## Which router answers

Two bundled router files. The cost preference picks which one prices each turn:

| Cost preference | Router | Config key | Tuned for |
|---|---|---|---|
| Quality margin: Best quality, Balanced (the default) | `llmrouterbench-mlp` | `router.artifact` | Better quality |
| Cost weight λ (the dashboard's Save more, the λ presets, `cost_preset` / `x-aar-cost-weight`) | `manifold-nirt-claude-judged` | `router.savings_artifact` | Most savings |

What they measured, each against its own baseline:

- **MLP, Best quality**: quality 0.626 on LLMRouterBench's held-out split, against 0.596 for always
  GPT-5, at 6% less cost than always GPT-5.
- **Manifold, λ presets**: out-of-fold on 99 judged turns, against always Opus (quality 0.711): λ 0.02
  keeps 0.706 at 24% saved; λ 0.2 gives 0.665 at 72% saved; λ 0.3 gives 0.657 at 76% saved.

The manifold scores the Claude tiers directly. It needs a BERT encoder that fastembed doesn't ship,
which `aar-gateway setup` doesn't install. Export it once from the
[agentic-llm-router](https://github.com/wkukka1/agentic-llm-router) repo with
`python scripts/routing/export_bert_encoder.py` (writes `~/.aar-gateway/encoders/bert-base-uncased-mean`,
436 MB). Until then λ turns fall back to the MLP, and `/health` shows why under `savings.error`.

Set the savings file with `router.savings_artifact` in `~/.aar-gateway/config.yaml`; `null` turns it off.
Details, including the λ rule and the dashboard's presets, are in [docs/gateway.md](docs/gateway.md#two-routers-quality-and-savings).

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
