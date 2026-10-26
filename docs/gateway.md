# Gateway: route Claude Code through the router

> Split out of [agentic-llm-router](https://github.com/wkukka1/agentic-llm-router). Sections that
> build router files, run the data pipeline or evaluate on benchmarks use that repo's scripts
> (`scripts/`, `training/`, `evaluation/`); everything for *running* the gateway is here.

`aar_gateway` is a local process that sits between Claude Code and the
model providers, in the same place [claude-code-router](https://github.com/musistudio/claude-code-router)
sits. Claude Code keeps working exactly as before — same login, same tools —
and at every new message the trained router decides which model answers that
turn. A dashboard shows what that cost against two baselines.

```
Claude Code ──► gateway (127.0.0.1:8787) ──► router picks a model for this turn
  (your login)        │                          │
                      │                          ├─ Claude model → Anthropic, on your own login
                      │                          └─ other model  → its API / Ollama, tools translated
                      └─► request log ──► /dashboard
```

## Quick start

```bash
uv tool install git+https://github.com/wkukka1/aar-gateway
aar-gateway setup          # installs the bundled router file + a config

aar-gateway start         # turns the gateway on: every Claude Code session goes through it
aar-gateway stop          # turns it off again: Claude Code talks to Anthropic directly
aar-gateway dashboard     # opens http://127.0.0.1:8787/dashboard
aar-gateway doctor        # if anything looks wrong
```

Or run it without installing: `uvx --from git+https://github.com/wkukka1/aar-gateway aar-gateway setup`.
From a checkout: `uv sync --extra dev`. The router file is described
under [The router file](#the-router-file).

The config is `AAR_GATEWAY_CONFIG` if set, else `~/.aar-gateway/config.yaml`
if it exists, else the defaults (router off).

## Two routers: quality and savings

The gateway serves two router files, and the cost preference decides which one prices a turn:

| Turn priced by | Router | Config key | Calibrated for |
|---|---|---|---|
| Quality margin: Best quality, Balanced or Save more | `llmrouterbench-mlp` (default) | `router.artifact` | Better quality: the trained network, read through each tier's `proxy` |
| Cost weight λ: the λ presets, or `x-aar-cost-weight` | `manifold-nirt-claude-judged` (default) | `router.savings_artifact` | Most savings: scores the Claude tiers directly |

* A turn with λ set is priced by the savings router. A turn on the margin is priced by the
  margin router. One turn never mixes the two: their predicted qualities are on different scales.
* The dashboard's **Save more** preset is a margin, so it stays on the margin router. For the
  manifold's savings, use a λ preset: `cost_preset` `balanced` or `save_more` (Advanced settings → Cost weight).
* When the user's own grades cover a prompt, quality comes from those grades under either rule
  (`quality_source: graded`), as before.
* If the savings file can't load (for example, its BERT encoder folder is missing), λ turns use the
  margin router, margin turns are unaffected, and `/health` reports why under `savings.error`.
  The bundled manifold needs the encoder described in [The judged manifold router](#the-judged-manifold-router-manifold-nirt-claude-judged).
* `aar-gateway router use` swaps the margin router only. The savings router is set in the config
  with `router.savings_artifact` (`null` turns it off).

## The router file

The gateway's margin router is one small file. **The shipped one is a
trained network** ([`MLPArtifactPayload`](../src/aar_gateway/_core/artifacts.py)):
a prompt's embedding in, a predicted score per model out (`Linear → ReLU →
Linear`, 256 hidden units). The file holds only the weights, **0.38 MB, no
training prompts or scores**, and is served with numpy. On LLMRouterBench's
held-out split it matches or beats the kNN router at every margin (Balanced:
0.612 at $19.79 per 1k against kNN's 0.596 at $17.25; at equal cost the
network is ~0.005–0.009 higher); loaded from the file it scores exactly as
it did in training.

The alternative (`--kind knn`) is a kNN index: the training prompts'
embeddings and their per-model scores
([`KNNArtifactPayload`](../src/aar_gateway/_core/artifacts.py)). A prompt's
predicted quality per model is the mean score of its 25 nearest training
prompts. For LLMRouterBench that is 8,602 prompts × 13 models, **6.1 MB**.
It needs no training data and no PyTorch: the text encoder
(`all-MiniLM-L6-v2`) runs on `fastembed` (ONNX), which gives the same vectors
as sentence-transformers (cosine 1.0000, the longest prompts included). On
500 held-out test prompts the file's predictions match the full `KNNRouter`
exactly (largest difference 0.0; same top model on all 500).

```bash
# build it (needs the pool's processed data and retrieval embeddings)
python scripts/routing/export_router.py --config configs/llmrouterbench.yaml \
    --dataset llmrouterbench --out router-llmrouterbench.npz      # trained network (default)
# or --kind knn --k 25 for the kNN index

# use it
aar-gateway setup --router router-llmrouterbench.npz
```

`setup` copies (or downloads) the file to `~/.aar-gateway/router.npz`, checks
it loads, and writes `interceptor.router.artifact` into a new config. In any
config, `router.artifact` takes precedence over `data_config`.

**A trained NIRT, as a router file** (`--kind nirt`). The package also ships
`routers/llmrouterbench-nirt.npz`, the NIRT run `lrb-nirt-projected` (13 models, 0.1 MB).
With the pool fixed, a NIRT whose query head is `Linear → ReLU → Linear` scores
`a_m · θ(e) − b_m`, and that is exactly the same MLP (`w2 = A·W2`, `b2 = A·b2 − B`,
with `A, B` the pool's per-model parameters). So it is saved in the `mlp` format and served
the same way, with no PyTorch. The exporter refuses anything that doesn't fold this way:
an interaction term, per-query difficulty, vector difficulty, batch-norm, a deeper or
linear query head, `free` model parameters, IRT-Router orientation, query features, or the
ZOIB/`BaselineNIRT` runs. It then checks the file against the run's own scores, both in memory and
after reloading through this package's loader (`lrb-nirt-projected`: max |ΔP| 1.8e-7 on
500 test prompts).

```bash
python scripts/routing/export_router.py --config configs/llmrouterbench.yaml     --kind nirt --run lrb-nirt-projected --dataset llmrouterbench     --out aar-gateway/src/aar_gateway/routers/llmrouterbench-nirt.npz
```

### Switching routers

`aar-gateway router` swaps the file the gateway serves. It never edits the config:
`router.artifact` keeps pointing at `~/.aar-gateway/router.npz`, and `use` replaces that
file (load-checked, atomically) and restarts a running gateway.

```bash
aar-gateway router list                  # bundled + added files; * marks the one served
aar-gateway router use llmrouterbench-nirt
aar-gateway router use llmrouterbench-mlp          # back to the default
aar-gateway router add my-router.npz [--name NAME] # into ~/.aar-gateway/routers/
```

`router list` also shows `manifold-nirt-claude-judged`, the savings router. Don't `use` it here: that
would serve it under the margin rule, which it is not calibrated for. It is set by `router.savings_artifact`.

Added files live in `~/.aar-gateway/routers/`, and a name there wins over a bundled one.
(`~/.aar-gateway/models/` is the text encoder's cache.) `status`, `doctor` and `/health`
name the router by its file's `artifact_id`, e.g. `nirt-lrb-nirt-projected`.

## Managing the gateway

| Command | What it does |
|---|---|
| `aar-gateway start` | starts the gateway and sends every Claude Code session through it. `start --quiet` (the session hook) only starts it |
| `aar-gateway stop` | sends Claude Code straight to Anthropic again, then stops the gateway |
| `aar-gateway restart` | stops, then starts the gateway; Claude Code's settings are left as they are |
| `aar-gateway status` | running or not, router loaded or why not, config, requests and saving, where the logs are |
| `aar-gateway logs [-n 50] [-f]` | the gateway's own log; `-f` follows it |
| `aar-gateway doctor` | checks Claude Code, the config, the router, the port and the log folder, with a fix for each problem |
| `aar-gateway config` | which config file is in use |
| `aar-gateway setup [--router FILE\|URL] [--force]` | install the router file and write a config (existing configs are kept without `--force`) |
| `aar-gateway router [list\|use NAME\|add FILE\|URL]` | list the router files, switch the one served, or add your own ([Switching routers](#switching-routers)) |
| `aar-gateway service install` / `uninstall` / `status` | start at login: a LaunchAgent on macOS, a systemd user unit on Linux; restarted if it crashes, not after `stop` |
| `aar-gateway serve` | the gateway in the foreground |
| `aar-gateway view [--host ADDRESS] [--port N]` | a read-only copy of the dashboard on another address (your Tailscale IPv4 by default, port 8788), for checking from a phone |

`status --json` and `doctor --json` print the same as JSON, as does `share status --json`. The command line is
[`aar_gateway/cli.py`](../src/aar_gateway/cli.py); `hook` and
`statusline`, which Claude Code calls itself, import only the standard
library (the subagent hook runs in about 0.07 s).

**Operating systems.** macOS is what this is run and tested on. Linux uses
the same process handling (and systemd for `service`) but hasn't been run
yet. Windows has its own code paths -- Claude Code is run as a child process
instead of replacing this one, the gateway is detached with Windows process
flags, and liveness is checked over HTTP (`os.kill(pid, 0)` would terminate
the process there) -- but is untested, and `service` isn't available there:
use `aar-gateway start`.

With `aar-gateway start` on, `claude` itself (`claude -p "..."`, `claude -c`) goes through
the gateway, so every flag works as usual. `start` also sets Claude Code's
status line to show **which model actually answered** and this session's
cost against no router, e.g.
`aar router · claude-sonnet-5-5 (routed) · $0.21 · saved $0.26 (55%)`.
That line is the reliable answer to "which model is this". Asking the model
isn't: Claude Code's system prompt tells it that it is the model *you*
picked, so a rerouted Sonnet still says "Opus". If you already have a
status line it is left alone (or set `AAR_GATEWAY_STATUSLINE=0`).

**Don't drop `ENABLE_TOOL_SEARCH=true`.** For any base URL that isn't
Anthropic's, Claude Code stops loading MCP tools on demand and sends every
tool's schema with every request. With a lot of MCP servers that means
~245k tokens per request instead of ~40k: several times the cost, and
compaction on every turn.

**If `aar-gateway` says `No module named 'router'`:** your virtualenv is in
an iCloud-synced folder (`~/Documents`, `~/Desktop`). macOS keeps marking its
`.pth` files hidden, and Python 3.13+ skips hidden `.pth` files. Either put
the venv outside iCloud (`python3 -m venv ~/.venvs/aar`) or
`export PYTHONPATH=$PWD/src`. The router itself is the `.npz` file `router.artifact` names,
not a training run, so the gateway needs nothing from the training repo.

## Check it works

```bash
curl -s localhost:8787/health          # router on? graded prompts + calibration offsets
claude -p "Read aar-gateway/src/aar_gateway/graded.py and explain min_prompts in two sentences."
tail -3 ~/.aar-gateway/requests.jsonl  # requested vs routed model, why, cost vs no-router
aar-gateway history show 1             # the turn was kept in the on-device history
aar-gateway-grade --n 2                # grade pending turns on the real tiers
curl -s localhost:8787/health          # graded_prompts went up; offsets recomputed, no restart
```

The status line in Claude Code and the dashboard's "Routed" column show which
model answered. Asking the model can't tell you: Claude Code's system prompt
names the model you picked.

## How one request flows

[`aar_gateway/interceptor.py`](../src/aar_gateway/interceptor.py):

1. **Is this a new turn?** A request whose last user message carries new
   human text starts a turn. Claude Code's `<system-reminder>` text, and the
   hook output it appends as a trailing `system` message, are not the human's
   words and are ignored. A request carrying only tool results continues the
   current turn and **stays on the turn's model**, so a tool loop never changes
   model halfway through.
2. **Which model?** On a new turn of the agent conversation, the picker
   decides (next section). Claude Code's side requests (title generation and
   similar, which carry no tools) are left alone: they cost almost nothing
   and some models reject their settings.
3. **Send it.** A Claude model: forwarded to Anthropic close to untouched —
   the client's own `Authorization`, every tool, thinking and beta settings,
   real streaming. Only `model` may change. Any other model: translated (see
   *Other models*).
4. **Fail open.** If the routed model errors, the request is retried on the
   model Claude Code asked for. A 400 is remembered per client version, so a
   model that client can't use is not tried again. Anthropic rejects models
   newer than the Claude Code version supports, for example.
5. **Log it** (see *Dashboard*).

The gateway never stores a credential for the Claude path. It forwards the
one Claude Code sent, and never writes headers, credentials or prompt text
to its log.

## How the router decides

[`aar_gateway/routing.py`](../src/aar_gateway/routing.py) composes the
existing pieces: a trained `RouterModel` scores the prompt, and
`MarginRoutingPolicy` takes **the cheapest candidate within the user's
quality margin of the best**.

**Candidates** are the models in `interceptor.models`, minus any that:

* are pricier than the model the user picked (the router only ever saves);
* can't fit the prompt (`context_window`). Claude Code's prompt is ~200k
  tokens, so Haiku 4.5's 200k window usually rules it out;
* reject the request's settings (`rejects`);
* were rejected for this client before;
* are non-Claude models that can't take the request without losing something.

**Quality** comes from a router fitted on **LLMRouterBench**. That
benchmark scores 13 models from 2025 (Claude Sonnet 4, GPT-5, Gemini 2.5
Pro/Flash, DeepSeek V3/R1, Qwen3, GLM-4.6, Kimi K2, Intern-S1) on 12,336
prompts across 10 benchmarks, including SWE-bench and τ²-bench (tool use).
See *Training data and router choice* below.

Each configured model names a `proxy`: the pool model whose predicted
quality stands in for it. No public dataset scores today's Claude tiers on
the same prompts, so:

* **Opus** reads **GPT-5**, the pool's strongest;
* **Sonnet** reads **Claude Sonnet 4**, the same family one generation back;
* **Haiku** reads **Gemini 2.5 Flash**, the pool's fast tier.

The router therefore answers "how much does this prompt need the top tier
versus the mid or fast tier", measured on 2025 models. That's much closer
than the 2023 pool the first version used, but it is still not a measurement
of Claude 5.x itself.

**Cost** is estimated without predicting tokens per model. The gateway keeps
a running average of the tokens each request in a conversation actually used:

* new uncached input;
* new cache *writes*: what the conversation added since the last request;
* output, thinking included.

It prices that same measured profile on each candidate model. A new
conversation starts from the average over all agent traffic. Per model, two
things then differ:

* **Price.** Claude Code's prompt is mostly cache reads, so the decisive
  difference is usually the price of the per-request cache writes (2x input:
  $8/MTok on Opus 5.5, $4 on Sonnet 5.5). The read prices can be equal: $0.20
  on both. A reads-only estimate made a warm Opus look as cheap as Sonnet, so
  nothing switched at any margin.
* **Cache state.** Caches are per model. A model not holding the conversation
  rewrites it once at the write price.

A model counts as warm if it served this conversation, or cached the same
system prompt and tools for any session in the last hour (Claude Code's
prefix is shared across sessions). The estimate is the average per request
over the next `horizon_requests` (default 20; one turn's tool loop is often
5–20 requests).

Without this the router would "save" money by switching and pay more in
cache rewrites than it saved.

## Cost preference

The margin is the user's cost preference: how much predicted quality a turn
may give up for a cheaper model. The presets are calibrated on
LLMRouterBench's held-out split (1,738 prompts, 13 models), against always
using GPT-5 (quality 0.596, $28.30 per 1k prompts), for the shipped trained
router (the kNN router's figures in brackets):

| Preset | Margin | Quality | Cost / 1k prompts |
|---|---|---|---|
| Best quality | 0.01 | 0.626 (0.615) | $26.52, −6% ($24.40, −14%) |
| Balanced (default) | 0.05 | 0.612 (0.596) | $19.79, −30% ($17.25, −39%) |
| Save more | 0.08 | 0.590 (0.580) | $14.82, −48% ($12.06, −57%) |

`0` (always the top prediction) is allowed, but on the benchmark it cost
*more* than always using GPT-5 for no measurable gain.

Three ways to set it:

* **From the dashboard.** Saved in `~/.aar-gateway/preferences.json` and kept
  across restarts.
* **With `POST /api/preferences`**, sending `{"preset": "save_more"}` or
  `{"quality_margin": 0.1}`.
* **For one Claude Code session only**, with a header:

```bash
ANTHROPIC_CUSTOM_HEADERS="x-aar-quality-margin: 0.1" claude
```

### Cost weight (λ) instead of the margin

λ is a second way to state the same preference. With λ set, the gateway takes the
model with the best predicted quality minus λ times its cost **relative to the
dearest eligible model for this request**:

    u(model) = predicted quality − λ × estimated cost / estimated cost of the dearest eligible model

Because the cost is relative, one λ means the same on a short request and a
200k-token one: λ 0.1 means "give up at most 0.1 predicted quality to go from
the dearest model's price to free". λ 0 always takes the top prediction; λ 1
almost always takes the cheapest model. While λ is set the margin is not used;
switching λ off (`null`) brings the margin back. The switch-margin stickiness
for upgrades applies either way.

Set it from the dashboard (Advanced settings → Cost weight), with
`POST /api/preferences` sending `{"cost_weight": 0.05}`, `{"cost_preset": "balanced"}`
or `{"cost_weight": null}` to switch it off, or for one session:

```bash
ANTHROPIC_CUSTOM_HEADERS="x-aar-cost-weight: 0.05" claude    # "off" = use the margin
```

The λ presets are calibrated for the savings router `manifold-nirt-claude-judged`
(below), out-of-fold on 99 LLM-judged mid-conversation turns, folds by
conversation, against always using Opus (quality 0.711):

| Preset | λ | Quality | Saved vs Opus | vs mix |
|---|---|---|---|---|
| `quality` | 0.02 | 0.706 | 24% | +0.009 |
| `balanced` | 0.2 | 0.665 | 72% | +0.061 |
| `save_more` | 0.3 | 0.657 | 76% | +0.073 |

λ between 0.03 and 0.1 sits on or below the mix line with this file (it moves
turns Opus → Sonnet without picking the right ones); skip that range. On a prompt
source the file never trained on, `save_more` measured +0.029 vs mix at 73% saved
(−0.081 quality), and on a separately graded 65-prompt pool the file did not beat
the mix line: the presets transfer to traffic like the judged turns, not to all
traffic. With another savings router file, re-measure before trusting them.

### The judged manifold router (`manifold-nirt-claude-judged`)

The default savings router (`router.savings_artifact`). A profile-manifold NIRT
(`investigations/profile_manifold/`) fine-tuned on the
99 judged turns (`scripts/routing/export_judged_router.py`: three seeds, every
weight at lr 1e-3 plus per-tier offsets, folded over the tiers' own profiles,
their logits averaged into one `mlp` file). It encodes prompts with raw BERT,
which fastembed doesn't ship: export the encoder once with
`python scripts/routing/export_bert_encoder.py` (writes
`~/.aar-gateway/encoders/bert-base-uncased-mean`, 436 MB; checked against the
training encoder to 2e-6). Until that folder exists the savings router fails to load: λ turns
then use the margin router, and `/health` shows the error under `savings.error`. Margin turns
don't need the encoder.

### Router files that score the tiers directly

The benchmark routers (`llmrouterbench-mlp`, `llmrouterbench-nirt`) predict quality for benchmark models and read each tier
through its `proxy`. A file whose columns are the tier ids themselves (exported
with `scripts/routing/export_router.py --kind nirt --pool
claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5`, which scores each tier from
its own profile in `configs/model_profiles.yaml`) is read directly: a tier whose
proxy is not in the file but whose own id is uses its own column, with no config
change. The decision log then shows `quality_source: router scores the models directly`.

## Dashboard

`aar-gateway dashboard` opens `http://127.0.0.1:8787/dashboard`. It shows:

* **Plan usage** against no router and against always the strongest model,
  with requests per minute over the last 30 minutes.
* **Cost preference**: Best quality, Balanced (the default) or Save more.
* **Your sessions**: each Claude Code session's cost, what it would have cost
  on the models it asked for, and the saving.
* **Recent requests**, 12 a page: the model asked for and the one that
  answered, why, cost and saving. Hover a row for tokens and latency.
* Collapsed: **Advanced settings** (below) and a **breakdown** by model and
  by subagent kind, including the average size of what each kind reports
  back.

It is live: the gateway pushes an event the moment it logs a request
(`GET /api/events`), and the page refreshes then; a 30-second poll is only a
safety net. Re-reading the log parses just the lines added since the last
read. Tables fit the page width; minor columns drop out on narrow screens.

**What is logged.** One JSON line per request in `AAR_GATEWAY_LOG` (default
`~/.aar-gateway/requests.jsonl`): the asked-for and routed model and why,
tokens (input, cache read, cache write, output), latency, status, fallback,
and three costs: this request at API prices, the same on the model asked for
(for a subagent, the session's own model), and on `interceptor.strongest`.
The baselines reuse the request's token counts, adjusted for what that model
would have had cached. Never headers, credentials or prompt text.

**Plan usage, not dollars.** On a subscription nothing is billed per token.
Scaling by API price gives the relative cost of the same work on each model,
the best available proxy for how much of the plan it used. Prices are in
[`aar_gateway/pricing.py`](../src/aar_gateway/pricing.py).

### Advanced settings

* **Routing**: Auto (default), Pin a model (every agent turn and subagent on
  the model you pick, even a pricier one; a request it can't take keeps the
  model Claude Code asked for), or Off (every request as asked).
* **Quality margin**: the number behind the three presets.
* **Let hard turns go up a tier** (off by default): the router may pick a
  model pricier than the one asked for, e.g. a Haiku session sends a hard
  question to Opus. Off, routing only ever saves.
* **Subagent tiers**: each kind's model, or "session's model"; "default"
  undoes it.

Choices apply from the next turn and survive restarts
(`~/.aar-gateway/preferences.json`). Each change is also appended to
`~/.aar-gateway/overrides.jsonl`: if a kind keeps being moved up, its default
tier is wrong.


## Other models (API keys, Ollama)

**Local models are opt-in.** `aar-gateway local add <model>` pulls a model
with Ollama if needed and adds it to `~/.aar-gateway/local-models.yaml`, a
file of its own that is merged into the configured models (it never replaces
one, and the main config is never rewritten). `aar-gateway local` reports
whether Ollama is installed and running, with the install command for your
OS when it isn't. The router never picks a local model by itself (it has no
stand-in in the benchmark); pin it, or pass `--model`.


A model with `lane: openai` is any OpenAI-compatible endpoint: OpenAI,
Gemini's compatible endpoint, OpenRouter, Groq, vLLM, Ollama.
[`aar_gateway/translate.py`](../src/aar_gateway/translate.py) converts the
Anthropic request and converts the answer back, so the agent's tool loop runs
unchanged. It carries tools, tool calls, tool results, images,
mid-conversation system messages and streaming.

The rule is **lossless or not at all**. A request that wouldn't survive the
trip is never routed there:

* more tools than `max_tools`;
* server tools;
* documents;
* images inside tool results.

Thinking blocks and cache markers are dropped, because no other provider can
use them.

Keys come from the environment variable named in `api_key_env`; a missing one
fails at startup. A model on this lane needs no Anthropic login: name it as
the model (`claude --model llama3.1:latest`, with the gateway on) or let the router pick it.

What this means for Claude Code, checked live:

* With `ENABLE_TOOL_SEARCH=true`, Claude Code sends ~12 ordinary tools and
  loads the rest itself through its `ToolSearch` tool, so its requests *are*
  translatable. The prompt is ~40k tokens: it fits API models (DeepSeek, GPT,
  Gemini), not a 32k local model, which the context-window filter enforces.
* Without tool search, the same request carries 180+ tool schemas (~245k
  tokens) and fits nothing but Claude.
* Local models under Ollama: `llama3.1` returns real tool calls (a
  tool → result → answer loop through the gateway works); `qwen2.5-coder:14b`
  writes tool calls as plain JSON text, which the agent can't act on.

## Training data and router choice

**Searched for:** routing data covering current Claude models (Hugging Face
Hub, October 2026).

* **No public dataset scores Opus, Sonnet and Haiku 4.x/5.x on the same
  prompts.** That data only comes from running them, or from graded gateway
  traffic.
* **LLMRouterBench** (`NPULH/LLMRouterBench`, ACL 2026) is the closest broad
  pool: 13 models from 2025, including Claude Sonnet 4, with real per-token
  costs. The repo already had its pipeline (`configs/llmrouterbench.yaml`);
  it had never been run. Running it found:
  * the archive layout had changed (`results/bench-release/` now);
  * 10 records carry negative token counts (now unknown, not dropped);
  * one excluded 14 MB file takes ~15 minutes to `json.loads`, so files
    outside the setting are now skipped by path before being read.
* **SWE-Router** (`SWE-Router/swebench-verified-*`) has agentic SWE-bench
  Verified runs, with pass/fail and cost, for Claude Opus 4.7, GPT-5.2,
  GPT-5 mini/nano, Gemini 3/3.1/2.5 Pro and DeepSeek V3.2/V4 Flash. Not
  merged into the training pool: its agent harness differs from
  LLMRouterBench's single-shot SWE-bench, so the two mix up "better model"
  with "better harness". On its own it shows the headroom on agentic coding:

  | | Solved | Cost (500 tasks) |
  |---|---|---|
  | Always Opus 4.7 | 78% | $158 |
  | Perfect router | 88% | $67 |

  DeepSeek V4 Flash, 22x cheaper per task, also solves 76% of the tasks
  Opus 4.7 solves.

**Results** on LLMRouterBench's held-out split, same decision rule as the
gateway:

| Policy | Trained NIRT | kNN (k=25) |
|---|---|---|
| margin 0.01 | 0.610 @ $29.06 | **0.615 @ $24.40** |
| margin 0.05 | 0.575 @ $11.37 | **0.596 @ $17.25** |
| margin 0.08 | 0.533 @ $0.93 (collapses to the cheapest) | **0.580 @ $12.06** |

Always GPT-5 scores 0.596 @ $28.30; the oracle 0.817 @ $3.96.

The trained NIRT (AUC 0.78, ECE 0.044) does not beat nearest neighbours, so
the gateway uses **kNN** (`interceptor.router.kind: knn`). This agrees with
the repo's earlier finding that the decision rule matters more than the
predictor. kNN loads in ~7 s and scores a prompt in ~17 ms.

To reproduce:

```bash
python scripts/data/download_llmrouterbench.py --config configs/llmrouterbench.yaml
python scripts/data/build_response_matrix.py   --config configs/llmrouterbench.yaml
python scripts/data/build_splits.py             --config configs/llmrouterbench.yaml
OMP_NUM_THREADS=1 python -m training.phase0 --config configs/llmrouterbench.yaml \
    --sources llmrouterbench --embeddings --taxonomy --query-bank
OMP_NUM_THREADS=1 python scripts/margin_sweep.py --config configs/llmrouterbench.yaml --router knn --k 25
```

On Apple silicon add `ROUTER_EMBED_DEVICE=mps` for the BERT step. Keep the
6.5 GB of results out of iCloud-synced folders (`~/Documents`, `~/Desktop`);
syncing them stalls every file operation in the project.

## Grading your own traffic

The benchmark stand-ins are a starting point. The real fix is routing data
from your own Claude Code work, graded on the actual Claude tiers.

1. **Capture.** Nothing to switch on: for every new turn of an agent
   conversation, the gateway keeps your message (only your words, never the
   conversation or tool results), the folder Claude Code ran in, and how the
   turn went, in the on-device history `~/.aar-gateway/history/` (on this
   computer only; `aar-gateway history off` stops it). `aar-gateway-grade`
   reads it, plus messages mined from Claude Code's own history into
   `~/.aar-gateway/turns.jsonl`.
2. **Grade.** Run `aar-gateway-grade --n 20`
   ([`evaluation/gateway/grade.py`](../src/evaluation/gateway/grade.py)).
   * **Replay:** each captured message is replayed on Opus 5.5, Sonnet 5.5
     and Haiku 5.5 through the `claude` CLI, on your own login, in the
     original folder, read-only (`--permission-mode plan`). The calls go
     straight to Anthropic, not through the gateway.
   * **Judge:** one call per message, with the answers anonymised and
     shuffled, scores each 0–10: "would a careful senior engineer accept
     this as correct and complete?"
   * **Output:** one row per (message, model) in `~/.aar-gateway/graded.jsonl`.
   * **Cost:** each graded message is 4 calls on your plan. `--n` bounds it,
     and nothing is graded twice. `aar-gateway-grade --summary` shows the
     per-tier results.
3. **Route on it.** Once `router.min_graded_prompts` (20) messages are
   graded, a new message with at least 3 similar graded ones (cosine ≥ 0.5,
   top 10) is routed on their grades for the real tiers: no stand-ins.
   * Otherwise the benchmark router decides, as before.
   * The dashboard marks each decision "your data" or "benchmark".
   * New grades are picked up without a restart.

**A bigger pool than your own traffic.** One person's grades are a
calibration layer, not a training set.
[`evaluation/gateway/pool.py`](../src/evaluation/gateway/pool.py) builds a
shared pool of real coding requests in `~/.cache/aar/routing-pool/`:

* `python -m evaluation.gateway.pool arena --n 120`: single-turn English
  coding prompts from LMArena (2025, CC-BY-4.0), replayed in an empty folder.
* `python -m evaluation.gateway.pool swe --n 80`: SWE-bench Verified issues,
  each replayed in its repository at the issue's commit (git worktrees),
  asking for root cause and fix. The maintainers' merged fix goes to the
  judge as a reference, so these labels rest on a known-good answer.

Grade the pool into its own file, not the one the gateway routes on:

```bash
aar-gateway-grade --turns ~/.cache/aar/routing-pool/turns.jsonl \
    --out ~/.cache/aar/routing-pool/graded.jsonl --n 25 --timeout 1200
```

Each row also gets `matches_best`: whether the judge found that answer as
good as the best of the three. "Did Haiku match the best?" is the label a
router is trained on. Rows carry `group` (the repository, or `arena`) so
train/test splits can be made by group.

**How good is routing between the tiers?**
[`evaluation/gateway/pool_eval.py`](../src/evaluation/gateway/pool_eval.py)
replays routing policies on graded prompts: each has a real score and cost
for every tier, so it shows what each policy would have picked, scored and
cost. It also shows what randomly mixing two fixed tiers achieves at the
same cost ("vs mix"); a router on that line has bought nothing with its
predictions.

```bash
python -m evaluation.gateway.pool_eval --graded ~/.cache/aar/routing-pool/graded.jsonl \
    --router-file ~/.aar-gateway/router.npz --by-source
```

On the 198-prompt pool (119 LMArena, 79 SWE-bench Verified; margin 0.05):

| Policy | Quality | vs mix | Matches best | vs always Opus |
|---|---|---|---|---|
| always Opus 5.5 | 0.883 | — | 92% | — |
| always Sonnet 5.5 | 0.805 | — | 72% | −56% usage |
| always Haiku 5.5 | 0.749 | — | 56% | −95% usage |
| benchmark router (today) | 0.828 | −0.002 | 77% | −39% usage |
| kNN on the graded pool (leave-one-out) | 0.839 | −0.005 | 81% | −29% usage |
| oracle (cheapest tier that matched the best) | 0.880 | +0.079 | 100% | −59% usage |

Today's router keeps ~94% of always-Opus quality at 39% less usage, but it
sits on the random-mix line: it isn't yet predicting *which* prompts can go
cheaper. Haiku matched the best answer on 110 of 198 prompts; the router
sent it 21. The oracle shows the headroom: 59% less usage at no loss. A
leave-one-out classifier on the benchmark router's predictions, prompt
length and source found almost no signal either (AUC 0.57 for "Haiku is
enough", 0.55 for Sonnet). The shipped trained router routes the same way
(0.849, +0.005 vs mix, −28% usage): it learned from the same stand-ins.

**Answering cheap first and escalating on a check doesn't pay off either.**
On the same 200 prompts, a checker read each prompt with Haiku's answer and
said keep it or escalate to Opus:

| Checker | Right when it says "keep" (base rate 55%) | vs mix | Checking cost |
|---|---|---|---|
| Haiku | 57% | +0.002 | $0.56 |
| Sonnet | 66% | −0.012 | $7.95 |

Haiku can't judge its own answers, and Sonnet checking costs nearly as much
as answering.

**More graded prompts of this kind won't close the gap either.** Three
checks, before spending the plan on 800 more:

* **The learning curve is flat.** Routing on the graded pool, trained on
  50, 100 or 150 prompts and tested on 40 held out (60 random splits),
  stays on the random-mix line at every size (kNN −0.000 → +0.002 →
  −0.001 vs mix; ridge +0.000 → +0.002 → +0.001).
* **The headroom is real, not judge noise.** The same stored answers for
  50 prompts were judged twice, in fresh random order. Scores correlate
  0.75 (mean difference 0.07 on a 0–1 scale); "Haiku is as good as the
  best" agrees 80% of the time; the oracle's pick agrees 72%. The oracle's
  picks from one pass, scored by the other, keep their quality (+0.038 vs
  always-Opus), so the 59% saving is there to be had.
* **The prompt doesn't say which tier is enough.** Leave-one-out AUC for
  "Haiku is enough": length, code hints and source 0.64; the benchmark
  router's predictions 0.63; the text embedding 0.57; all of them 0.64.
  For "Haiku or Sonnet is enough", all of them 0.67.

**A smarter reader of the prompt helps a little, on repo work only.**
Asked directly ("would Haiku answer this as well as the best?"), Opus and
Haiku reach AUC 0.70–0.71 with the kind of task known (today's router
0.62). Routing within each kind, half to Haiku, half to Opus, quality vs
the random-mix line:

| Scorer | General questions (120) | Repo bugs (80) |
|---|---|---|
| Today's router | −0.007 | +0.015 |
| Haiku reading the prompt (an API call per turn) | −0.009 | +0.035 |
| Opus reading the prompt | −0.008 | +0.034 |
| Student, bge-base-en-v1.5 (210 MB ONNX) | −0.014 | +0.027 |
| Student, nomic-embed-text-v1.5-Q (130 MB) | −0.014 | +0.021 |
| Student, jina-embeddings-v2-small-en (120 MB) | −0.004 | +0.014 |
| Student, all-MiniLM-L6-v2 (shipped) | −0.011 | +0.007 |
| Student, bge-small-en-v1.5 (67 MB) | −0.006 | +0.002 |

The students are small ONNX encoders (no torch, every OS) plus a numpy
MLP, trained to copy Haiku's estimate on 3,670 prompts Haiku labelled
(Arena coding questions, SWE-bench Verified issues, benchmark prompts and
the user's own turns; none of the 200 test prompts). None helps on general
questions; on repo bugs the best gets three quarters of the way to Haiku,
but 80 prompts leave about ±0.02 of noise, so none is clearly better than
today's router. Not shipped: a decision would need a larger graded test
set first.

So a per-turn router that reads only the prompt can't find most of that
saving, however much graded data it gets. What works is routing by the
kind of work instead: search subagents on Haiku save ~97% with no measured
loss (see [Subagents](#subagents)). The next gains are more of that
(delegating more of a turn's lookup work to cheap subagents) and signals
from the session itself (whether the user had to correct or redo a cheap
model's answer), not more prompts.

Known limits:

* The replay sees the repository as it is now, not as it was at the turn.
* Read-only means an edit request is graded on its plan, not on an edit.
* The judge is a Claude model, so a family preference is possible.
  Anonymising and shuffling limit that; they don't remove it.

### Turns from the middle of a conversation

The pool above is mostly first messages. Real turns usually come later in a
conversation, after "now do X" or "that didn't work". So a second pool,
built 2026-10-21, takes 100 turns that each have at least one earlier human
message. There are 25 from each source:

* the user's own Claude Code sessions;
* [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat): other
  developers' Claude Code sessions;
* [ConvCodeBench](https://huggingface.co/datasets/ConvCodeWorld/convcodebench):
  a coding task, the model's failed attempt, then the test output and a
  reviewer's notes;
* [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M): English
  coding chats with ChatGPT.

How a turn is answered and graded:

* The turn is the last few exchanges (text only, up to 3,000 characters)
  followed by the message.
* Each tier answers the whole turn. The router sees only the message, as the
  live gateway does (`human_text`).
* Sonnet 5.5 judges, not Opus, to save tokens, so the scores are a little
  off the 200-prompt pool's. Compare the savings and vs mix columns, which
  depend on the gaps between models, not the absolute scores.
* Two windows of plan usage graded 99 of the 100 turns: $30 of
  API-equivalent answers, plus the judge calls, which aren't metered.

Build the pool with `python -m evaluation.gateway.multiturn
{own,swe-chat,convcode,wildchat} --n 25`, then grade with `aar-gateway-grade
--turns … --out … --judge claude-sonnet-5-5`.

| Policy (99 turns) | Quality | vs mix | Below Opus |
|---|---|---|---|
| Always Opus | 0.711 | — | 0% |
| Always Sonnet | 0.678 | — | 56% |
| Today's router | 0.702 | +0.005 | 23% |
| Pool kNN (leave-one-out) | 0.705 | +0.016 | 38% |
| Oracle | 0.760 | +0.073 | 42% |

| Source | Router: quality (Opus) | Router: below Opus | Router: vs mix | Oracle: below Opus |
|---|---|---|---|---|
| Own sessions | 0.633 (0.658) | 19% | +0.016 | 27% |
| SWE-chat | 0.640 (0.668) | 16% | −0.018 | 57% |
| ConvCodeBench | 0.784 (0.740) | 64% | +0.033 | 69% |
| WildChat | 0.748 (0.776) | 11% | −0.015 | 46% |

What the second pool shows:

* **The same as single prompts.** Today's router sits on the mix line
  (+0.005) and saves about a quarter of Opus usage. The oracle shows about
  twice that saving is there to find.
* **Fixing after failed tests suits Sonnet.** On ConvCodeBench, Sonnet
  outscores Opus (0.83 vs 0.74), and the router's mostly-Sonnet picks keep
  Opus quality at a third of its usage.
* **Mid-conversation follow-ups.** On these (own sessions, SWE-chat,
  WildChat) the router sends roughly three in four turns to Opus. A message
  like "did you test it?" says little about the work by itself, which is why
  routing on the kind of work and on session signals is the next step, not a
  better prompt reader.
* **Noise.** 25 turns per source leaves about ±0.03 of noise on the per-source
  vs mix figures.

The installed routers on the same 99 turns. As in [Measured against
installed routers](#measured-against-installed-routers), every router got a
stub instead of a real model, so this cost $0. Each request carried the whole
turn as one message, so here the live gateway also saw the earlier
conversation:

| Router | Quality | vs mix | Below Opus | H/S/O |
|---|---|---|---|---|
| aar-gateway MLP (live default) | 0.712 | +0.010 | 15% | 4/28/67 |
| aar-gateway NIRT | 0.711 | +0.000 | 0% | 0/0/99 |
| LiteLLM complexity router | 0.678 | −0.002 | 54% | 2/92/5 |
| claude-code-router, claude-code-proxy, RouteLLM | ≈0.711 | ≈0 | ≈0% | (almost) all Opus |

No router beats the mix line by more than the noise. LiteLLM's saving is
what always-Sonnet gives (56% below Opus, quality 0.678). The NIRT router
sent every turn to Opus, and both aar routers failed the "lookup and design
get different tiers" activation check in this run. Run report:
`~/.aar-router-bench/results/20261021-225549/`.

## Subagents

Claude Code's subagents (Agent tool spawns) each run in a fresh context and
return only their final message to the main agent, so the work that fills a
context window can run on a cheaper model without the main conversation
paying for it. [`subagents.py`](../src/aar_gateway/subagents.py) gives
each one a tier.

**Where the model is set: at spawn time.** Before Claude Code runs any tool
it runs its `PreToolUse` hooks with the call's input. `aar-gateway start`
installs one for the Agent tool (`aar-gateway hook`). It sends the call's
`subagent_type` and `prompt` to the gateway (`POST /api/subagent-spawn`) and
returns the call with `model` set (`haiku`, `sonnet` or `opus`). Claude Code
then starts the subagent on that model. Verified on Claude Code 2.1.296: the
subagent ran on Haiku 5.5 while the main agent stayed on Opus 5.5.

* If the gateway is down, the call's tool isn't Agent, or there's no
  decision, the hook prints nothing and the call goes ahead unchanged.
* A `model` the main agent set itself is left alone.
* Routing Off or Pin (dashboard) leaves spawns alone.
* `AAR_GATEWAY_HOOK=0` stops `aar-gateway start` from installing the hook.

**At the gateway, a fallback.** A subagent's requests carry
`cc_is_subagent=true` in their system prompt's billing line (undocumented,
seen on 2.1.293–2.1.296). A subagent the hook didn't see is still
recognised there and routed the same way. One it did see is labelled "set
at spawn".

**Three kinds**, by what the subagent may do:

| Kind | Tools | Tier | Output budget in its prompt |
|---|---|---|---|
| `lookup` | read and run, no edits | Haiku 5.5 | ≤ 10 bullets of `path:line` + one sentence, or the exact structured result |
| `make` | all | Sonnet 5.5 | ≤ 10 one-line items: files touched, tests, findings |
| `judge` | read and run, no edits | session's model | conclusion, ≤ 8 findings with evidence, next step |
| Explore (built-in) | Claude Code's | Haiku 5.5 | Claude Code's |
| Plan (built-in) | Claude Code's | Sonnet 5.5 | Claude Code's |

A subagent's kind, first match wins:

1. Its `subagent_type` is one of ours.
2. It's Claude Code's Explore or Plan.
3. Keywords at the start of its task:
   * root cause, debug, architecture, trade-offs → judge;
   * a task opening with write/generate/create/build/implement/fix/refactor,
     or asking to review, write tests, plan, design or propose → make;
   * find, locate, search, list, count, extract, label, summarise → lookup.

No match: the general router decides, as for any turn.

**Rules.**

* A tier is a ceiling on cost, never an upgrade: it never moves a subagent
  above the session's model.
* Each subagent's tool loop stays on the model its first request got.
* Moving a subagent never costs the main conversation a cache rewrite: its
  system prompt and tools differ, so its prompt cache is separate.
* A subagent's saving is measured against the session's own model, not the
  model the hook set.
* Each subagent's final report is logged (`report_tokens`): it's what the
  main agent carries for the rest of the conversation, and the number that
  says whether the output budgets work.

**The definitions work without the gateway too.** Each pins its tier with
Claude Code's own `model:` field. `aar-gateway agents --install` writes them
to `~/.claude/agents/` (existing files are kept); `aar-gateway agents` prints
them as `--agents` JSON. `AAR_GATEWAY_AGENTS=0`, or passing your own
`--agents`, stops `aar-gateway start` from adding them.

**Config.** `interceptor.subagents.roles` maps a kind to a model (`null`
keeps the session's); `enabled: false` routes subagents like any turn.

**Baseline from subagents you already ran.** `aar-gateway-subagents` reads
Claude Code's own subagent transcripts
(`~/.claude/projects/*/<session>/subagents/`), gives each the kind the
gateway would, and prices its recorded tokens on the model it ran on and on
the kind's model. This measures usage, not quality.

**Grading subagent tasks.** `aar-gateway-grade --from-subagents --subagents
only --n 10` adds past subagent tasks that can still be replayed and grades
them on each tier. The on-device history also keeps each new subagent's
task, tagged with its kind.

## Configuration

Everything is in [`gateway.example.yaml`](../src/aar_gateway/gateway.example.yaml), commented. Environment
variables:

* `AAR_GATEWAY_CONFIG`: path to the config.
* `AAR_GATEWAY_HOST` / `AAR_GATEWAY_PORT`: default `127.0.0.1:8787`.
* `AAR_GATEWAY_LOG`: the request log.
* `AAR_GATEWAY_UPSTREAM`: default `https://api.anthropic.com`.
* `AAR_GATEWAY_ALLOWED_HOSTS`: extra host names the gateway answers to,
  comma-separated (by default only `127.0.0.1`, `localhost`, `[::1]`).

A request with no Anthropic credentials, for a model not on an `openai`
lane, gets a 401: the gateway has nothing of its own to answer with.

**Only this machine's programs, never a web page.** A page open in your
browser can send requests to `localhost`. The gateway refuses any request
whose `Origin` header names a site other than its own dashboard, and any
whose `Host` isn't a local name (a "DNS rebinding" page, whose own domain
points at your machine). Without this, a site you visit could switch routing
off or spend an `openai`-lane API key through the gateway. Claude Code and
the CLI send no `Origin`, so they are unaffected.

**If the router can't load** (its data missing, a broken run), the gateway
still starts and forwards every request as asked. `/health` and the
dashboard say why.

**A model Anthropic rejects** for your Claude Code version (a 400) is not
tried again for 10 minutes. A 400 about the request itself (prompt too long,
image too large) never blocks the model.

The request log rotates at 50 MB into `requests.jsonl.1`; the dashboard reads
both, so totals survive one rotation.

## Verified end to end

Each of these was run through real Claude Code on a Claude subscription,
with the gateway moving the turn from Opus 5.5 to Sonnet 5.5:

| What | Claude Code did | Result |
|---|---|---|
| Multi-step tool loop | `Bash` (count files), then `Read` (a source file) | correct answer; every request of the loop stayed on Sonnet ("same turn") |
| Skill | `Skill(aar-architecture)` | skill loaded and answered |
| On-demand tool | `ToolSearch(select:WebFetch)`, then `WebFetch` | fetched and answered |
| Non-Claude model | tool call → tool result → streamed answer, via local `llama3.1` | correct answer |

## Compared with claude-code-router

claude-code-router (CCR) routes with fixed rules, per request:

* `provider,model` in the model name: that model;
* a prompt over `longContextThreshold` (60k): the long-context model;
* a `<CCR-SUBAGENT-MODEL>` tag in a subagent's system prompt: that model;
* a Haiku request (Claude Code's background work): the background model;
* a `web_search` tool: the web-search model;
* `thinking` present: the think model;
* otherwise: the default.

It also has per-model request transformers, and a status line. This gateway
borrows the status line and the session id (read from `metadata.user_id`,
as CCR does). For the rest, the trained router replaces the rules:

* **Difficulty.** A per-prompt quality prediction stands in for the coarse
  "thinking on → big model" rule.
* **Long context and web search.** Hard filters (context window, lossless
  translation) cover them.
* **Cost.** Cost is estimated with the prompt cache in it, which no CCR rule
  sees.
* **Background requests** stay on the model Claude Code asked for: they are
  ~1k tokens, and some models reject their settings.

### Measured against installed routers

[`evaluation/gateway/router_compare.py`](../../src/evaluation/gateway/router_compare.py)
runs real routers, each installed in its own environment
([`scripts/router_bench/install.ps1`](../../scripts/router_bench/install.ps1)), with
their upstream pointed at a local stub. Every graded prompt goes through each router
as a Claude Code turn. The stub records which tier the router forwarded to, and that
tier's graded score and usage are what the decision is worth. This compares routing
policies: it spends nothing, and it never judges the stub's canned answers.

```bash
python -m evaluation.gateway.router_compare capture   # one `claude -p "hi"`, to the stub only
python -m evaluation.gateway.router_compare run --graded ~/.aar-gateway/eval-rules/graded.jsonl \
    --graded ~/.aar-gateway/eval40/graded.jsonl --sensitivity --smoke
```

On 65 graded prompts, with the request asking for Opus (Claude Code's main loop),
2026-10-20:

| Router | Quality | vs mix | Matches best | Est. usage | Saved vs no router [95% CI] | H/S/O |
|---|---|---|---|---|---|---|
| aar-gateway, MLP router (live default, no own grades) | 0.706 | +0.011 | 74% | $54.66 | $2.98, 5% [−0.14, 8.30] | 3/6/56 |
| aar-gateway, NIRT router (`llmrouterbench-nirt`) | 0.703 | 0 | 72% | $57.64 | $0, 0% | 0/0/65 |
| claude-code-router 1.0.73 | 0.703 | 0 | 72% | $57.64 | $0, 0% | 0/0/65 |
| claude-code-proxy | 0.703 | 0 | 72% | $57.64 | $0, 0% | 0/0/65 |
| LiteLLM 1.106.1 complexity router | 0.662 | +0.005 | 49% | $33.34 | $24.30, 42% [17.08, 32.28] | 15/46/4 |
| LiteLLM auto-router (our tier utterances) | 0.678 | −0.002 | 63% | $48.70 | $8.94, 16% [5.01, 13.87] | 4/21/40 |
| RouteLLM 0.2.0 bert, Opus/Haiku | 0.703 | 0 | 72% | $57.64 | $0, 0% | 0/0/65 |
| oracle (ceiling, not a router) | 0.748 | +0.091 | 100% | $34.17 | $23.47, 41% | 31/15/19 |

"No router" is every prompt on the model Claude Code asked for (the gateway's
`vs_no_router`). The run reports it for every requested model: asked for Sonnet, the MLP
router saves 82% and the NIRT router 83% (both send nearly everything to Haiku, ΔQ −0.015,
CI includes 0). claude-code-router then costs 41% *more* than no router, because its `think`
rule upgrades to Opus.

**The NIRT router never leaves Opus at the default margin.** `router.margin` is the
gateway's cost aversion, and NIRT's smallest predicted Opus-over-Haiku gap on these prompts is
0.071, above the default 0.05. `--margins 0,0.05,0.1,0.15,...` sweeps it live through each
gateway's `x-aar-quality-margin` override. NIRT saves 15% at 0.125 (+0.011 vs mix) and 54% at
0.15 (ΔQ −0.032 [−0.060, −0.005], +0.017 vs mix, the most above the mix line of any router
here). Picking the margin on a random half of the prompts and scoring it on the other half
(200 splits), NIRT holds +0.010 vs mix, saving 40% at ΔQ −0.026, and picks 0.125–0.15 in 95%
of splits; MLP holds 0.000, saving 28%. A margin near 0.15 for the NIRT file is worth testing on
fresh graded data before it becomes the default. NIRT never picks Sonnet.

* **No router predicts which prompts can go cheaper.** Every row sits within ±0.013 of the
  random-mix line. The oracle is +0.09 above it.
* **aar-gateway (MLP) vs CCR: no detectable difference.** Paired bootstrap 95% CI: quality
  −0.003 [−0.012, +0.006], usage +$2.98 [−0.14, +8.30].
* **LiteLLM's complexity router saves the most, and pays for it in quality.** Usage is
  −$21 [−30, −14] vs aar and quality −0.045 [−0.080, −0.009]. That puts it on the mix line:
  it costs the same as randomly mixing Haiku and Sonnet.
* **CCR on today's Claude Code is always Opus.** Claude Code sends `thinking` on every
  main-loop turn, so the `think` rule fires every time. Without `thinking`, it is always
  Sonnet (`default`). All four of its rules activate as documented.
* **claude-code-proxy is a name mapper, not a router.** haiku→SMALL, sonnet→BIG, anything
  else passes through. It also rejects current Claude Code requests (HTTP 422 on the
  trailing `system` message), so `claude -p` through it fails.
* **RouteLLM's Arena calibration doesn't transfer to coding prompts.** The threshold
  calibrated for aar's 86% Opus share sends 100%, and one calibrated for 10% sends 26%. No
  point on its cost/quality curve is more than +0.013 above the mix line.
* **LiteLLM's auto-router changes with the request shape.** It embeds the whole last user
  message, so Claude Code's `<system-reminder>` blocks make it look simple, and it sends 62
  of 65 prompts to Haiku in the Claude Code track, against 4 in the plain-API track.
* **aar-gateway honours the requested ceiling, and that dominates.** Asked for Sonnet, it
  sends 64 of 65 prompts to Haiku: the predicted gap between Haiku and Sonnet is within
  the 0.05 margin.

**The ZOIB router, if it were integrated.** The ZOIB run (`lrb-zoib-projected`: 8-d,
interaction term, taxonomy relevance) doesn't fold into the `mlp` router file, so it can't run
behind the gateway yet. [`evaluation/gateway/zoib_replay.py`](../../src/evaluation/gateway/zoib_replay.py)
replays it the way the gateway would: text → MiniLM → relevance from the 41 cluster centroids →
ZOIB expected score → the tiers' stand-ins → the margin rule. From raw text it reproduces the
run's own predictions to 7e-7 on 300 held-out queries, and the MLP and NIRT files replayed the
same way give exactly the live gateways' picks. Asked for Opus:

| Margin | ZOIB saved | ZOIB ΔQ vs Opus [95% CI] | ZOIB vs mix | NIRT saved | MLP saved |
|---|---|---|---|---|---|
| 0.05 (default) | 0% | −0.002 | −0.001 | 0% | 5% |
| 0.075 | 15% | −0.009 [−0.022, +0.002] | +0.013 | 0% | 16% |
| 0.10 | 34% | −0.023 [−0.045, −0.000] | +0.021 | 4% | 34% |
| 0.125 | 56% | −0.038 [−0.071, −0.005] | +0.012 | 15% | 54% |
| 0.15 | 75% | −0.063 [−0.098, −0.026] | −0.008 | 54% | 79% |

Margin chosen on one half of the prompts and scored on the other (200 splits): ZOIB +0.008 vs mix,
33% saved, ΔQ −0.026 (it chooses 0.075–0.125); NIRT +0.010, 40%; MLP 0.000, 28%. ZOIB and NIRT
are both a little above the mix line once the margin is wider, and can't be told apart on 65
prompts. Like NIRT, ZOIB never picks Sonnet.

Caveats: 65 prompts, so quality gaps under ~0.03 are noise. The usage figures are what the
grading replays metered, at API prices. Every router saw a cold cache. aar-gateway's
`/health` reported the shipped router loaded with 0 graded prompts of its own.

## Not done yet

* **Your graded data starts empty.** Until about 20 of your turns are
  graded (`aar-gateway-grade`), decisions rest on 2025 benchmark models
  through stand-ins.
* **No grading of live answers.** Grading replays captured turns afterwards.
  It does not check, in the moment, whether the answer you got held up.
* **No multi-user hosting.** One local process per person, like
  claude-code-router; it binds to `127.0.0.1`.
* **Haiku 4.5 can't serve Claude Code.** It rejects the `effort` setting
  Claude Code sends. Haiku 5.5, which accepts it, is the fast tier.
