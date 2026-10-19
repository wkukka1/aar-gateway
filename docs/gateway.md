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

aar-gateway code          # starts the gateway if needed, then runs Claude Code through it
aar-gateway dashboard     # opens http://127.0.0.1:8787/dashboard
aar-gateway doctor        # if anything looks wrong
```

Or run it without installing: `uvx --from git+https://github.com/wkukka1/aar-gateway aar-gateway setup`.
From a checkout: `uv sync --extra dev`. The router file is described
under [The router file](#the-router-file).

The config is `AAR_GATEWAY_CONFIG` if set, else `~/.aar-gateway/config.yaml`
if it exists, else the defaults (router off).

## The router file

The gateway serves its router from one small file. **The shipped one is a
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

## Managing the gateway

| Command | What it does |
|---|---|
| `aar-gateway start` / `stop` / `restart` | the gateway in the background (`code` and `dashboard` start it if needed) |
| `aar-gateway status` | running or not, router loaded or why not, config, requests and saving, where the logs are |
| `aar-gateway logs [-n 50] [-f]` | the gateway's own log; `-f` follows it |
| `aar-gateway doctor` | checks Claude Code, the config, the router, the port and the log folder, with a fix for each problem |
| `aar-gateway config` | which config file is in use |
| `aar-gateway setup [--router FILE\|URL] [--force]` | install the router file and write a config (existing configs are kept without `--force`) |
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

`aar-gateway code` passes every further argument to `claude`
(`aar-gateway code -p "..."`, `aar-gateway code -c`). It is the same as
running `ENABLE_TOOL_SEARCH=true ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude`
with the gateway already up (`aar-gateway start`). It also sets Claude Code's
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
`export PYTHONPATH=$PWD/src`. Run from the repo root: the router loads its
trained run from `data/processed/nirt_runs/` (or set `interceptor.router.runs_dir`).

## Check it works

```bash
curl -s localhost:8787/health          # router on? graded prompts + calibration offsets
aar-gateway code -p "Read aar-gateway/src/aar_gateway/graded.py and explain min_prompts in two sentences."
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
ANTHROPIC_CUSTOM_HEADERS="x-aar-quality-margin: 0.1" aar-gateway code
```

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
the model (`aar-gateway code --model llama3.1:latest`) or let the router pick it.

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

## Subagents

Claude Code's subagents (Agent tool spawns) each run in a fresh context and
return only their final message to the main agent, so the work that fills a
context window can run on a cheaper model without the main conversation
paying for it. [`subagents.py`](../src/aar_gateway/subagents.py) gives
each one a tier.

**Where the model is set: at spawn time.** Before Claude Code runs any tool
it runs its `PreToolUse` hooks with the call's input. `aar-gateway code`
installs one for the Agent tool (`aar-gateway hook`). It sends the call's
`subagent_type` and `prompt` to the gateway (`POST /api/subagent-spawn`) and
returns the call with `model` set (`haiku`, `sonnet` or `opus`). Claude Code
then starts the subagent on that model. Verified on Claude Code 2.1.296: the
subagent ran on Haiku 5.5 while the main agent stayed on Opus 5.5.

* If the gateway is down, the call's tool isn't Agent, or there's no
  decision, the hook prints nothing and the call goes ahead unchanged.
* A `model` the main agent set itself is left alone.
* Routing Off or Pin (dashboard) leaves spawns alone.
* `AAR_GATEWAY_HOOK=0` stops `aar-gateway code` from installing the hook.

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
`--agents`, stops `aar-gateway code` from adding them.

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
