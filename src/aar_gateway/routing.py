"""The interceptor's model picker: a trained router (:mod:`._core.scorers`)
scores the request, :class:`~._core.policy.MarginRoutingPolicy` decides.

Per human turn (not per request -- the interceptor keeps a turn's tool loop on
one model), for every configured model the user could be moved to:

* **quality** -- the router's predicted quality for the model's ``proxy`` in
  its training pool (the shipped router knows 2023 RouterBench models, so
  proxies map current models onto its tiers until it is retrained);
* **cost** -- estimated dollars per request on that model *including the
  prompt cache*, averaged over the next ``horizon_requests``. Caches are per
  model: a model that already holds the prefix reads it at the cache-read
  rate, any other rewrites it at the cache-write rate. Claude Code's prefix is
  ~200k tokens and one turn's tool loop re-sends it many times, so a switch
  pays off only if the cheaper reads over the horizon beat the one-time
  rewrite -- which is what stops the router "saving" money by switching.

Hard filters first -- never pricier than the model the user chose, prompt fits
the context window, and for a non-Anthropic model the request translates
losslessly (:func:`~aar_gateway.translate.untranslatable_reason`) -- then the
policy takes the cheapest model within ``margin`` predicted quality of the best.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from ._core.constraints import OptimizationObjective, RoutingConstraints
from ._core.context import RoutingContext, RoutingRequest
from ._core.decision import ModelScore
from ._core.profile import LLMProfile
from ._core.policy import MarginRoutingPolicy
from ._core.signals import PromptSignals
from .pricing import CLAUDE_PRICES, WRITE_1H, WRITE_5M, cache_read_rate, canonical_model, has_price, price_at
from .translate import untranslatable_reason

#: ``(body, ctx) -> decision dict`` or ``None`` (keep the client's model).
#: ``ctx``: requested, human_text, est_prompt_tokens, last_model, last_prompt_tokens,
#: quality_margin (the user's cost preference; ``None`` = the configured default),
#: cost_weight (lambda on relative cost; when set it replaces the margin rule),
#: warm_tokens ({model: prefix tokens it has cached recently}), blocked
#: ({model: why the upstream rejected it for this client}).
ModelPicker = Callable[[dict, dict], dict | None]

log = logging.getLogger(__name__)


def incompatibility(body: Mapping, spec: Mapping) -> str | None:
    """Request settings a model is known to reject with a 400 (``rejects`` in
    its config). Claude Code's agent requests use adaptive thinking and
    ``effort``; its side requests switch thinking off."""
    rejects = set(spec.get("rejects") or ())
    thinking = (body.get("thinking") or {}).get("type")
    if "effort" in rejects and (body.get("output_config") or {}).get("effort"):
        return "rejects output_config.effort"
    if "thinking_disabled" in rejects and thinking == "disabled":
        return "rejects thinking disabled"
    if "adaptive_thinking" in rejects and thinking == "adaptive":
        return "rejects adaptive thinking"
    return None


def unavailable_reason(model: str, body: Mapping, settings: Mapping, blocked: Mapping[str, str],
                       est_prompt_tokens: int) -> str | None:
    """Why ``model`` can't take this request -- not configured, rejected for
    this client before, prompt too long, rejects its settings, or not
    translatable for an ``openai`` lane -- else ``None``. The one place these
    rules live: the picker, a subagent's kind and a pinned model all ask it."""
    spec = settings["models"].get(canonical_model(model))
    if spec is None:
        return "not a configured model"
    if model in blocked:
        return f"rejected earlier for this client: {blocked[model]}"
    if spec.get("context_window") and est_prompt_tokens > spec["context_window"]:
        return "prompt exceeds context window"
    why = incompatibility(body, spec)
    if why:
        return why
    if spec.get("lane") == "openai":
        why = untranslatable_reason(body, max_tools=spec.get("max_tools"))
        return f"not translatable: {why}" if why else None
    return None


def price_for(settings: Mapping, model: str | None) -> Mapping | None:
    model = canonical_model(model)
    spec = settings["models"].get(model) or {}
    prices = spec.get("prices")  # an empty one is the config saying "no price", not "use Claude's"
    return prices if prices is not None else CLAUDE_PRICES.get(model)


def router_column(model: str, spec: Mapping, pool) -> str | None:
    """The router-file column that scores ``model``: its ``proxy`` when the file
    has it, else the model's own id when the file scores it directly (a file
    folded over the Claude tiers' own profiles), else the proxy (absent -> skipped)."""
    proxy = spec.get("proxy")
    if proxy in pool or model not in pool:
        return proxy
    return model


def load_router_model(rcfg: Mapping):
    """The trained ranker the gateway scores prompts with: the router file at
    ``artifact`` (``aar-gateway setup`` installs one) -- a trained MLP (weights
    only) or a kNN index, served with numpy and fastembed alone, no training
    data and no PyTorch."""
    if not rcfg.get("artifact"):
        raise ValueError("router.enabled needs router.artifact: the path of a router file "
                         "(run `aar-gateway setup` to install one)")
    return _load_file(rcfg["artifact"])


def _load_file(path):
    from ._core.scorers import load_router
    from .config import encoders_dir

    return load_router(path, encoders_dir=encoders_dir())


def load_savings_model(rcfg: Mapping) -> tuple[object | None, str | None]:
    """The savings router (``router.savings_artifact``) for lambda-priced turns, as
    ``(router, error)``. Off when the path is unset. A failure is returned, not raised:
    the margin router still serves, and lambda turns fall back to it."""
    if not rcfg.get("savings_artifact"):
        return None, None
    try:
        router = _load_file(rcfg["savings_artifact"])
        router.route_text(["warm up the encoder"])
        return router, None
    except Exception as exc:  # noqa: BLE001 -- a broken savings file must not take the margin router down
        log.warning("savings router not loaded, lambda turns use the margin router", exc_info=True)
        return None, f"{type(exc).__name__}: {exc}"[:400]


def calibration_offsets(graded_file: Path, router_model, settings: Mapping,
                        shrink: float = 20.0) -> dict[str, float]:
    """Per model: how far the user's graded scores sit from its stand-in's
    predictions, over the prompts the user had graded. Shrunk toward 0 by
    ``n / (n + shrink)`` so a handful of grades can't swing it. Measured on
    the first 23 graded prompts: Opus +0.27, Sonnet +0.30, Haiku +0.35 -- the
    stand-ins rated Haiku 0.10 below Opus, the user's own work 0.02."""
    from .graded import load_graded

    prompts, models, scores = load_graded(graded_file)
    if not prompts:
        return {}
    frame = router_model.route_text(prompts).score_frame().reset_index(drop=True)
    offsets = {}
    for j, model in enumerate(models):
        proxy = router_column(model, settings["models"].get(model) or {}, frame.columns)
        rows = np.flatnonzero(~np.isnan(scores[:, j]))
        if proxy not in frame.columns or not len(rows):
            continue
        gap = float(scores[rows, j].mean() - frame[proxy].iloc[rows].mean())
        offsets[model] = gap * len(rows) / (len(rows) + shrink)
    return offsets


def build_picker(settings: Mapping, router_model=None, graded=None,
                 graded_file: Path | None = None, savings_model=None) -> ModelPicker | None:
    """``graded`` (:func:`aar_gateway.graded.graded_predictor`): the
    user's own turns graded on the real tiers. When it has enough similar
    prompts it supplies quality directly, with no stand-ins; otherwise the
    benchmark ``router_model`` and each model's ``proxy`` do, each shifted by
    its :func:`calibration_offsets` from ``graded_file``.

    ``savings_model``: the router that scores lambda-priced turns (see
    :func:`load_savings_model`); with none, those turns use ``router_model``."""
    rcfg = settings["router"]
    savings_error = None
    if router_model is None:
        if not rcfg.get("enabled"):
            return None
        router_model = load_router_model(rcfg)
        router_model.route_text(["warm up the encoder"])  # first encode is slow
        if savings_model is None:
            savings_model, savings_error = load_savings_model(rcfg)
    policy = MarginRoutingPolicy(margin=float(rcfg.get("margin", 0.05)))
    lock = threading.Lock()  # one encoder forward at a time; requests run in a thread pool
    #: ``(graded file mtime, graded prompts, per-model shifts)`` as one tuple, replaced whole,
    #: so a reader without the lock never sees a half-updated calibration
    calibration: dict = {"state": (None, 0, {})}

    def offsets() -> dict[str, float]:
        """Recomputed when the graded file changes; called under ``lock``."""
        if graded_file is None:
            return {}
        try:
            mtime = os.path.getmtime(graded_file)
        except OSError:
            return {}
        if mtime != calibration["state"][0]:
            from .graded import load_graded

            calibration["state"] = (
                mtime, len(load_graded(graded_file)[0]),
                calibration_offsets(graded_file, router_model, settings,
                                    float(rcfg.get("calibration_shrink", 20))))
        return calibration["state"][2]

    def calibration_status() -> dict:
        """For ``/health``: the per-model shifts and how many graded prompts
        they rest on, as last computed. Never takes ``lock`` (held across an
        encoder forward): a ``/health`` that waited for it would time out and
        read as a dead gateway. ``pick`` refreshes it when the graded file changes."""
        _, n, shifts = calibration["state"]
        return {"graded_prompts": n, "offsets": {m: round(v, 3) for m, v in sorted(shifts.items())}}

    try:
        with lock:
            offsets()  # known from the start, not only after the first request
    except Exception:  # noqa: BLE001 -- a bad graded file must not stop the picker; pick() retries
        log.warning("could not read the graded file at startup", exc_info=True)
    write_mult = WRITE_1H if rcfg.get("cache_ttl", "1h") == "1h" else WRITE_5M
    out_tokens = float(rcfg.get("expected_output_tokens", 800))

    horizon = max(int(rcfg.get("horizon_requests", 20)), 1)
    #: from this many prompt tokens a conversation is not moved to a cheaper model. Its cache is
    #: rewritten in full by the move, and the savings assume it stays there. Seen live: 300k-735k
    #: token conversations moved down and back, paying the rewrite twice, $0.3-$5.9 a turn against
    #: $0.1-$0.3 on the model asked for. Upgrades and the model already holding the cache are unaffected.
    long_tokens = int(rcfg.get("long_conversation_tokens", 200_000))
    #: extra predicted-quality gap needed to move a conversation off its
    #: current model. The router's per-prompt scores are noisy; without this a
    #: conversation flip-flops between models turn to turn and every flip
    #: rewrites the ~200k-token prompt cache (seen live: 2% dearer than no router).
    switch_margin = float(rcfg.get("switch_margin", 0.05))
    # A conversation's difficulty changes slowly and one message's prediction
    # is noisy, so each turn's per-model prediction is blended with the
    # previous turn's: weight on the new message (1.0 = judge each message
    # alone). Found live: two messages 34 s apart flipped Opus -> Haiku ->
    # Opus, and Opus reread the whole conversation into its cache.
    smoothing = min(max(float(rcfg.get("turn_smoothing", 0.5)), 0.0), 1.0)

    def warm_tokens(model: str, ctx: Mapping) -> float:
        """Prefix tokens already in ``model``'s cache: the whole conversation
        if it served the previous request, else whatever it cached for any
        recent request -- Claude Code's system prompt and tools are the same
        across sessions, so a fresh session often starts warm."""
        if ctx.get("last_model") == model and ctx.get("last_prompt_tokens"):
            return float(ctx["last_prompt_tokens"])
        return float((ctx.get("warm_tokens") or {}).get(model, 0))

    def estimate(model: str, spec: Mapping, price: Mapping, ctx: Mapping) -> float:
        """Average $ per request over the next ``horizon`` requests if this
        model takes the conversation. Per request, not per turn -- one turn's
        tool loop is many requests over the same prefix.

        The tokens are not predicted per model: they are the conversation's
        measured per-request profile (``ctx["usage_profile"]``: new uncached
        input, new cache writes, output -- running averages from real traffic),
        and that same profile is priced on each candidate. A steady request
        reads the prefix from cache and *writes* what the conversation added
        since -- at 2x input price, which is where Opus and Sonnet differ even
        when their cache-read prices are equal. A model not yet holding the
        prefix rewrites it once on its first request."""
        total = float(ctx["est_prompt_tokens"])
        price = price_at(price, total)
        prof = ctx.get("usage_profile") or {}
        out = float(prof.get("output_tokens", out_tokens))
        if spec["lane"] != "anthropic":
            return 1e-6 * (total * price["input"] + out * price["output"])
        uncached = float(prof.get("input_tokens", 0.0))
        tail = min(float(prof.get("cache_creation_input_tokens", 0.0)), total)
        prefix = max(total - tail - uncached, 0.0)
        write = price["input"] * write_mult
        steady = (prefix * cache_read_rate(price) + tail * write + uncached * price["input"]
                  + out * price["output"])
        warm = min(warm_tokens(model, ctx), prefix)
        first = steady + (prefix - warm) * (write - cache_read_rate(price))
        return 1e-6 * (first + (horizon - 1) * steady) / horizon

    def pick(body: dict, ctx: dict) -> dict | None:
        requested = canonical_model(ctx["requested"])
        if requested not in settings["models"]:
            return None
        # by default a turn can only move to a cheaper model than the one asked
        # for; with ``allow_upgrade`` a hard turn may go up (e.g. Haiku -> Opus)
        ceiling = None if ctx.get("allow_upgrade") else price_for(settings, requested)
        own = graded(ctx["human_text"]) if graded is not None else None
        use_graded = bool(own) and len(own["quality"]) >= 2
        lam = ctx.get("cost_weight")
        # lambda turns are priced by the savings router (calibrated for lambda, scores the tiers
        # directly); margin turns by the margin router, shifted by the user's grades
        active = savings_model if lam is not None and savings_model is not None else router_model
        pool = set(active.model_ids)
        shift: dict[str, float] = {}
        if not use_graded:
            with lock:
                row = active.route_text([ctx["human_text"]]).score_frame().iloc[0]
                shift = offsets() if active is router_model else {}
        scores, skipped, raw_quality = [], {}, {}
        requested_price = price_for(settings, requested)
        for model, spec in settings["models"].items():
            price = price_for(settings, model)
            proxy = router_column(model, spec, pool)
            reason = unavailable_reason(model, body, settings, ctx.get("blocked") or {}, ctx["est_prompt_tokens"])
            if reason is None:
                if use_graded and model not in own["quality"]:
                    reason = "no graded data for this model"
                elif not use_graded and proxy not in pool:
                    reason = f"proxy {proxy!r} not in the router's pool"
                elif not has_price(price):
                    reason = "no price"
                elif has_price(ceiling) and (price["input"] > ceiling["input"] or price["output"] > ceiling["output"]):
                    reason = "pricier than the requested model"
                elif (ctx["est_prompt_tokens"] >= long_tokens and model not in (requested, ctx.get("last_model"))
                      and has_price(requested_price) and price["input"] < requested_price["input"]):
                    reason = "conversation too long to move to a cheaper model"
            if reason is not None:
                quality = math.nan
            elif use_graded:
                quality = own["quality"][model]
            else:
                quality = float(row.get(proxy, math.nan)) + shift.get(model, 0.0)
            if reason is None and not math.isfinite(quality):
                reason = "no prediction"
            if reason:
                skipped[model] = reason
                continue
            previous = (ctx.get("prev_quality") or {}).get(model)
            if ctx.get("turn_smoothing", True) and previous is not None and math.isfinite(previous):
                quality = smoothing * quality + (1.0 - smoothing) * float(previous)
            raw_quality[model] = quality
            profile = LLMProfile(name=model, provider=spec["lane"], model_id=model,
                                 input_cost_per_token=price["input"] * 1e-6,
                                 output_cost_per_token=price["output"] * 1e-6,
                                 context_window=spec.get("context_window") or 0)
            scores.append(ModelScore(model=profile, score=quality, expected_quality=quality,
                                     expected_cost=estimate(model, spec, price, ctx)))
        if not scores:
            return {"model": ctx["requested"], "reason": "no eligible candidates", "skipped": skipped}
        direct = not use_graded and all(router_column(s.model.model_id, settings["models"][s.model.model_id], pool)
                                        == s.model.model_id for s in scores)
        if lam is not None:
            # lambda on RELATIVE cost: u = q - lam * cost / cost of the dearest eligible
            # model for this request, so one lam means the same on a short and a
            # 200k-token request. Folded into the objective's $-slope, so utility()
            # stays the one definition; margin 0 makes the policy an argmax of it.
            c_ref = max(s.expected_cost for s in scores)
            objective = OptimizationObjective(cost_weight=float(lam) / c_ref if c_ref > 0 else 0.0)
            constraints = RoutingConstraints(objective=objective, quality_margin=0.0)
        else:
            constraints = RoutingConstraints(quality_margin=ctx.get("quality_margin"))
        request = RoutingRequest(request_id="gateway", prompt=ctx["human_text"], constraints=constraints)
        context = RoutingContext(request=request, signals=PromptSignals(),
                                 candidates=[s.model for s in scores])
        decision = policy.decide(context, scores)
        chosen = next(s for s in scores if s.model.model_id == decision.selected_model_id)
        stay = next((s for s in scores if s.model.model_id == requested), None)
        reason = decision.reason if lam is None else f"max quality - {float(lam):g} x relative cost"
        current = next((s for s in scores if s.model.model_id == ctx.get("last_model")), None)
        if current is not None and chosen is not current \
                and chosen.model.output_cost_per_token > current.model.output_cost_per_token:
            # an upgrade for quality: only worth a cache rewrite if the gap is clear
            margin = (0.0 if lam is not None
                      else policy.margin if ctx.get("quality_margin") is None else ctx["quality_margin"])
            if current.expected_quality >= max(raw_quality.values()) - margin - switch_margin:
                chosen = current
                reason = f"stays on {current.model.model_id}: gap under {margin + switch_margin:g}"
        return {
            "model": chosen.model.model_id,
            "reason": reason,
            "model_quality": {m: round(q, 4) for m, q in raw_quality.items()},  # carried to the next turn
            "predicted_quality": round(raw_quality[chosen.model.model_id], 4),
            "best_predicted_quality": round(max(raw_quality.values()), 4),
            "est_cost": round(chosen.expected_cost, 6),
            "est_cost_requested": round(stay.expected_cost, 6) if stay else None,
            "candidates": len(scores),
            "quality_source": (f"graded ({own['neighbors']} similar of {own['graded_prompts']})" if use_graded
                               else "router scores the models directly" if direct
                               else "benchmark stand-ins, calibrated on your grades" if shift
                               else "benchmark stand-ins"),
            "quality_margin": policy.margin if ctx.get("quality_margin") is None else ctx["quality_margin"],
            "cost_weight": lam,
            "router": getattr(active, "name", None),
            "skipped": skipped,
        }

    pick.calibration_status = calibration_status
    pick.router_kind = getattr(router_model, "kind", None)  # what actually loaded
    pick.router_name = getattr(router_model, "name", None)  # the file's artifact_id when it has one
    pick.savings_name = getattr(savings_model, "name", None)
    pick.savings_error = savings_error  # why the savings router isn't serving, if it was asked for
    return pick
