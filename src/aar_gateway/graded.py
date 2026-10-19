"""The gateway's own routing data: the user's Claude Code turns, graded on the
real Claude tiers by ``aar-gateway-grade`` (``aar_gateway.grading.grade``).

For a new prompt, a tier's predicted quality is the similarity-weighted mean
of its grades over the most similar graded prompts. When too few graded
prompts are similar enough, there is no prediction and the router falls back
to the benchmark router with its stand-ins (``proxy``). The file is re-read
when it changes, so grades added while the gateway runs are picked up.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .config import read_jsonl

Encoder = Callable[[Sequence[str]], np.ndarray]  # unit-normalized rows
GradedPredictor = Callable[[str], dict | None]


#: path -> ((mtime_ns, size), result): the file is parsed once per change, however many readers ask
_loaded: dict[str, tuple[tuple, tuple]] = {}


def _usable(row: object) -> bool:
    """A graded row with every field the router reads. One malformed row must not make every
    routing decision fail, so it is skipped."""
    if not isinstance(row, dict) or row.get("score") is None:
        return False
    try:
        float(row["score"])
    except (TypeError, ValueError):
        return False
    return all(isinstance(row.get(k), str) for k in ("prompt_hash", "prompt", "model"))


def load_graded(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """``(prompts, models, scores)``: one row per graded prompt, ``nan`` where
    a model has no grade for it (failed replay, unparseable verdict). Cached until the
    file changes: the picker's calibration and the predictor both read it."""
    try:
        st = os.stat(path)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    cached = _loaded.get(str(path))
    if stamp is not None and cached and cached[0] == stamp:
        return cached[1]
    rows = [r for r in read_jsonl(path) if _usable(r)]
    prompts: dict[str, str] = {}
    for r in rows:
        prompts.setdefault(r["prompt_hash"], r["prompt"])
    hashes, models = list(prompts), sorted({r["model"] for r in rows})
    scores = np.full((len(hashes), len(models)), np.nan)
    at, col = {h: i for i, h in enumerate(hashes)}, {m: j for j, m in enumerate(models)}
    for r in rows:
        scores[at[r["prompt_hash"]], col[r["model"]]] = float(r["score"])
    result = ([prompts[h] for h in hashes], models, scores)
    if stamp is not None:
        _loaded[str(path)] = (stamp, result)
    return result


def graded_predictor(path: Path, encode: Encoder, *, k: int = 10, min_similarity: float = 0.5,
                     min_neighbors: int = 3, min_prompts: int = 20) -> GradedPredictor:
    state: dict = {"mtime": None, "prompts": [], "models": [], "scores": None, "vectors": None}
    lock = threading.Lock()

    def refresh() -> None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime == state["mtime"]:
            return
        prompts, models, scores = load_graded(path)
        vectors = encode(prompts) if prompts else None
        state.update(mtime=mtime, prompts=prompts, models=models, scores=scores, vectors=vectors)

    def predict(text: str) -> dict | None:
        with lock:
            refresh()
            if len(state["prompts"]) < min_prompts:
                return None
            sims = state["vectors"] @ encode([text])[0]
            top = np.argsort(-sims)[:k]
            top = top[sims[top] >= min_similarity]
            if len(top) < min_neighbors:
                return None
            weights = sims[top]
            quality = {}
            for j, model in enumerate(state["models"]):
                col = state["scores"][top, j]
                ok = ~np.isnan(col)
                if ok.sum() >= min_neighbors:
                    quality[model] = float(np.average(col[ok], weights=weights[ok]))
            return {"quality": quality, "neighbors": int(len(top)),
                    "max_similarity": round(float(sims[top].max()), 3),
                    "graded_prompts": len(state["prompts"])}

    return predict
