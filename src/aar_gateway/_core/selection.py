"""The decision rule: which model to pick from a row of predicted qualities.

``margin > 0`` picks the cheapest model whose utility is within ``margin`` of
the best; ``margin == 0`` is argmax (ties go to the lowest column).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cost_matrix(model_costs, shape: tuple) -> np.ndarray:
    """``model_costs`` as a ``[Q, M]`` array: exactly ``(M,)`` (per model) or
    ``(Q, M)`` (per cell); anything else raises rather than broadcasting."""
    c = np.asarray(model_costs, np.float64)
    q, m = shape
    if c.ndim == 1 and c.shape == (m,):
        return np.broadcast_to(c[None, :], (q, m))
    if c.ndim == 2 and c.shape == (q, m):
        return c
    raise ValueError(f"model_costs has shape {c.shape}; expected ({m},) per-model or ({q}, {m}) per-cell "
                     f"for a {q}x{m} score matrix")


def cost_aware_utility(pred: np.ndarray, *, lam: float = 0.0, model_costs: Optional[np.ndarray] = None,
                       eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """``[Q, M]`` utility ``pred - lam * C``, non-finite cells and ineligible
    candidates masked to ``-inf``."""
    util = np.asarray(pred, np.float64).copy()
    C = cost_matrix(model_costs, util.shape) if model_costs is not None else None
    util[~np.isfinite(util)] = -np.inf   # a missing prediction must never be picked
    if lam and C is not None:
        util = util - lam * C
        util[~np.isfinite(util)] = -np.inf   # a NaN / inf cost yields NaN, which argmax would pick
    if eligible is not None:
        util = np.where(np.asarray(eligible, bool), util, -np.inf)
    return util


def no_selectable_rows(util: np.ndarray) -> np.ndarray:
    """``[Q]`` bool: rows with no finite utility, which :func:`routing_decision`
    sends to column 0."""
    return ~np.isfinite(np.asarray(util, np.float64)).any(axis=1)


def routing_decision(pred: np.ndarray, *, lam: float = 0.0, model_costs: Optional[np.ndarray] = None,
                     eligible: Optional[np.ndarray] = None, margin: float = 0.0) -> np.ndarray:
    """Per-query model index for ``argmax_m [pred - lam * C(m)]``, or with
    ``margin > 0`` the cheapest model whose utility is within ``margin`` of the
    best (cost ties fall back to the higher utility, then the lower column).
    A positive ``margin`` needs ``model_costs``."""
    util = cost_aware_utility(pred, lam=lam, model_costs=model_costs, eligible=eligible)
    rows_all_bad = no_selectable_rows(util)
    if rows_all_bad.any():   # nothing eligible -> column 0, flagged by the caller
        util[rows_all_bad, 0] = -1e18
    if not margin:
        return util.argmax(axis=1)
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if model_costs is None:
        raise ValueError("routing_decision: margin > 0 needs model_costs -- the rule picks the cheapest "
                         "candidate within the margin, and without prices there is no cheapest.")
    costs = cost_matrix(model_costs, util.shape)
    costs = np.where(np.isfinite(costs), costs, np.inf)   # an unpriced model is the dearest, never the cheapest
    within = util >= util.max(axis=1, keepdims=True) - margin
    effective = np.where(within, costs, np.inf)
    columns = np.broadcast_to(np.arange(util.shape[1]), util.shape)
    return np.lexsort((columns, -util, effective), axis=1)[:, 0]
