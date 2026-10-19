"""Per-request routing constraints and the utility function scored against them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .decision import ModelScore


@dataclass
class OptimizationObjective:
    """Weights the four terms a routing utility trades off. ``utility()`` is
    the one place ``u(m, i, lambda)`` is defined -- offline signal analysis
    (``evaluation``) must import this rather than reimplementing it, or
    offline results silently stop matching what the router actually optimizes."""

    quality_weight: float = 1.0
    cost_weight: float = 0.0
    latency_weight: float = 0.0
    risk_weight: float = 0.0

    def utility(self, score: "ModelScore") -> float:
        # A zero-weight term is switched off: skip it rather than multiply, since
        # ``0 * nan`` and ``0 * inf`` are ``nan`` and would poison the whole sum.
        terms = (
            (self.quality_weight, score.expected_quality),
            (-self.cost_weight, score.expected_cost),
            (-self.latency_weight, score.expected_latency),
            (-self.risk_weight, 1.0 - score.confidence),
        )
        return sum((weight * value for weight, value in terms if weight), 0.0)


@dataclass
class RoutingConstraints:
    """Hard and soft limits a candidate must clear before it's scored.

    ``required_capabilities`` is checked before scoring
    (``RouterModel.filter_candidates``). ``max_cost`` / ``max_latency`` /
    ``min_quality`` are hard filters applied to the *scored* candidates by
    ``Router.route`` (``min_quality`` needs the prediction), compared
    against ``ModelScore.expected_cost`` / ``expected_latency`` /
    ``expected_quality``. ``objective`` is the soft utility tradeoff applied to
    whatever candidates survive.

    ``quality_margin`` is the caller's **quality budget**: how much predicted
    quality this request will give up to reach a cheaper model. It is the
    relative twin of ``min_quality``, and the more useful of the two, because
    it adapts per prompt. ``min_quality`` is an absolute floor, so on an easy
    prompt -- where every model clears it -- it still lets the request pay for
    the dearest one. ``quality_margin`` spends the budget only where the models
    actually differ.

    Prefer it to raising ``objective.cost_weight`` when what you mean is "spend
    less". ``cost_weight`` is a slope in quality-per-dollar: its usable range
    depends on the pool's price scale, and a small mis-set either does nothing
    or collapses every request onto the cheapest model. ``quality_margin`` is
    bounded, in quality units, and states something a caller can say out loud
    -- "I accept an answer up to 0.05 worse if it is cheaper".

    ``None`` means the caller did not specify and the policy's own default
    applies. Note what it does *not* promise: the same margin buys different
    savings on different pools (0.1 cut cost 68% on RouterBench, 81% on
    SPROUT). What travels between pools is the quality floor, which is why
    this is a quality budget and not a savings target. To go the other way --
    a savings target to a margin -- calibrate against measured data with
    ``evaluation.prompt_decomposition.policy_eval.margin_for_target_saving``.
    """

    max_cost: float | None = None
    max_latency: float | None = None
    min_quality: float | None = None
    required_capabilities: list[str] = field(default_factory=list)
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
    quality_margin: float | None = None
