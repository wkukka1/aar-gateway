"""``MarginRoutingPolicy``: turns a list of :class:`~.decision.ModelScore` into a
:class:`~.decision.RoutingDecision` -- the cheapest candidate within ``margin`` of the best."""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

import numpy as np

from .decision import DecisionDestination, RoutingDecision
from .selection import routing_decision

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .context import RoutingContext
    from .decision import ModelScore


class RoutingPolicy(abc.ABC):
    name: str = "policy"
    version: str = "0"
    escalation_threshold: float = 0.0

    @abc.abstractmethod
    def decide(self, context: "RoutingContext", scores: list["ModelScore"]) -> RoutingDecision:
        raise NotImplementedError

    @abc.abstractmethod
    def select_destination(self, context: "RoutingContext",
                           scores: list["ModelScore"]) -> DecisionDestination:
        raise NotImplementedError

    def _ranked(self, context: "RoutingContext",
                scores: list["ModelScore"]) -> list["ModelScore"]:
        """``scores`` by the request's objective utility, best first (non-finite last)."""
        objective = context.request.constraints.objective

        def key(score: "ModelScore") -> float:
            u = float(objective.utility(score))
            return u if math.isfinite(u) else -math.inf

        return sorted(scores, key=key, reverse=True)


class MarginRoutingPolicy(RoutingPolicy):
    """Take the cheapest candidate whose utility is within ``margin`` of the best.

    The measured problem with :class:`DefaultRoutingPolicy` is that argmax on a
    ranker whose scores are dominated by global model capability picks the
    strongest model almost every time, which scores about the same as having no
    router at all while costing the same. The ranking is not the issue: the
    model that should have been picked is usually rank 2. The decision rule is.

    So this rule keeps the ranking and changes only the selection: among the
    candidates the ranker rates within ``margin`` of its best, take the cheapest.
    A *relative* margin, not an absolute score threshold -- a predicted 0.7
    means different things on an easy and a hard query, and an absolute bar
    measured worse than no router at all, while the relative one measured
    better .

    What a given margin buys is pool-dependent: measure it on your own traffic.

    ``escalation_threshold`` is honoured here rather than ignored: when even
    the best candidate is rated below it, the request goes to the parent
    orchestrator instead of being answered by a model nobody believes in.
    """

    name = "margin"
    version = "1"

    def __init__(self, margin: float = 0.02, escalation_threshold: float = 0.0) -> None:
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        self.margin = float(margin)
        self.escalation_threshold = float(escalation_threshold)

    def _margin_for(self, context: "RoutingContext") -> float:
        """The request's quality budget, or this policy's default.

        Cost sensitivity is a property of the *request*, not of the deployment:
        the same service answers "summarise this changelog" and "write the
        migration", and they do not deserve the same budget. `None` on the
        constraints means the caller did not say, so the policy's own value
        stands.
        """
        asked = getattr(context.request.constraints, "quality_margin", None)
        if asked is None:
            return self.margin
        asked = float(asked)
        if asked < 0:
            raise ValueError(f"quality_margin must be >= 0, got {asked}")
        return asked

    def decide(self, context: "RoutingContext", scores: list["ModelScore"]) -> RoutingDecision:
        objective = context.request.constraints.objective
        margin = self._margin_for(context)
        ranked = self._ranked(context, scores)
        chosen = None
        if ranked:
            # the served rule (routing_decision), not a copy: cost ties keep the
            # better-ranked model and an unpriced model never wins on price.
            # `tiny` keeps exact utility ties cost-broken at margin=0 (by design,
            # see test_the_margin_zero_tie_break_differs_on_purpose).
            util = np.array([[float(objective.utility(s)) for s in ranked]])
            costs = np.array([s.expected_cost for s in ranked], np.float64)
            m = margin or np.finfo(np.float64).tiny
            chosen = ranked[int(routing_decision(util, model_costs=costs, margin=m)[0])]
        return RoutingDecision(
            request_id=context.request.request_id,
            ranked_models=ranked,
            selected_model_id=chosen.model.model_id if chosen else None,
            destination=self.select_destination(context, scores),
            reason=(f"cheapest within {margin:g} utility of the best"
                    if chosen else "no candidates scored"),
            confidence=chosen.confidence if chosen else 0.0,
            policy_version=self.version,
        )

    def select_destination(self, context: "RoutingContext",
                           scores: list["ModelScore"]) -> DecisionDestination:
        ranked = self._ranked(context, scores)
        if not ranked:
            return DecisionDestination.USER
        objective = context.request.constraints.objective
        if float(objective.utility(ranked[0])) < self.escalation_threshold:
            return DecisionDestination.PARENT_ORCHESTRATOR
        return DecisionDestination.USER
