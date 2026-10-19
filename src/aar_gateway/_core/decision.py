"""The output of a ranking pass (:class:`ModelScore`) and of a routing decision
(:class:`RoutingDecision`)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from .profile import LLMProfile


@dataclass
class ModelScore:
    model: "LLMProfile"
    score: float
    expected_quality: float = 0.0
    expected_cost: float = 0.0
    expected_latency: float = 0.0
    confidence: float = 1.0
    supported: bool = True


class DecisionDestination(enum.Enum):
    USER = "user"
    PARENT_ORCHESTRATOR = "parent_orchestrator"
    NEW_ORCHESTRATOR = "new_orchestrator"


@dataclass
class RoutingDecision:
    request_id: str
    ranked_models: list[ModelScore] = field(default_factory=list)
    #: The model actually chosen. Usually ``ranked_models[0]``, but not always:
    #: a cost-aware policy deliberately takes a cheaper model that ranks below
    #: the top, so the choice cannot be read off the ranking. Left ``None`` by
    #: a policy that does not distinguish the two; use :meth:`selected`.
    selected_model_id: Optional[str] = None
    destination: DecisionDestination = DecisionDestination.USER
    reason: str = ""
    confidence: float = 1.0
    artifact_id: Optional[str] = None
    policy_version: str = ""

    def selected(self) -> Optional[ModelScore]:
        """The chosen :class:`ModelScore`: ``selected_model_id`` when a policy
        set one, otherwise the top of the ranking."""
        if self.selected_model_id is not None:
            for score in self.ranked_models:
                if score.model.model_id == self.selected_model_id:
                    return score
        return self.ranked_models[0] if self.ranked_models else None
