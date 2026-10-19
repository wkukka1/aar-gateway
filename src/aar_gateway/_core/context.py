"""The request / context pair the policy decides over."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .constraints import RoutingConstraints
from .signals import PromptSignals

if TYPE_CHECKING:  # pragma: no cover
    from .profile import LLMProfile


@dataclass
class RoutingRequest:
    request_id: str
    prompt: str
    constraints: RoutingConstraints = field(default_factory=RoutingConstraints)


@dataclass
class RoutingContext:
    request: RoutingRequest
    signals: PromptSignals
    candidates: list["LLMProfile"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
