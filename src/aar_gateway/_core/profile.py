"""``LLMProfile``: static config for one model the router can pick."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMProfile:
    """``input_cost_per_token`` / ``output_cost_per_token`` default to ``None``
    (*unknown*), not ``0.0``: a genuinely free model sets ``0.0`` explicitly."""

    name: str
    provider: str
    model_id: str
    version: str = ""
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    context_window: int = 0
    capabilities: list[str] = field(default_factory=list)
    serving_provider: str | None = None
