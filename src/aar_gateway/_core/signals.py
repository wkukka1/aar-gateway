"""A named observation about a prompt, and the bag of them a request carries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Signal:
    name: str
    value: Any
    confidence: float = 1.0
    produced_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSignals:
    """Every :class:`Signal` a decomposer
    produced for one request, plus any raw embeddings computed along the way."""

    signals: dict[str, Signal] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def add(self, signal: Signal) -> None:
        if signal.name in self.signals:
            raise ValueError(f"Signal {signal.name} already exists")
        self.signals[signal.name] = signal

    def add_embedding(self, name: str, vector) -> None:
        self.embeddings[name] = vector

    def get(self, name: str) -> Optional[Signal]:
        return self.signals.get(name)

    def get_embedding(self, name: str):
        return self.embeddings.get(name)
