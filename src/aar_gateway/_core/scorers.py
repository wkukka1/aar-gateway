"""Routers served from a router file: text in, per-model predicted quality out.

Both load a file :func:`~.artifacts.load_router_file` reads and need numpy and
an encoder (``fastembed`` when installed) -- no training data, no PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import KNNArtifactPayload, MLPArtifactPayload, load_router_file
from .encoder import EncoderConfig, TextEncoder, light_backend
from .selection import routing_decision


class ScoreResult:
    """What :meth:`TextRouter.route_text` returns: the ``[text x model]`` predicted-quality matrix."""

    def __init__(self, scores: pd.DataFrame, selected: np.ndarray):
        self.scores = scores
        self.selected = selected

    def score_frame(self) -> pd.DataFrame:
        return self.scores

    @property
    def selected_model_ids(self) -> list[str]:
        return [str(self.scores.columns[i]) for i in self.selected]


def _artifact_encoder(spec: dict, encoders_dir=None) -> dict:
    """The encoder a file names, served without PyTorch when it can be: a
    sentence-transformers model runs on ``fastembed`` (same vectors) if installed.
    A local ONNX encoder's relative ``model_path`` is a folder under ``encoders_dir``."""
    spec = {k: v for k, v in spec.items()
            if k in ("backend", "model_name", "normalize", "max_seq_length", "model_path", "pooling", "dim")}
    spec["backend"] = light_backend(spec.get("backend", "sentence_transformer"))
    if spec.get("model_path") and encoders_dir is not None and not Path(spec["model_path"]).is_absolute():
        spec["model_path"] = str(Path(encoders_dir) / spec["model_path"])
    return spec


class TextRouter:
    """The contract the gateway's model picker uses: a pool of model ids and a
    predicted quality per model for a prompt."""

    kind = "router"

    def __init__(self, model_ids: Sequence[str], *, name: Optional[str] = None, model_costs=None,
                 encoder=None, encoder_spec: Optional[dict] = None):
        ids = [str(m) for m in model_ids]
        if not ids:
            raise ValueError("a router needs a non-empty candidate pool")
        self._model_ids = ids
        self.name = name or self.kind
        self._costs = np.asarray(model_costs, dtype=float) if model_costs is not None else None
        self._encoder = encoder
        self._encoder_spec = dict(encoder_spec or {})
        self._encoders_dir = None   # where a local ONNX encoder's folder lives (load_router sets it)

    @property
    def model_ids(self) -> list[str]:
        return list(self._model_ids)

    @property
    def can_route_text(self) -> bool:
        return True

    @property
    def default_model_costs(self) -> Optional[np.ndarray]:
        """Each model's mean training cost, as the file carries it."""
        return self._costs

    def _encode(self, texts: Sequence[str], encoder=None) -> np.ndarray:
        encoder = encoder or self._encoder
        if encoder is None:
            encoder = self._encoder = TextEncoder(EncoderConfig(
                name=self.name, **_artifact_encoder(self._encoder_spec, self._encoders_dir)))
        return np.asarray(encoder.encode(list(texts)), dtype=np.float32)

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        raise NotImplementedError

    def route_text(self, texts: Sequence[str], *, encoder=None) -> ScoreResult:
        scores = self.predict_scores_text(list(texts), encoder=encoder)
        scores.index = [str(i) for i in range(len(scores))]
        return ScoreResult(scores, routing_decision(scores.to_numpy(np.float64)))

    def warmup(self) -> None:
        self.route_text(["warmup"])


class KNNRouter(TextRouter):
    """Predicted quality = the mean score of the ``k`` training prompts nearest the
    prompt (cosine)."""

    kind = "knn"

    @classmethod
    def from_payload(cls, payload: KNNArtifactPayload, *, encoder=None, name: Optional[str] = None) -> "KNNRouter":
        if not isinstance(payload, KNNArtifactPayload):
            raise TypeError(f"KNNRouter needs a KNNArtifactPayload, got {type(payload).__name__}")
        router = cls(payload.model_ids, name=name or f"knn(k={payload.k})", model_costs=payload.model_costs,
                     encoder=encoder, encoder_spec=payload.encoder)
        E = np.asarray(payload.embeddings, dtype=np.float32)
        router._k = int(payload.k)
        router._E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
        router._C = np.asarray(payload.scores, dtype=np.float32)
        return router

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        q = self._encode(texts, encoder)
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        k = min(self._k, len(self._E))
        nearest = np.argpartition(-(q @ self._E.T), k - 1, axis=1)[:, :k]
        return pd.DataFrame(self._C[nearest].mean(axis=1), index=list(range(len(q))), columns=self._model_ids)


class MLPRouter(TextRouter):
    """``sigmoid(W2 relu(W1 x + b1) + b2)`` over the prompt embedding."""

    kind = "mlp"

    @classmethod
    def from_payload(cls, payload: MLPArtifactPayload, header: dict, *, encoder=None,
                     name: Optional[str] = None) -> "MLPRouter":
        sd = payload.state_dict
        router = cls(payload.model_ids, name=name or "mlp", model_costs=header.get("model_costs"),
                     encoder=encoder, encoder_spec=header.get("encoder") or {})
        router._w = tuple(np.asarray(sd[k], dtype=np.float32) for k in ("0.weight", "0.bias", "3.weight", "3.bias"))
        return router

    def _forward(self, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._w
        h = np.maximum(x @ w1.T + b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ w2.T + b2)))

    def predict_scores_text(self, texts: Sequence[str], *, encoder=None) -> pd.DataFrame:
        x = self._encode(texts, encoder)
        return pd.DataFrame(self._forward(x), index=list(range(len(x))), columns=self._model_ids)


def load_router(path, *, encoder=None, encoders_dir=None) -> TextRouter:
    """The router a file holds (``aar-gateway setup`` installs one). ``encoders_dir``:
    where a file's local ONNX encoder folder (``encoder.model_path``) lives."""
    payload, header = load_router_file(Path(path).expanduser())
    if header["format"] == "mlp":
        router = MLPRouter.from_payload(payload, header, encoder=encoder,
                                        name=header.get("artifact_id") or f"mlp:{Path(path).stem}")
    else:
        router = KNNRouter.from_payload(payload, encoder=encoder)
    router._encoders_dir = encoders_dir
    return router
