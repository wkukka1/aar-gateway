"""Reading a router file: a small ``.npz`` the gateway serves with numpy alone (a trained MLP's weights, or a
kNN index), no training data and no PyTorch. Writing one is the training side's job (agentic-llm-router)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MLPArtifactPayload:
    """A trained ``e_q -> R^M`` MLP router: ``sigmoid(W2 relu(W1 x + b1) + b2)``, scored per model.

    ``pathway`` names the embedding space the network was trained on; a file
    saved without one reads as ``"irt"``."""

    state_dict: dict[str, Any]
    model_ids: list[str]
    in_dim: int
    hidden: int
    dropout: float
    pathway: str = "irt"
    query_pathway: Any = None
    query_features: Any = None


@dataclass
class KNNArtifactPayload:
    """A kNN router as plain data: the training queries' embeddings and their
    per-model scores. A prompt's predicted quality per model is the mean score
    of its ``k`` nearest training queries (cosine).

    LLMRouterBench's 8,602 training queries are ~7 MB as an ``.npz``
    (embeddings in float16), and ``encoder`` names a model the ``fastembed``
    backend serves without PyTorch.

    ``scores`` are column-mean imputed (a query a model never saw gets that
    model's mean). ``model_costs`` is each model's mean training cost."""

    embeddings: Any                 # [N, d] unit vectors (float16 on disk)
    scores: Any                     # [N, M] float32
    model_ids: list[str]
    k: int
    encoder: dict[str, Any]         # EncoderConfig fields: backend, model_name, normalize, max_seq_length
    model_costs: Optional[list[float]] = None


    @classmethod
    def load(cls, path) -> tuple["KNNArtifactPayload", dict[str, Any]]:
        """``(payload, header)`` from a router file."""
        import json

        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            header = json.loads(z["header"].tobytes().decode())
            if header.get("format") != "knn":
                raise ValueError(f"{path} is not a kNN router file (format={header.get('format')!r})")
            payload = cls(embeddings=z["embeddings"].astype(np.float32), scores=z["scores"].astype(np.float32),
                          model_ids=list(header["model_ids"]), k=int(header["k"]),
                          encoder=dict(header["encoder"]), model_costs=header.get("model_costs"))
        return payload, header


def load_router_file(path) -> tuple[MLPArtifactPayload | KNNArtifactPayload, dict[str, Any]]:
    """``(payload, header)`` from a router ``.npz`` (``aar-gateway setup``):
    a :class:`KNNArtifactPayload` or an :class:`MLPArtifactPayload` whose
    ``state_dict`` holds numpy arrays, by the header's ``format``."""
    import json

    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(z["header"].tobytes().decode())
        kind = header.get("format")
        if kind == "knn":
            return KNNArtifactPayload.load(path)
        if kind == "mlp":
            state = {"0.weight": z["w1"], "0.bias": z["b1"], "3.weight": z["w2"], "3.bias": z["b2"]}
            payload = MLPArtifactPayload(state_dict=state, model_ids=list(header["model_ids"]),
                                         in_dim=int(header["in_dim"]), hidden=int(header["hidden"]),
                                         dropout=float(header.get("dropout", 0.0)),
                                         pathway=header.get("pathway", "retrieval"))
            return payload, header
    raise ValueError(f"{path} is not a router file this version can load (format={kind!r})")
