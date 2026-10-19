"""Writes router files for tests. The package only reads them; real ones are produced by the
training side (agentic-llm-router: ``scripts/routing/export_router.py``)."""

from __future__ import annotations

import json

import numpy as np


def _header(d: dict):
    return np.frombuffer(json.dumps(d).encode(), dtype=np.uint8)


def save_knn(payload, path, meta=None):
    header = {"format": "knn", "model_ids": payload.model_ids, "k": payload.k, "encoder": payload.encoder,
              "model_costs": payload.model_costs, **(meta or {})}
    np.savez_compressed(path, embeddings=np.asarray(payload.embeddings, dtype=np.float16),
                        scores=np.asarray(payload.scores, dtype=np.float32), header=_header(header))


def save_mlp(payload, path, encoder, model_costs=None, meta=None):
    sd = payload.state_dict
    header = {"format": "mlp", "model_ids": payload.model_ids, "in_dim": payload.in_dim, "hidden": payload.hidden,
              "dropout": payload.dropout, "pathway": payload.pathway, "encoder": encoder,
              "model_costs": model_costs, **(meta or {})}
    np.savez_compressed(path, w1=np.asarray(sd["0.weight"], np.float32), b1=np.asarray(sd["0.bias"], np.float32),
                        w2=np.asarray(sd["3.weight"], np.float32), b2=np.asarray(sd["3.bias"], np.float32),
                        header=_header(header))
