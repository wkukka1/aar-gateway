"""Text encoders for the router file's prompts: ``fastembed`` (ONNX, no PyTorch) by default,
``sentence_transformer`` when PyTorch is installed and fastembed is not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_BACKENDS = ("sentence_transformer", "fastembed")


@dataclass
class EncoderConfig:
    name: str = "default"                     # label only
    backend: str = "fastembed"                # {"sentence_transformer", "fastembed"}
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
    max_seq_length: int = 256

    def __post_init__(self):
        if self.backend not in _BACKENDS:
            raise ValueError(f"unsupported embedding backend {self.backend!r}; expected one of {_BACKENDS}")


def light_backend(backend: str = "sentence_transformer") -> str:
    """``fastembed`` in place of ``sentence_transformer`` when fastembed is installed --
    the same vectors without PyTorch. Any other backend is returned as is."""
    if backend != "sentence_transformer":
        return backend
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return backend
    return "fastembed"


class TextEncoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.cfg.backend == "fastembed":
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.cfg.model_name)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.cfg.model_name)
                if self.cfg.max_seq_length:
                    self._model.max_seq_length = int(self.cfg.max_seq_length)
                self._model.eval()
        return self._model

    @property
    def dim(self) -> int:
        return int(self.encode(["dim"]).shape[1])

    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        texts = [("" if t is None else str(t)) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = int(batch_size or self.cfg.batch_size)
        if self.cfg.backend == "fastembed":
            vec = np.asarray(list(self.model.embed(texts, batch_size=bs)), dtype=np.float32)
            if self.cfg.normalize:
                vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        else:
            vec = self.model.encode(texts, batch_size=bs, normalize_embeddings=self.cfg.normalize,
                                    convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(vec, dtype=np.float32)
