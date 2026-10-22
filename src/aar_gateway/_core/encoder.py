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
    #: a LOCAL ONNX encoder (fastembed only): the folder holding ``model.onnx`` and
    #: its tokenizer files, registered as a fastembed custom model -- how a router
    #: file trained on an encoder fastembed doesn't ship (raw BERT) is served
    #: without PyTorch. ``pooling`` ({"mean", "cls"}) and ``dim`` describe it.
    model_path: Optional[str] = None
    pooling: Optional[str] = None
    dim: Optional[int] = None

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


def _local_fastembed(cfg: EncoderConfig):
    """A fastembed model served from ``cfg.model_path``: registered once as a custom
    model (pooling over the ONNX ``last_hidden_state``, no normalisation here --
    :meth:`TextEncoder.encode` applies ``cfg.normalize``), truncated at
    ``max_seq_length`` like the encoder it was trained with."""
    from pathlib import Path

    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    path = Path(cfg.model_path).expanduser()
    if not (path / "model.onnx").is_file():
        raise FileNotFoundError(f"encoder {cfg.model_name!r}: no model.onnx in {path} "
                                "(create it with scripts/routing/export_bert_encoder.py)")
    if cfg.model_name not in {m["model"] for m in TextEmbedding.list_supported_models()}:
        TextEmbedding.add_custom_model(
            model=cfg.model_name, pooling=PoolingType.CLS if cfg.pooling == "cls" else PoolingType.MEAN,
            normalization=False, sources=ModelSource(hf=cfg.model_name), dim=int(cfg.dim or 768),
            model_file="model.onnx")
    model = TextEmbedding(cfg.model_name, specific_model_path=str(path))
    if cfg.max_seq_length:
        model.model.tokenizer.enable_truncation(max_length=int(cfg.max_seq_length))
    return model


class TextEncoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.cfg.backend == "fastembed" and self.cfg.model_path:
                self._model = _local_fastembed(self.cfg)
            elif self.cfg.backend == "fastembed":
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
