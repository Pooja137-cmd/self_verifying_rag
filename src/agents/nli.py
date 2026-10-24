"""Local NLI scorer (CPU) on a sentence-transformers CrossEncoder. No LLM, no cost."""
from __future__ import annotations

import os

import numpy as np

DEFAULT_NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
_FALLBACK_ORDER = ["contradiction", "entailment", "neutral"]  # documented order for cross-encoder NLI models


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class NLIScorer:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: str = "cpu",
                 batch_size: int = 16, max_length: int = 512) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_pairs = 0
        self._model = None
        self._idx: dict[str, int] | None = None

    def _label_index(self) -> dict[str, int]:
        m = self._model
        cfg = getattr(m, "config", None) or getattr(getattr(m, "model", None), "config", None)
        id2label = getattr(cfg, "id2label", None) or {}
        names = [str(id2label[i]).lower() for i in sorted(id2label)] if id2label else []
        if sorted(names) != ["contradiction", "entailment", "neutral"]:
            names = list(_FALLBACK_ORDER)
        return {n: names.index(n) for n in ("entailment", "neutral", "contradiction")}

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            self._idx = self._label_index()
        return self._model

    def score(self, premises: list[str], hypotheses: list[str]) -> list[dict[str, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        if not premises:
            return []
        model = self._load()
        out = np.asarray(model.predict(list(zip(premises, hypotheses)), batch_size=self.batch_size,
                                       show_progress_bar=False), dtype=np.float64)
        if out.ndim == 1:
            out = out[None, :]
        if not (np.all(out >= 0) and np.allclose(out.sum(axis=1), 1.0, atol=1e-3)):
            out = _softmax(out)  # raw logits -> probabilities
        self.n_pairs += len(premises)
        return [{k: float(row[i]) for k, i in (self._idx or {}).items()} for row in out]
