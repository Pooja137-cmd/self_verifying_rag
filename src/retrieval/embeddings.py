"""Local embeddings via sentence-transformers (CPU). The model loads lazily."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_instruction: str = "",
                 batch_size: int = 32, device: str = "cpu") -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vecs = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(list(texts), show_progress)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.query_instruction + q for q in queries], False)