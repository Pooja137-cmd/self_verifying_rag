import re
import zlib

import numpy as np
import pytest

from src.retrieval.chunking import Document, chunk_corpus
from src.retrieval.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: no model download needed."""
    model_name = "fake-bow"
    query_instruction = ""
    dim = 128

    def _enc(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

    def embed_passages(self, texts, show_progress=False):
        return self._enc(list(texts))

    def embed_queries(self, queries):
        return self._enc(list(queries))


DOCS = [
    Document(doc_id="eiffel", title="Eiffel Tower", text="The Eiffel Tower is a landmark in Paris France."),
    Document(doc_id="photo", title="Photosynthesis", text="Photosynthesis converts light energy into chemical energy."),
    Document(doc_id="wall", title="Great Wall", text="The Great Wall of China is thousands of kilometres long."),
]


def build():
    return VectorStore.build(chunk_corpus(DOCS, 50, 10), FakeEmbedder(), show_progress=False)


def test_search_returns_relevant_first():
    res = build().search("Where is the Eiffel Tower?", k=2)
    assert res[0].chunk.doc_id == "eiffel" and res[0].rank == 1
    assert res[0].score >= res[1].score


def test_k_larger_than_index():
    assert len(build().search("anything", k=100)) == 3


def test_save_load_roundtrip(tmp_path):
    store = build()
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx", FakeEmbedder())
    a = store.search("light energy plants", 3)
    b = loaded.search("light energy plants", 3)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]
    assert loaded.meta["corpus_sha256"] == store.meta["corpus_sha256"]


def test_load_rejects_wrong_embedder(tmp_path):
    build().save(tmp_path / "idx")
    other = FakeEmbedder()
    other.model_name = "different-model"
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path / "idx", other)


def test_build_rejects_empty():
    with pytest.raises(ValueError):
        VectorStore.build([], FakeEmbedder())