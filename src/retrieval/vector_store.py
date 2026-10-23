"""FAISS exact inner-product index over L2-normalised vectors (= cosine similarity)."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.chunking import Chunk
from src.utils import read_jsonl, sha256_text, write_jsonl


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "score": self.score, "chunk_id": self.chunk.chunk_id,
                "doc_id": self.chunk.doc_id, "title": self.chunk.title, "text": self.chunk.text}


class VectorStore:
    def __init__(self, embedder, index: faiss.Index, chunks: list[Chunk], meta: dict[str, Any]) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.meta = meta

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, show_progress: bool = True,
              meta_extra: dict[str, Any] | None = None) -> "VectorStore":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        vecs = embedder.embed_passages([c.embed_text for c in chunks], show_progress=show_progress)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        meta = {
            "embedding_model": embedder.model_name,
            "query_instruction": getattr(embedder, "query_instruction", ""),
            "dim": int(vecs.shape[1]),
            "n_chunks": len(chunks),
            "n_docs": len({c.doc_id for c in chunks}),
            "corpus_sha256": sha256_text("\n".join(f"{c.chunk_id}\t{c.text}" for c in chunks)),
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(meta_extra or {}),
        }
        return cls(embedder, index, chunks, meta)

    def search_many(self, queries: list[str], k: int = 5) -> list[list[SearchResult]]:
        k = min(k, self.index.ntotal)
        if k <= 0 or not queries:
            return [[] for _ in queries]
        q = np.ascontiguousarray(self.embedder.embed_queries(queries), dtype=np.float32)
        scores, ids = self.index.search(q, k)
        out: list[list[SearchResult]] = []
        for row_scores, row_ids in zip(scores, ids):
            out.append([SearchResult(self.chunks[int(i)], float(s), r + 1)
                        for r, (s, i) in enumerate(zip(row_scores, row_ids)) if i >= 0])
        return out

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.search_many([query], k)[0]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        write_jsonl(d / "chunks.jsonl", (c.model_dump() for c in self.chunks))
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder) -> "VectorStore":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["embedding_model"] != embedder.model_name:
            raise ValueError(f"Index was built with {meta['embedding_model']!r} but the embedder is "
                             f"{embedder.model_name!r}. Rebuild the index or use the same model.")
        index = faiss.read_index(str(d / "index.faiss"))
        chunks = [Chunk(**row) for row in read_jsonl(d / "chunks.jsonl")]
        if index.ntotal != len(chunks):
            raise ValueError("Index and chunks.jsonl are out of sync")
        return cls(embedder, index, chunks, meta)