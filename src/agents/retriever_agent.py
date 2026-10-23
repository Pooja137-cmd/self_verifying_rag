"""Retrieval Agent: initial retrieval and expansion-based re-retrieval (kept separate from verification)."""
from __future__ import annotations

from src.retrieval.vector_store import SearchResult


class RetrievalAgent:
    def __init__(self, store, planner, top_k: int = 5, extra: int = 3) -> None:
        self.store = store
        self.planner = planner
        self.top_k = top_k
        self.extra = extra

    def retrieve(self, question: str) -> list[SearchResult]:
        return self.store.search(self.planner.initial_query(question), k=self.top_k)

    def re_retrieve(self, question: str, prior: list[SearchResult]) -> tuple[list[SearchResult], str]:
        """Original results first, then up to `extra` new chunks from the expanded query."""
        q2 = self.planner.expand_query(question, prior)
        candidates = self.store.search(q2, k=self.top_k * 2)
        seen = {r.chunk.chunk_id for r in prior}
        merged = list(prior)
        added = 0
        for r in candidates:
            if r.chunk.chunk_id in seen:
                continue
            merged.append(r)
            seen.add(r.chunk.chunk_id)
            added += 1
            if added >= self.extra:
                break
        merged = [SearchResult(r.chunk, r.score, i + 1) for i, r in enumerate(merged)]
        return merged, q2
