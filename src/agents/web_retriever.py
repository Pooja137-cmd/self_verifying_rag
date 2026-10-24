"""Web Retrieval Agent: same interface as RetrievalAgent (retrieve / re_retrieve), so the existing
verification pipeline runs unchanged on live web evidence. Builds a temporary in-memory FAISS index."""
from __future__ import annotations

import hashlib

from src.agents.retriever_agent import RetrievalAgent
from src.retrieval.chunking import Document, chunk_corpus
from src.retrieval.vector_store import VectorStore


class WebRetrievalAgent:
    def __init__(self, embedder, client, planner, top_k: int = 5, extra: int = 3, n_pages: int = 4,
                 chunk_size: int = 150, overlap: int = 30) -> None:
        self.embedder, self.client, self.planner = embedder, client, planner
        self.top_k, self.extra, self.n_pages = top_k, extra, n_pages
        self.chunk_size, self.overlap = chunk_size, overlap
        self.docs: list[Document] = []
        self.seen: set[str] = set()
        self.store: VectorStore | None = None
        self.log: list[dict] = []

    def _add_pages(self, query: str) -> int:
        hits = self.client.search(query, self.n_pages * 2)
        added = 0
        for hit in hits:
            if added >= self.n_pages:
                break
            url = hit["url"]
            if url in self.seen:
                continue
            self.seen.add(url)
            page = self.client.fetch(hit)
            if not page:
                continue
            self.docs.append(Document(
                doc_id="web-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10],
                title=page["title"] or hit.get("title", ""), text=page["text"],
                metadata={"url": page["url"], "source": "web"}))
            added += 1
        self.log.append({"query": query, "hits": len(hits), "pages_added": added})
        return added

    def _rebuild(self) -> None:
        chunks = chunk_corpus(self.docs, self.chunk_size, self.overlap)
        self.store = VectorStore.build(chunks, self.embedder, show_progress=False) if chunks else None

    def retrieve(self, question: str) -> list:
        self._add_pages(question)
        self._rebuild()
        return self.store.search(question, k=self.top_k) if self.store else []

    def re_retrieve(self, question: str, prior: list) -> tuple[list, str]:
        if prior:
            self._add_pages(f"{question} {prior[0].chunk.title}")
            self._rebuild()
        if self.store is None:
            return list(prior), question
        inner = RetrievalAgent(self.store, self.planner, top_k=self.top_k, extra=self.extra)
        return inner.re_retrieve(question, prior)
