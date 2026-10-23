"""Vanilla RAG: query -> embed -> FAISS -> top-k context -> LLM -> answer. Returns a full log record."""
from __future__ import annotations

import time

from src.llm.prompts import RAG_SYSTEM, build_rag_prompt


def clean_answer(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("answer:"):
        t = t[len("answer:"):].strip()
    return t.splitlines()[0].strip() if t else ""


def run_vanilla_rag(question: str, store, llm, top_k: int = 5, max_tokens: int = 48) -> dict:
    t0 = time.perf_counter()
    before = llm.usage.snapshot()

    results = store.search(question, k=top_k)
    retrieval_s = time.perf_counter() - t0

    contexts = [f"{r.chunk.title}: {r.chunk.text}" if r.chunk.title else r.chunk.text for r in results]
    resp = llm.generate(build_rag_prompt(question, contexts), system=RAG_SYSTEM, max_tokens=max_tokens)

    return {
        "question": question,
        "retrieved": [{"rank": r.rank, "score": r.score, "chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id}
                      for r in results],
        "contexts": contexts,
        "raw_output": resp.text,
        "answer": clean_answer(resp.text),
        "model": resp.model,
        "retrieval_s": retrieval_s,
        "latency_s": time.perf_counter() - t0,
        **llm.usage.delta(before),
    }