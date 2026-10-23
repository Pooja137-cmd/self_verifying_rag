"""Prompt templates. Verification prompts are added in later phases."""
from __future__ import annotations

RAG_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: insufficient information. "
    "Reply with the shortest possible answer (a few words, no explanation, no full sentence)."
)


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
