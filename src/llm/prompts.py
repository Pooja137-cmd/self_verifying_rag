"""Prompt templates and output parsing."""
from __future__ import annotations

import json
import re

ABSTAIN = "insufficient information"

# Legacy short-answer prompt (used by the first baseline run only).
RAG_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: insufficient information. "
    "Reply with the shortest possible answer (a few words, no explanation, no full sentence)."
)

GEN_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    'Respond ONLY with JSON in exactly this form: {"answer": "<shortest possible answer, a few words>", '
    '"statement": "<one complete sentence (max 30 words) that states the answer to the question, '
    'using only facts from the context>"}. '
    'If the context does not contain the answer, respond exactly with '
    '{"answer": "insufficient information", "statement": ""}.'
)


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def is_abstain(answer: str) -> bool:
    return ABSTAIN in (answer or "").lower()


def parse_generation(text: str) -> tuple[str, str, bool]:
    """Return (answer, statement, parsed_ok). Falls back to the first raw line if JSON is unusable."""
    t = (text or "").strip()
    candidates = [t]
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            ans = str(obj.get("answer", "")).strip()
            stm = str(obj.get("statement", "")).strip()
            if ans:
                return ans, stm, True
    first = t.splitlines()[0].strip() if t else ""
    return first, "", False
