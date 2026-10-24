"""Prompts for the Revise step."""
from __future__ import annotations

REVISE_SYSTEM = (
    "You correct a draft answer using ONLY the provided context. A verifier reported that some claims of the "
    "draft are not supported by the context. Respond ONLY with JSON: "
    '{"answer": "<shortest possible answer, a few words>", "statement": "<one complete sentence (max 30 words) '
    'stating the answer, using only facts from the context>"}. '
    'If the context does not contain the answer, respond exactly with {"answer": "insufficient information", "statement": ""}.'
)


def build_revise_prompt(question: str, contexts: list[str], answer: str, statement: str, feedback: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    fb = "\n".join(feedback) if feedback else "- (no details)"
    return (f"Context:\n{ctx}\n\nQuestion: {question}\n\nDraft answer: {answer}\nDraft statement: {statement}\n\n"
            f"Verifier feedback (claims of the draft that were NOT supported):\n{fb}\n\nCorrected JSON:")
