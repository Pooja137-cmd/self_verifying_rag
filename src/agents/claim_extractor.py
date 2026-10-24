"""Claim Extraction Agent. Default is rule-based sentence splitting (zero LLM calls), now abbreviation-aware.
Optional atomic decomposition via the LLM (one call); falls back to sentences."""
from __future__ import annotations

import json
import re

from src.retrieval.chunking import clean_text
from src.retrieval.sentences import split_sentences_safe

ABSTAIN_MARK = "insufficient information"

CLAIM_PROMPT = (
    "Split the answer into atomic factual claims. Each claim must be a short, self-contained statement "
    "that can be checked on its own. Do not add new facts. Respond ONLY with JSON: "
    '{"claims": ["...", "..."]}\n\nAnswer: <<ANSWER>>'
)


def extract_claims_rule(answer: str, min_words: int = 3) -> list[str]:
    text = clean_text(answer)
    text = re.sub(r"^answer:\s*", "", text, flags=re.I)
    if not text or ABSTAIN_MARK in text.lower():
        return []
    return [s for s in split_sentences_safe(text) if len(s.split()) >= min_words and not s.endswith("?")]


class ClaimExtractor:
    def __init__(self, llm=None, atomic: bool = False, max_claims: int = 6, max_tokens: int = 160) -> None:
        self.llm = llm
        self.atomic = atomic and llm is not None
        self.max_claims = max_claims
        self.max_tokens = max_tokens

    def extract(self, answer: str) -> list[str]:
        base = extract_claims_rule(answer)
        if not base or not self.atomic:
            return base[: self.max_claims]
        try:
            resp = self.llm.generate(CLAIM_PROMPT.replace("<<ANSWER>>", answer), json_mode=True,
                                     max_tokens=self.max_tokens)
            claims = [c.strip() for c in json.loads(resp.text)["claims"] if isinstance(c, str) and c.strip()]
            return (claims or base)[: self.max_claims]
        except Exception:
            return base[: self.max_claims]
