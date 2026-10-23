"""Evidence Verification Agent (cheap signals only; no LLM call).

Signals per query: retrieval confidence (score-based), lexical coverage of the question by the
evidence (a crude sufficiency proxy), and cross-chunk contradiction (NLI). All thresholds are
configurable placeholders and must be tuned/validated experimentally, not assumed correct.
An LLM sufficiency check for MEDIUM-risk cases is added with the risk controller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.agents.schemas import ChunkAssessment, EvidenceAssessment

_STOP = set("""the and for are was were what which who whom whose when where why how did does has have had with
that this from into than then there their they them his her its she him you your been being not but can could
would should about after before between during over under also other both each any all out off one two""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOP}


def question_coverage(question: str, texts: list[str]) -> float:
    q = content_words(question)
    if not q:
        return 1.0
    return len(q & content_words(" ".join(texts))) / len(q)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class EvidenceConfig:
    score_lo: float = 0.45   # placeholder: cosine score treated as "no confidence"
    score_hi: float = 0.80   # placeholder: cosine score treated as "full confidence"


class EvidenceVerifier:
    def __init__(self, nli=None, cfg: EvidenceConfig | None = None) -> None:
        self.nli = nli
        self.cfg = cfg or EvidenceConfig()

    def _rescale(self, s: float) -> float:
        c = self.cfg
        return _clip01((s - c.score_lo) / (c.score_hi - c.score_lo))

    def assess(self, question: str, results: list, check_contradiction: bool = True) -> EvidenceAssessment:
        scores = [float(r.score) for r in results]
        top1 = scores[0] if scores else 0.0
        gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
        mean = float(np.mean(scores)) if scores else 0.0
        texts = [r.chunk.text for r in results]
        coverage = question_coverage(question, texts)

        per_chunk_con = [0.0] * len(results)
        n_pairs = 0
        if self.nli is not None and check_contradiction and len(results) > 1:
            idx_pairs = list(combinations(range(len(results)), 2))
            nli_out = self.nli.score([texts[i] for i, _ in idx_pairs], [texts[j] for _, j in idx_pairs])
            n_pairs = len(idx_pairs)
            for (i, j), o in zip(idx_pairs, nli_out):
                per_chunk_con[i] = max(per_chunk_con[i], o["contradiction"])
                per_chunk_con[j] = max(per_chunk_con[j], o["contradiction"])
        max_con = max(per_chunk_con) if per_chunk_con else 0.0

        chunks = [ChunkAssessment(chunk_id=r.chunk.chunk_id, doc_id=r.chunk.doc_id, retrieval_score=float(r.score),
                                  relevance=self._rescale(float(r.score)), contradiction=c)
                  for r, c in zip(results, per_chunk_con)]
        return EvidenceAssessment(
            chunks=chunks, top1_score=top1, score_gap=gap, mean_score=mean, coverage=coverage,
            max_contradiction=max_con, retrieval_risk=1.0 - self._rescale(top1),
            coverage_risk=1.0 - coverage, contradiction_risk=max_con, n_nli_pairs=n_pairs)
