"""Structured outputs shared by the verification agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


class ChunkAssessment(BaseModel):
    chunk_id: str
    doc_id: str
    retrieval_score: float
    relevance: float          # retrieval score rescaled to [0, 1] (heuristic)
    contradiction: float      # max P(contradiction) against any other retrieved chunk


class EvidenceAssessment(BaseModel):
    chunks: list[ChunkAssessment]
    top1_score: float
    score_gap: float          # top1 - top2
    mean_score: float
    coverage: float           # share of question content words found in the evidence
    max_contradiction: float
    retrieval_risk: float     # each *_risk is in [0, 1]; higher = riskier
    coverage_risk: float
    contradiction_risk: float
    n_nli_pairs: int = 0

    def signals(self) -> dict[str, float]:
        return {"retrieval_risk": self.retrieval_risk, "coverage_risk": self.coverage_risk,
                "contradiction_risk": self.contradiction_risk}


class ClaimResult(BaseModel):
    claim: str
    verdict: Verdict
    entailment: float
    contradiction: float
    confidence: float
    uncertain: bool           # near the decision threshold: candidate for an LLM second opinion
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
