"""Shared typed state and condition definitions for the LangGraph workflow."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    question: str
    results: list            # SearchResult objects (current evidence)
    round: int               # number of re-retrievals performed so far
    evidence: dict
    risk_score: float
    risk_level: str          # LOW / MEDIUM / HIGH / NA
    raw_output: str
    answer: str
    statement: str
    parse_ok: bool
    claims: list
    claim_fallback: bool
    claim_results: list
    claim_risk: float
    claim_verified: bool
    critic_action: str
    final_answer: str
    abstained: bool
    flagged_unverified: bool
    history: list


@dataclass
class PipelineConfig:
    name: str = "A"
    evidence_verification: bool = False
    claim_verification: bool = False
    adaptive: bool = False          # risk-gated claim verification (skip when risk is LOW)
    always_on: bool = False         # re-retrieve unconditionally and verify everything
    max_rounds: int = 1             # max re-retrievals
    final_policy: str = "abstain"   # "abstain" or "accept_flagged" when claims stay unsupported
    use_contradiction: bool = False
    gen_max_tokens: int = 96

    def to_dict(self) -> dict:
        return asdict(self)


_CONDITIONS = {
    "A": {},                                                                   # vanilla RAG
    "B": dict(evidence_verification=True),                                     # + evidence verification
    "C": dict(claim_verification=True),                                        # + claim verification
    "D": dict(evidence_verification=True, claim_verification=True),            # evidence + claim
    "E": dict(evidence_verification=True, claim_verification=True, adaptive=True),   # risk-adaptive
    "F": dict(evidence_verification=True, claim_verification=True, always_on=True),  # always-on
}


def make_condition(name: str, **overrides) -> PipelineConfig:
    if name not in _CONDITIONS:
        raise ValueError(f"Unknown condition {name!r}; choose from {sorted(_CONDITIONS)}")
    return PipelineConfig(name=name, **{**_CONDITIONS[name], **overrides})
