"""Shared typed state and condition definitions for the LangGraph workflow."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    question: str
    results: list
    round: int               # re-retrievals performed
    revisions: int           # revise steps performed
    evidence: dict
    risk_score: float
    risk_level: str          # LOW / MEDIUM / HIGH / NA
    raw_output: str
    answer: str
    statement: str
    parse_ok: bool
    claims: list
    claim_fallback: bool
    answer_in_claims: bool
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
    adaptive: bool = False
    always_on: bool = False
    max_rounds: int = 1
    revise: bool = False
    max_revisions: int = 1
    final_policy: str = "abstain"   # abstain | accept_flagged | abstain_if_contradicted
    use_contradiction: bool = False
    gen_max_tokens: int = 96

    def to_dict(self) -> dict:
        return asdict(self)


_CONDITIONS = {
    "A": {},
    "B": dict(evidence_verification=True),
    "C": dict(claim_verification=True),
    "D": dict(evidence_verification=True, claim_verification=True),
    "E": dict(evidence_verification=True, claim_verification=True, adaptive=True),
    "F": dict(evidence_verification=True, claim_verification=True, always_on=True),
}


def make_condition(name: str, **overrides) -> PipelineConfig:
    if name not in _CONDITIONS:
        raise ValueError(f"Unknown condition {name!r}; choose from {sorted(_CONDITIONS)}")
    return PipelineConfig(name=name, **{**_CONDITIONS[name], **overrides})
