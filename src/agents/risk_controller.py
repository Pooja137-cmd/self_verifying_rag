"""Risk Controller. Weights and thresholds are CONFIGURABLE PLACEHOLDERS, not scientifically derived;
calibrate with scripts/calibrate_risk.py on questions disjoint from the evaluation set."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class RiskConfig:
    w_retrieval: float = 0.5
    w_evidence: float = 0.5          # weight of coverage_risk (sufficiency proxy)
    w_contradiction: float = 0.0     # signal implemented, off by default (not yet measured)
    w_claim: float = 0.0             # claim risk is logged but does not drive decisions in this pilot
    low_threshold: float = 0.35
    high_threshold: float = 0.60

    @classmethod
    def load(cls, path: str | Path) -> "RiskConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class RiskController:
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()

    def score(self, signals: dict[str, float]) -> float:
        c = self.cfg
        weights = {"retrieval_risk": c.w_retrieval, "coverage_risk": c.w_evidence,
                   "contradiction_risk": c.w_contradiction, "claim_risk": c.w_claim}
        num = den = 0.0
        for name, w in weights.items():
            if w > 0 and name in signals:
                num += w * signals[name]
                den += w
        return num / den if den > 0 else 0.0

    def level(self, score: float) -> str:
        if score >= self.cfg.high_threshold:
            return "HIGH"
        if score < self.cfg.low_threshold:
            return "LOW"
        return "MEDIUM"

    def assess(self, signals: dict[str, float]) -> tuple[float, str]:
        s = self.score(signals)
        return s, self.level(s)
