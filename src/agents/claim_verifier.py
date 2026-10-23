"""Claim Verification Agent (NLI-based; no LLM call). Each claim is checked against (a) all evidence
concatenated (needed for multi-hop claims) and (b) each chunk alone; best entailment / contradiction win.
Thresholds are configurable placeholders to be tuned on dev data. 'uncertain' flags near-threshold cases
for a later LLM second opinion."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.schemas import ClaimResult


@dataclass
class ClaimVerifierConfig:
    support_threshold: float = 0.5
    contradict_threshold: float = 0.5
    uncertain_band: float = 0.15
    joint_max_words: int = 320   # keep the joint premise inside the NLI model's 512-token window


class ClaimVerifier:
    def __init__(self, nli, cfg: ClaimVerifierConfig | None = None) -> None:
        self.nli = nli
        self.cfg = cfg or ClaimVerifierConfig()

    def verify(self, claims: list[str], evidence_texts: list[str], evidence_ids: list[str]) -> list[ClaimResult]:
        if not claims:
            return []
        if not evidence_texts:
            return [ClaimResult(claim=c, verdict="INSUFFICIENT_EVIDENCE", entailment=0.0, contradiction=0.0,
                                confidence=1.0, uncertain=False, explanation="no evidence available")
                    for c in claims]
        cfg = self.cfg
        joint_words = " ".join(evidence_texts).split()[: cfg.joint_max_words]
        premises = [" ".join(joint_words)] + list(evidence_texts)
        used = [None] + list(range(len(evidence_texts)))   # None = joint premise

        results: list[ClaimResult] = []
        for claim in claims:
            out = self.nli.score(premises, [claim] * len(premises))
            ent_i = max(range(len(out)), key=lambda i: out[i]["entailment"])
            ent = out[ent_i]["entailment"]
            con = max(o["contradiction"] for o in out)
            if ent >= cfg.support_threshold:
                verdict, conf = "SUPPORTED", ent
            elif con >= cfg.contradict_threshold and con > ent:
                verdict, conf = "CONTRADICTED", con
            else:
                verdict, conf = "INSUFFICIENT_EVIDENCE", 1.0 - max(ent, con)
            if used[ent_i] is None:   # joint premise = the chunks that fit the window
                n_fit = max(1, min(len(evidence_ids), len(joint_words) // 80 + 1))
                support = list(evidence_ids[:n_fit])
            else:
                support = [evidence_ids[used[ent_i]]]
            results.append(ClaimResult(
                claim=claim, verdict=verdict, entailment=ent, contradiction=con, confidence=float(conf),
                uncertain=abs(ent - cfg.support_threshold) <= cfg.uncertain_band,
                supporting_chunk_ids=support if verdict == "SUPPORTED" else [],
                explanation=f"NLI best entailment={ent:.2f}, max contradiction={con:.2f}"))
        return results
