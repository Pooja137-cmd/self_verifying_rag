"""Claim Verification Agent (NLI-based; no LLM call).
premise_mode:
  joint   - all evidence concatenated (truncated) + each chunk alone (original v1 behaviour)
  pairs   - each chunk alone + every pair of the top chunks (v2)
  windows - (v3) sentence windows ranked by lexical overlap with the claim, plus cross-chunk window pairs;
            contradiction is counted only from premises that overlap the claim lexically, so spurious NLI
            'contradiction' on unrelated text is ignored.
Thresholds are configurable placeholders; tune on dev data only."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from src.agents.evidence_verifier import content_words
from src.agents.schemas import ClaimResult
from src.retrieval.sentences import split_sentences_safe


@dataclass
class ClaimVerifierConfig:
    support_threshold: float = 0.5
    contradict_threshold: float = 0.5
    uncertain_band: float = 0.15
    joint_max_words: int = 320
    premise_mode: str = "joint"
    pair_top: int = 8
    window: int = 2
    top_windows: int = 6
    per_chunk_windows: int = 2
    min_overlap: float = 0.34


def _overlap(claim_cw: set, text: str) -> float:
    if not claim_cw:
        return 0.0
    return len(claim_cw & content_words(text)) / len(claim_cw)


class ClaimVerifier:
    def __init__(self, nli, cfg: ClaimVerifierConfig | None = None) -> None:
        self.nli = nli
        self.cfg = cfg or ClaimVerifierConfig()

    def _verdict(self, ent: float, con: float) -> tuple[str, float]:
        cfg = self.cfg
        if ent >= cfg.support_threshold:
            return "SUPPORTED", ent
        if con >= cfg.contradict_threshold and con > ent:
            return "CONTRADICTED", con
        return "INSUFFICIENT_EVIDENCE", 1.0 - max(ent, con)

    # ---- v1 / v2 premises (claim-independent)
    def _premises(self, texts: list[str]) -> tuple[list[str], list[tuple[int, ...]]]:
        cfg = self.cfg
        premises: list[str] = []
        used: list[tuple[int, ...]] = []
        if cfg.premise_mode == "pairs":
            for i, t in enumerate(texts):
                premises.append(t)
                used.append((i,))
            top = min(cfg.pair_top, len(texts))
            for i, j in combinations(range(top), 2):
                premises.append(" ".join((texts[i] + " " + texts[j]).split()[: cfg.joint_max_words]))
                used.append((i, j))
        elif cfg.premise_mode == "joint":
            joint_words = " ".join(texts).split()[: cfg.joint_max_words]
            n_fit = max(1, min(len(texts), len(joint_words) // 80 + 1))
            premises.append(" ".join(joint_words))
            used.append(tuple(range(n_fit)))
            for i, t in enumerate(texts):
                premises.append(t)
                used.append((i,))
        else:
            raise ValueError(f"unknown premise_mode {cfg.premise_mode!r}")
        return premises, used

    # ---- v3 windows
    def _windows(self, texts: list[str], titles: list[str] | None) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for ci, text in enumerate(texts):
            title = titles[ci] if titles and ci < len(titles) and titles[ci] else ""
            body = text[len(title) + 2:] if title and text.startswith(title + ": ") else text
            sents = split_sentences_safe(body) or ([body.strip()] if body.strip() else [])
            for i in range(len(sents)):
                for w in range(1, self.cfg.window + 1):
                    if i + w > len(sents):
                        break
                    seg = " ".join(sents[i:i + w])
                    out.append((f"{title}. {seg}" if title else seg, ci))
        return out

    def _verify_windows(self, claim: str, texts: list[str], ids: list[str], titles: list[str] | None) -> ClaimResult:
        cfg = self.cfg
        ccw = content_words(claim)
        wins = self._windows(texts, titles)
        if not wins:
            return ClaimResult(claim=claim, verdict="INSUFFICIENT_EVIDENCE", entailment=0.0, contradiction=0.0,
                               confidence=1.0, uncertain=False, explanation="no usable evidence text")
        ranked = sorted(wins, key=lambda w: (-_overlap(ccw, w[0]), len(w[0])))
        chosen: list[tuple[str, int]] = []
        per: dict[int, int] = {}
        for prem, ci in ranked:
            if per.get(ci, 0) >= cfg.per_chunk_windows:
                continue
            chosen.append((prem, ci))
            per[ci] = per.get(ci, 0) + 1
            if len(chosen) >= cfg.top_windows:
                break
        premises = [p for p, _ in chosen]
        used = [(ci,) for _, ci in chosen]
        for (pa, ca), (pb, cb) in combinations(chosen, 2):
            if ca == cb:
                continue
            premises.append(pa + " " + pb)
            used.append(tuple(sorted({ca, cb})))
        out = self.nli.score(premises, [claim] * len(premises))
        ent_i = max(range(len(out)), key=lambda i: out[i]["entailment"])
        ent = out[ent_i]["entailment"]
        relevant = [i for i, p in enumerate(premises) if _overlap(ccw, p) >= cfg.min_overlap]
        con = max((out[i]["contradiction"] for i in relevant), default=0.0)
        verdict, conf = self._verdict(ent, con)
        support = [ids[k] for k in used[ent_i]]
        return ClaimResult(
            claim=claim, verdict=verdict, entailment=ent, contradiction=con, confidence=float(conf),
            uncertain=abs(ent - cfg.support_threshold) <= cfg.uncertain_band,
            supporting_chunk_ids=support if verdict == "SUPPORTED" else [],
            explanation=f"windows: best entailment={ent:.2f}, relevant-premise contradiction={con:.2f} "
                        f"({len(relevant)}/{len(premises)} premises on-topic)")

    def verify(self, claims: list[str], evidence_texts: list[str], evidence_ids: list[str],
               titles: list[str] | None = None) -> list[ClaimResult]:
        if not claims:
            return []
        if not evidence_texts:
            return [ClaimResult(claim=c, verdict="INSUFFICIENT_EVIDENCE", entailment=0.0, contradiction=0.0,
                                confidence=1.0, uncertain=False, explanation="no evidence available")
                    for c in claims]
        cfg = self.cfg
        if cfg.premise_mode == "windows":
            return [self._verify_windows(c, evidence_texts, evidence_ids, titles) for c in claims]
        premises, used = self._premises(evidence_texts)
        results: list[ClaimResult] = []
        for claim in claims:
            out = self.nli.score(premises, [claim] * len(premises))
            ent_i = max(range(len(out)), key=lambda i: out[i]["entailment"])
            ent = out[ent_i]["entailment"]
            con = max(o["contradiction"] for o in out)
            verdict, conf = self._verdict(ent, con)
            support = [evidence_ids[k] for k in used[ent_i]]
            results.append(ClaimResult(
                claim=claim, verdict=verdict, entailment=ent, contradiction=con, confidence=float(conf),
                uncertain=abs(ent - cfg.support_threshold) <= cfg.uncertain_band,
                supporting_chunk_ids=support if verdict == "SUPPORTED" else [],
                explanation=f"NLI best entailment={ent:.2f}, max contradiction={con:.2f}"))
        return results
