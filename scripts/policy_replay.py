"""Replay verification policies on a DEV results file (no LLM). Uses first-round evidence only
(re-retrieval is not simulated), so this approximates, not reproduces, the full pipeline.
Correctness proxy: gold answer contained in the predicted answer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.agents.claim_extractor import ClaimExtractor  # noqa: E402
from src.agents.claim_verifier import ClaimVerifier  # noqa: E402
from src.agents.nli import NLIScorer  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, normalize_answer  # noqa: E402
from src.graph.workflow import claims_from_generation  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "dev_A"
rows = [r for r in read_jsonl(RESULTS_DIR / f"{name}.jsonl") if not r["abstained"]]
cv, ext = ClaimVerifier(NLIScorer()), ClaimExtractor()

feat = []
for r in rows:
    claims, _ = claims_from_generation(ext, r["question"], r["answer"], r.get("statement") or "")
    res = cv.verify(claims, r["contexts"], r["retrieved_chunk_ids"])
    ans = normalize_answer(r["answer"])
    feat.append({
        "min_ent": min(c.entailment for c in res), "max_con": max(c.contradiction for c in res),
        "in_ctx": bool(ans) and ans in normalize_answer(" ".join(r["contexts"])),
        "correct": contains_gold(r["answer"], r["gold_answer"]) == 1.0})

n_c = sum(f["correct"] for f in feat)
n_w = len(feat) - n_c
print(f"dev answered={len(feat)}  correct={n_c}  wrong={n_w}")
if n_c < 3 or n_w < 3:
    print("Too few correct or wrong answers for a meaningful comparison; run more dev questions.")
    sys.exit(0)


def evaluate(label, rejects):
    rej = [rejects(f) for f in feat]
    caught = sum(r and not f["correct"] for r, f in zip(rej, feat))
    rej_ok = sum(r and f["correct"] for r, f in zip(rej, feat))
    missed = sum((not r) and (not f["correct"]) for r, f in zip(rej, feat))
    kept_ok = sum((not r) and f["correct"] for r, f in zip(rej, feat))
    prec = caught / sum(rej) if sum(rej) else float("nan")
    print(f"{label:<34} rejected={sum(rej):>3}  caught_wrong={caught:>2}/{n_w}  "
          f"rejected_correct={rej_ok:>2}/{n_c}  missed_wrong={missed:>2}  kept_correct={kept_ok:>2}  "
          f"rejection_precision={prec:.2f}")


print("\nstrict: reject unless entailment >= t")
for t in (0.3, 0.5, 0.7, 0.9):
    evaluate(f"  strict t={t}", lambda f, t=t: f["min_ent"] < t)
print("\ncontradiction-only: reject when contradiction >= t")
for t in (0.5, 0.8, 0.95):
    evaluate(f"  contradiction t={t}", lambda f, t=t: f["max_con"] >= t)
print("\nhybrid: reject if contradicted(0.8), or (entailment<0.5 and answer string absent from evidence)")
evaluate("  hybrid", lambda f: f["max_con"] >= 0.8 or (f["min_ent"] < 0.5 and not f["in_ctx"]))
print("\nA good policy catches many wrong answers while rejecting few correct ones. "
      "With this dev size every difference of 1-2 answers is noise.")
