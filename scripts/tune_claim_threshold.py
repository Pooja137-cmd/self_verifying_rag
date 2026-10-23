"""Choose the claim-support threshold on a DEV results file (never the test set).
Proxy label: the generated answer matches the gold answer (this is answer correctness, not claim-level truth)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.agents.claim_extractor import ClaimExtractor  # noqa: E402
from src.agents.claim_verifier import ClaimVerifier  # noqa: E402
from src.agents.nli import NLIScorer  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold  # noqa: E402
from src.graph.workflow import claims_from_generation  # noqa: E402
from src.utils import read_jsonl, write_json  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "dev_A"
rows = [r for r in read_jsonl(RESULTS_DIR / f"{name}.jsonl") if not r["abstained"]]
cv, ext = ClaimVerifier(NLIScorer()), ClaimExtractor()

ent, correct = [], []
for r in rows:
    claims, _ = claims_from_generation(ext, r["question"], r["answer"], r.get("statement") or "")
    res = cv.verify(claims, r["contexts"], r["retrieved_chunk_ids"])
    ent.append(min(c.entailment for c in res))
    correct.append(int(contains_gold(r["answer"], r["gold_answer"])))
ent, correct = np.array(ent), np.array(correct)
print(f"dev answered={len(rows)}  correct={int(correct.sum())}  wrong={int((1 - correct).sum())}")
if correct.sum() < 3 or (1 - correct).sum() < 3:
    print("Need at least 3 correct and 3 wrong answers on dev; run more dev questions. Nothing written.")
    sys.exit(0)

auc = float(roc_auc_score(correct, ent))
print(f"AUC (min claim entailment vs answer correctness): {auc:.3f}\n")
print(f"{'thr':>5} {'accept|correct':>15} {'accept|wrong':>13}")
best_t, best_j = 0.5, -9.0
for t in np.arange(0.05, 0.96, 0.05):
    tpr = float((ent[correct == 1] >= t).mean())
    fpr = float((ent[correct == 0] >= t).mean())
    print(f"{t:>5.2f} {tpr:>15.2f} {fpr:>13.2f}")
    if tpr - fpr > best_j:
        best_j, best_t = tpr - fpr, float(t)
print(f"\nchosen support_threshold={best_t:.2f} (max Youden J={best_j:.2f}; dev n={len(rows)}, noisy)")
if auc < 0.6:
    print("WARNING: AUC < 0.6: the NLI verifier barely separates correct from wrong answers here.")
write_json(RESULTS_DIR / "claim_threshold.json", {"support_threshold": best_t, "dev_file": name,
                                                   "auc": auc, "youden_j": best_j, "n": len(rows)})
print(f"saved {RESULTS_DIR / 'claim_threshold.json'}")
