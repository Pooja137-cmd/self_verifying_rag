"""Compare claim-verification variants on a DEV results file (never the test set). No LLM calls.
correct = answered and gold contained in prediction; wrong = answered otherwise.
Features use the first-round evidence stored in the dev file (re-retrieval is not simulated)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.agents.claim_extractor import ClaimExtractor  # noqa: E402
from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig  # noqa: E402
from src.agents.nli import NLIScorer  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, normalize_answer  # noqa: E402
from src.graph.workflow import claims_from_generation  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "dev_A"
rows = [r for r in read_jsonl(RESULTS_DIR / f"{name}.jsonl") if not r["abstained"]]
nli, ext = NLIScorer(), ClaimExtractor()
ver = {"v1": ClaimVerifier(nli, ClaimVerifierConfig(premise_mode="joint")),
       "v2": ClaimVerifier(nli, ClaimVerifierConfig(premise_mode="pairs"))}

recs = []
for r in rows:
    claims, _ = claims_from_generation(ext, r["question"], r["answer"], r.get("statement") or "")
    ans = normalize_answer(r["answer"])
    rec = {"id": r["id"], "correct": int(contains_gold(r["answer"], r["gold_answer"]) == 1.0),
           "in_ctx": int(bool(ans) and ans in normalize_answer(" ".join(r["contexts"])))}
    for k, v in ver.items():
        res = v.verify(claims, r["contexts"], r["retrieved_chunk_ids"])
        rec[f"ent_{k}"] = min(c.entailment for c in res)
        rec[f"con_{k}"] = max(c.contradiction for c in res)
    recs.append(rec)
df = pd.DataFrame(recs)
y = df["correct"].values
n_c, n_w = int(y.sum()), int(len(y) - y.sum())
base_err = n_w / len(y)
print(f"dev answered={len(y)}  correct={n_c}  wrong={n_w}  base error rate among answered={base_err:.2f}")
(RESULTS_DIR / "report").mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DIR / "report" / "dev_verifier_features.csv", index=False)
if n_c < 5 or n_w < 5:
    print("Too few correct or wrong answers for a meaningful comparison; extend the dev run.")
    sys.exit(0)


def auc_ci(labels, scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    return np.percentile(vals, [2.5, 97.5])


print("\nAUC for separating correct from wrong answers (0.5 = no signal), bootstrap 95% CI:")
for feat, lab, score in (("answer string in evidence", y, df["in_ctx"].values),
                         ("v1 min-entailment", y, df["ent_v1"].values),
                         ("v2 min-entailment (pairs)", y, df["ent_v2"].values),
                         ("v1 max-contradiction (predicts wrong)", 1 - y, df["con_v1"].values),
                         ("v2 max-contradiction (predicts wrong)", 1 - y, df["con_v2"].values)):
    lo, hi = auc_ci(lab, score)
    print(f"  {feat:<40} {roc_auc_score(lab, score):.3f}  [{lo:.2f}, {hi:.2f}]")

inctx = df["in_ctx"].values == 1
ent1, ent2 = df["ent_v1"].values, df["ent_v2"].values
rules = {"answer string NOT in evidence": ~inctx}
for t in (0.1, 0.3, 0.5, 0.7):
    rules[f"v1 entailment < {t}"] = ent1 < t
for t in (0.1, 0.3, 0.5, 0.7):
    rules[f"v2 entailment < {t}"] = ent2 < t
rules["v1 contradiction >= 0.8"] = df["con_v1"].values >= 0.8
rules["v2 contradiction >= 0.8"] = df["con_v2"].values >= 0.8
rules["v2 ent<0.5 AND answer not in evidence"] = (ent2 < 0.5) & ~inctx

print(f"\nRejection rules (random rejection would have precision ~{base_err:.2f}, lift ~1.0):")
print(f"{'rule':<40} {'rejected':>8} {'wrong':>6} {'correct':>8} {'precision':>10} {'lift':>6}")
for label, rej in rules.items():
    n_rej = int(rej.sum())
    caught = int((rej & (y == 0)).sum())
    lost = int((rej & (y == 1)).sum())
    prec = caught / n_rej if n_rej else float("nan")
    print(f"{label:<40} {n_rej:>8} {caught:>3}/{n_w:<2} {lost:>5}/{n_c:<2} {prec:>10.2f} {prec / base_err:>6.2f}")
print("\nlift > 1 means better than random rejection. At this dev size, treat differences of 1-2 answers as noise.")
print(f"saved {RESULTS_DIR / 'report' / 'dev_verifier_features.csv'}")
