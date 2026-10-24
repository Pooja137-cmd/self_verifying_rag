"""Compare claim-verifier v1 (joint premise) vs v3 (sentence windows) on DEV answers. No LLM calls.
Sanity check: claims that are near-verbatim copies of evidence should be accepted (~100%).
The threshold is chosen on dev only; if v3's AUC CI includes 0.5 the default 0.5 is kept (no tuning on noise)."""
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
from src.config import INDEX_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, normalize_answer  # noqa: E402
from src.graph.workflow import claims_from_generation  # noqa: E402
from src.utils import read_jsonl, write_json  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "dev_A"
rows = [r for r in read_jsonl(RESULTS_DIR / f"{name}.jsonl") if not r["abstained"]]
titles_by_chunk = {c["chunk_id"]: c.get("title", "")
                   for c in read_jsonl(INDEX_DIR / "hotpot_dev200" / "chunks.jsonl")}
nli, ext = NLIScorer(), ClaimExtractor()
v1 = ClaimVerifier(nli, ClaimVerifierConfig(premise_mode="joint"))
v3 = ClaimVerifier(nli, ClaimVerifierConfig(premise_mode="windows"))

recs, copies = [], []
for r in rows:
    claims, _ = claims_from_generation(ext, r["question"], r["answer"], r.get("statement") or "")
    titles = [titles_by_chunk.get(cid, "") for cid in r["retrieved_chunk_ids"]]
    a = v1.verify(claims, r["contexts"], r["retrieved_chunk_ids"])
    b = v3.verify(claims, r["contexts"], r["retrieved_chunk_ids"], titles=titles)
    ctx_norm = normalize_answer(" ".join(r["contexts"]))
    for c, x, y in zip(claims, a, b):
        copies.append({"copy": normalize_answer(c) in ctx_norm, "ent1": x.entailment, "ent3": y.entailment})
    ans = normalize_answer(r["answer"])
    recs.append({"id": r["id"], "correct": int(contains_gold(r["answer"], r["gold_answer"]) == 1.0),
                 "in_ctx": int(bool(ans) and ans in ctx_norm),
                 "ent1": min(x.entailment for x in a), "con1": max(x.contradiction for x in a),
                 "ent3": min(x.entailment for x in b), "con3": max(x.contradiction for x in b),
                 "contradicted3": int(any(x.verdict == "CONTRADICTED" for x in b))})
df, cp = pd.DataFrame(recs), pd.DataFrame(copies)
(RESULTS_DIR / "report").mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DIR / "report" / "dev_verifier_v3_features.csv", index=False)
y = df["correct"].values
n_c, n_w = int(y.sum()), int(len(y) - y.sum())
base_err = n_w / len(y)
print(f"dev answered={len(y)}  correct={n_c}  wrong={n_w}  base error rate={base_err:.2f}")

c = cp[cp["copy"]]
if len(c):
    print(f"\nSANITY - {len(c)} claims are near-verbatim copies of evidence. Accepted (entailment>=0.5): "
          f"v1 {(c.ent1 >= 0.5).mean():.2f}   v3 {(c.ent3 >= 0.5).mean():.2f}   (ideal ~1.00)")
if n_c < 5 or n_w < 5:
    print("Too few correct/wrong answers for AUC; extend the dev run. Nothing written.")
    sys.exit(0)


def auc_ci(lab, score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(lab), len(lab))
        if lab[idx].min() != lab[idx].max():
            vals.append(roc_auc_score(lab[idx], score[idx]))
    return np.percentile(vals, [2.5, 97.5])


print("\nAUC (0.5 = no signal), bootstrap 95% CI:")
auc3 = ci3 = None
for label, lab, sc in (("v1 min-entailment (predicts correct)", y, df.ent1.values),
                       ("v3 min-entailment (predicts correct)", y, df.ent3.values),
                       ("v1 max-contradiction (predicts wrong)", 1 - y, df.con1.values),
                       ("v3 gated contradiction (predicts wrong)", 1 - y, df.con3.values)):
    lo, hi = auc_ci(lab, sc)
    a = roc_auc_score(lab, sc)
    print(f"  {label:<42} {a:.3f}  [{lo:.2f}, {hi:.2f}]")
    if label.startswith("v3 min"):
        auc3, ci3 = a, (lo, hi)

rules = {"answer string NOT in evidence": df.in_ctx.values == 0, "v3 any claim CONTRADICTED": df.contradicted3.values == 1}
for t in (0.1, 0.3, 0.5, 0.7):
    rules[f"v1 min-ent < {t}"] = df.ent1.values < t
for t in (0.1, 0.3, 0.5, 0.7):
    rules[f"v3 min-ent < {t}"] = df.ent3.values < t
print(f"\nRejection rules (random rejection: precision ~{base_err:.2f}, lift ~1.0):")
print(f"{'rule':<34} {'rejected':>8} {'wrong':>7} {'correct':>8} {'precision':>10} {'lift':>6}")
for label, rej in rules.items():
    k = int(rej.sum())
    caught, lost = int((rej & (y == 0)).sum()), int((rej & (y == 1)).sum())
    prec = caught / k if k else float("nan")
    print(f"{label:<34} {k:>8} {caught:>3}/{n_w:<3} {lost:>4}/{n_c:<3} {prec:>10.2f} {prec / base_err:>6.2f}")

best_t, best_j = 0.5, -9.0
for t in np.arange(0.3, 0.91, 0.1):
    j = float((df.ent3.values[y == 1] >= t).mean() - (df.ent3.values[y == 0] >= t).mean())
    if j > best_j:
        best_j, best_t = j, float(t)
informative = ci3[0] > 0.5
chosen = round(best_t, 2) if informative else 0.5
print(f"\nv3 AUC CI {'excludes' if informative else 'INCLUDES'} 0.5 -> support_threshold = {chosen} "
      f"({'max Youden J=%.2f' % best_j if informative else 'default kept; not tuned on noise'})")
write_json(RESULTS_DIR / "claim_threshold_v3.json",
           {"support_threshold": chosen, "premise_mode": "windows", "dev_file": name, "n": len(y),
            "auc_v3": float(auc3), "auc_ci": [float(ci3[0]), float(ci3[1])], "tuned": bool(informative)})
print(f"saved {RESULTS_DIR / 'claim_threshold_v3.json'}")
