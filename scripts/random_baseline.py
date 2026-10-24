"""Abstention-matched random baseline. For each condition X, among questions A answered:
if X withheld `w` answers at random, how many of A's wrong answers would it catch by chance?
Hypergeometric one-sided p-value, Bonferroni-adjusted over the conditions compared. Pilot-size caveats apply."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from scipy.stats import hypergeom  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold  # noqa: E402
from src.utils import read_jsonl  # noqa: E402
import os  # noqa: E402

PREFIX = os.environ.get("RESULT_PREFIX", "cond_")
REPORT_DIR = "report" if PREFIX == "cond_" else "report_" + PREFIX.strip("_")


def load(c):
    p = RESULTS_DIR / f"{PREFIX}{c}.jsonl"
    return {r["id"]: r for r in read_jsonl(p)} if p.exists() else None


def wrong(r):
    return (not r["abstained"]) and contains_gold(r["final_answer"], r["gold_answer"]) != 1.0


A = load("A")
rows = []
for c in "BCDEF":
    X = load(c)
    if X is None:
        print(f"condition {c}: NOT YET MEASURED")
        continue
    ids = [i for i, r in A.items() if not r["abstained"] and i in X]
    N = len(ids)
    K = sum(wrong(A[i]) for i in ids)
    w = sum(X[i]["abstained"] for i in ids)
    caught = sum(wrong(A[i]) and X[i]["abstained"] for i in ids)
    lost_ok = w - caught
    p = float(hypergeom.sf(caught - 1, N, K, w)) if w > 0 else 1.0
    rows.append({"condition": c, "A_answered": N, "A_wrong": K, "withheld": w, "wrong_caught": caught,
                 "correct_lost": lost_ok, "expected_caught_random": round(w * K / N, 2),
                 "wrong_after(any)": sum(wrong(X[i]) for i in ids),
                 "expected_wrong_after_random": round(K * (1 - w / N), 2), "p_one_sided": round(p, 4)})
df = pd.DataFrame(rows)
df["p_bonferroni"] = (df["p_one_sided"] * len(df)).clip(upper=1.0).round(4)
print(df.to_string(index=False))
out = RESULTS_DIR / REPORT_DIR
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "random_baseline.csv", index=False)
print("\np small and caught >> expected  => better than random withholding. "
      "Otherwise the reliability gain is explained by abstention alone.")
