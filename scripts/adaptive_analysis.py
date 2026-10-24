"""Adaptive vs always-on verification, relative to vanilla A, on the same questions. No LLM calls.
python scripts/adaptive_analysis.py [prefix]   (default prefix: cond_)
Correctness = gold answer contained in the final answer (lenient). Abstention is not counted as wrong."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

prefix = sys.argv[1] if len(sys.argv) > 1 else "cond_"


def load(c):
    p = RESULTS_DIR / f"{prefix}{c}.jsonl"
    return {r["id"]: r for r in read_jsonl(p)} if p.exists() else None


def ok(r):
    return (not r["abstained"]) and contains_gold(r["final_answer"], r["gold_answer"]) == 1.0


def wrong(r):
    return (not r["abstained"]) and contains_gold(r["final_answer"], r["gold_answer"]) != 1.0


data = {c: load(c) for c in "ABCDEF"}
if data["A"] is None:
    sys.exit(f"{prefix}A.jsonl not found")
miss = [c for c, d in data.items() if d is None]
if miss:
    print("NOT YET MEASURED:", "".join(miss))
ids = sorted(set.intersection(*[set(d) for d in data.values() if d is not None]))
A = data["A"]
nA_wrong = sum(wrong(A[i]) for i in ids)
nA_ok = sum(ok(A[i]) for i in ids)
print(f"prefix '{prefix}'  common questions: {len(ids)}   A answered-wrong: {nA_wrong}   A correct: {nA_ok}\n")

rows = []
for c in "BCDEF":
    X = data[c]
    if X is None:
        continue
    R = [X[i] for i in ids]
    prev_abs = sum(wrong(A[i]) and X[i]["abstained"] for i in ids)
    prev_fix = sum(wrong(A[i]) and ok(X[i]) for i in ids)
    prev_other = sum(wrong(A[i]) and wrong(X[i]) and A[i]["final_answer"] != X[i]["final_answer"] for i in ids)
    lost_abs = sum(ok(A[i]) and X[i]["abstained"] for i in ids)
    lost_wrong = sum(ok(A[i]) and wrong(X[i]) for i in ids)
    new_wrong = sum((not wrong(A[i])) and wrong(X[i]) and not ok(A[i]) for i in ids)
    checks = sum(r["claim_verified"] for r in R)
    noop = sum(r["claim_verified"] and r.get("critic_action") is None for r in R)
    on_ok = sum(r["claim_verified"] and ok(A[r["id"]]) for r in R)
    rr_unneeded = sum(r["re_retrieved"] and r["initial_all_hit"] for r in R)
    rr = sum(r["re_retrieved"] for r in R)
    prevented = prev_abs + prev_fix
    rows.append({
        "cond": c, "wrong_prevented": prevented, "of_which_abstained": prev_abs, "of_which_corrected": prev_fix,
        "correct_lost": lost_abs + lost_wrong, "correct_abstained": lost_abs, "correct_turned_wrong": lost_wrong,
        "net(prevented-lost)": prevented - lost_abs - lost_wrong,
        "claim_checks": checks, "checks_no_effect(all supported 1st pass)": noop, "checks_on_A_correct": on_ok,
        "checks_per_wrong_prevented": round(checks / prevented, 2) if prevented else None,
        "re_retrievals": rr, "re_retrievals_unneeded(gold already complete)": rr_unneeded,
        "extra_llm_calls/q": round(float(np.mean([r["llm_calls"] for r in R]) - np.mean([A[i]["llm_calls"] for i in ids])), 3),
        "extra_tokens/q": round(float(np.mean([r["total_tokens"] for r in R]) - np.mean([A[i]["total_tokens"] for i in ids])), 1),
        "nli_pairs/q": round(float(np.mean([r["nli_pairs"] for r in R])), 1),
        "extra_standalone_s/q": round(float(np.mean([r["latency_s"] for r in R]) - np.mean([A[i]["latency_s"] for i in ids])), 1)})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.T.to_string(header=False) if len(df) else "no conditions")
(RESULTS_DIR / "report").mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DIR / "report" / f"adaptive_{prefix.strip('_')}.csv", index=False)

if data["D"] is not None and data["E"] is not None:
    D, E = data["D"], data["E"]
    print("\nAdaptive (E) vs always-verify (D): where their final decisions differ")
    for lab, a, b in (("E answered, D abstained", E, D), ("D answered, E abstained", D, E)):
        sel = [i for i in ids if (not a[i]["abstained"]) and b[i]["abstained"]]
        print(f"  {lab}: {len(sel)}   (the answering side was correct in {sum(ok(a[i]) for i in sel)})")
    print(f"  claim verification performed: D {sum(D[i]['claim_verified'] for i in ids)}   "
          f"E {sum(E[i]['claim_verified'] for i in ids)}   (E saves {sum(D[i]['claim_verified'] for i in ids) - sum(E[i]['claim_verified'] for i in ids)} checks)")
print("\nLow counts: with this n every difference of 1-3 questions is within noise; report counts, not rates.")
