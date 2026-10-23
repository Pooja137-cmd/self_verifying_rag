"""Build paper-ready tables and the reliability-vs-cost figure from saved condition results.
Only conditions with a results file are included; others are reported as NOT YET MEASURED.
All conditions are compared on the SAME question ids (intersection)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, exact_match  # noqa: E402
from src.evaluation.evaluator import summarize_condition  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

LABELS = {"A": "A Vanilla RAG", "B": "B +Evidence", "C": "C +Claim", "D": "D +Evidence+Claim",
          "E": "E Risk-adaptive", "F": "F Always-on"}
OUT = RESULTS_DIR / "report"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, dict]] = {}
    for c in LABELS:
        p = RESULTS_DIR / f"cond_{c}.jsonl"
        if p.exists():
            data[c] = {r["id"]: r for r in read_jsonl(p)}
        else:
            print(f"condition {c}: NOT YET MEASURED (no results file)")
    if "A" not in data:
        print("Need condition A to build the report.")
        return
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    print(f"conditions: {sorted(data)}   common questions: {len(ids)}")
    if not ids:
        return
    rows = {c: [data[c][i] for i in ids] for c in data}

    # ---- main table
    table, summaries = [], {}
    for c in sorted(rows):
        s = summarize_condition(rows[c])
        summaries[c] = s
        k_wrong = round(s["wrong_answer_rate"] * s["n"])
        lo, hi = wilson(k_wrong, s["n"])
        table.append({
            "condition": LABELS[c], "n": s["n"], "EM": s["EM"], "contains_gold": s["contains_gold"],
            "abstain": s["abstain_rate"], "wrong_rate": s["wrong_answer_rate"],
            "wrong_95CI": f"[{lo:.2f}, {hi:.2f}]", "final_all_hit": s["final_all_hit"],
            "re_retrieval": s["re_retrieval_rate"], "claim_verif": s["claim_verification_rate"],
            "latency_s": s["avg_latency_s"], "llm_calls": s["avg_llm_calls"], "tokens": s["avg_total_tokens"]})
    tdf = pd.DataFrame(table)
    tdf.to_csv(OUT / "table_main.csv", index=False)

    # ---- paired vs A
    paired = []
    A = {r["id"]: r for r in rows["A"]}

    def lenient(r):
        return contains_gold(r["final_answer"], r["gold_answer"]) == 1.0

    def wrong(r):
        return (not r["abstained"]) and not lenient(r)

    for c in sorted(rows):
        if c == "A":
            continue
        X = {r["id"]: r for r in rows[c]}
        paired.append({
            "condition": LABELS[c],
            "newly_EM_correct": sum(exact_match(X[i]["final_answer"], X[i]["gold_answer"]) == 1.0 and
                                    exact_match(A[i]["final_answer"], A[i]["gold_answer"]) != 1.0 for i in ids),
            "lost_EM_correct": sum(exact_match(A[i]["final_answer"], A[i]["gold_answer"]) == 1.0 and
                                   exact_match(X[i]["final_answer"], X[i]["gold_answer"]) != 1.0 for i in ids),
            "wrong->abstained": sum(wrong(A[i]) and X[i]["abstained"] for i in ids),
            "wrong->not_wrong_answered": sum(wrong(A[i]) and not wrong(X[i]) and not X[i]["abstained"] for i in ids),
            "newly_wrong": sum((not wrong(A[i])) and wrong(X[i]) for i in ids),
            "correct->abstained": sum(lenient(A[i]) and X[i]["abstained"] for i in ids)})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(OUT / "table_paired_vs_A.csv", index=False)

    md = ["# Results (pilot; counts on the common question set)\n", md_table(tdf), "\n"]
    if len(pdf):
        md += ["## Paired changes relative to A (question counts)\n", md_table(pdf), "\n"]
    md.append("wrong_rate = answered AND gold answer not contained; abstention is not counted as wrong. "
              "CI = Wilson 95%; with this n, overlapping intervals mean the difference is not established.\n")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # ---- reliability vs cost figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in sorted(summaries):
        s = summaries[c]
        ax[0].scatter(s["avg_latency_s"], s["wrong_answer_rate"], s=70)
        ax[0].annotate(c, (s["avg_latency_s"], s["wrong_answer_rate"]), xytext=(5, 5), textcoords="offset points")
        ax[1].scatter(s["avg_latency_s"], s["EM"], s=70)
        ax[1].annotate(c, (s["avg_latency_s"], s["EM"]), xytext=(5, 5), textcoords="offset points")
    ax[0].set_xlabel("avg standalone latency per question (s)")
    ax[0].set_ylabel("wrong-answer rate (lower is better)")
    ax[0].set_title("Reliability vs cost")
    ax[1].set_xlabel("avg standalone latency per question (s)")
    ax[1].set_ylabel("exact match")
    ax[1].set_title("Accuracy vs cost")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Pilot, n={len(ids)} questions, single local 3B model")
    fig.tight_layout()
    fig.savefig(OUT / "reliability_vs_cost.png", dpi=160)
    print(f"\nsaved tables and figure to {OUT}")


if __name__ == "__main__":
    main()
