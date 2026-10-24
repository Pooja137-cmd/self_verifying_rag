"""Error analysis: exclusive outcome per question + non-exclusive diagnostic flags, with a CSV for manual review."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold  # noqa: E402
from src.utils import read_jsonl  # noqa: E402
import os  # noqa: E402

PREFIX = os.environ.get("RESULT_PREFIX", "cond_")
REPORT_DIR = "report" if PREFIX == "cond_" else "report_" + PREFIX.strip("_")

OUT = RESULTS_DIR / REPORT_DIR


def classify(r: dict) -> dict:
    lenient = contains_gold(r["final_answer"], r["gold_answer"]) == 1.0
    if r["abstained"]:
        outcome = "abstain_gold_present" if r["final_all_hit"] else "abstain_gold_missing"
    else:
        outcome = "correct" if lenient else "wrong_answer"
    verdicts = [c["verdict"] for c in r.get("claim_results", [])]
    return {
        "id": r["id"], "condition": r["condition"], "outcome": outcome,
        "retrieval_gap": not r["final_all_hit"],
        "claim_accepted_wrong": (not r["abstained"]) and not lenient and r["claim_verified"]
                                and bool(verdicts) and all(v == "SUPPORTED" for v in verdicts),
        "claim_rejected_correct": r["abstained"] and r.get("critic_action") == "ABSTAIN"
                                  and contains_gold(r.get("answer") or "", r["gold_answer"]) == 1.0,
        "unnecessary_reretrieval": r["re_retrieved"] and r["initial_all_hit"],
        "reretrieval_fixed_gap": r["re_retrieved"] and (not r["initial_all_hit"]) and r["final_all_hit"],
        "claim_extraction_fallback": bool(r.get("claim_fallback")),
        "generator_json_parse_failed": r.get("parse_ok") is False,
        "risk_level": r.get("risk_level"), "question": r["question"], "gold": r["gold_answer"],
        "final": r["final_answer"], "pre_critic_answer": r.get("answer"), "claim_verdicts": "|".join(verdicts),
        "latency_s": round(r["latency_s"], 1)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for c in "ABCDEF":
        p = RESULTS_DIR / f"{PREFIX}{c}.jsonl"
        if p.exists():
            recs += [classify(r) for r in read_jsonl(p)]
        else:
            print(f"condition {c}: NOT YET MEASURED")
    df = pd.DataFrame(recs)
    if df.empty:
        return
    df.to_csv(OUT / "error_analysis.csv", index=False)

    print("\nOutcome counts per condition:")
    print(pd.crosstab(df["condition"], df["outcome"]).to_string())
    flags = ["retrieval_gap", "claim_accepted_wrong", "claim_rejected_correct", "unnecessary_reretrieval",
             "reretrieval_fixed_gap", "claim_extraction_fallback", "generator_json_parse_failed"]
    print("\nDiagnostic flag counts per condition:")
    print(df.groupby("condition")[flags].sum().to_string())
    if "E" in set(df["condition"]):
        print("\nCondition E: outcomes by risk level (does risk predict trouble?):")
        print(pd.crosstab(df[df.condition == "E"]["risk_level"], df[df.condition == "E"]["outcome"]).to_string())
    print(f"\nsaved {OUT / 'error_analysis.csv'}  (open in Excel; hand-check the 'wrong_answer' and "
          f"'claim_accepted_wrong' rows)")
    print("Not tracked in this pilot (no LLM revision step): incorrect revision, over-verification beyond the flags above.")


if __name__ == "__main__":
    main()
