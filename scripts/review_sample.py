"""Write a CSV of condition A's wrong answers for MANUAL error labelling (open in Excel)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, normalize_answer  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

rows = [r for r in read_jsonl(RESULTS_DIR / "cond_A.jsonl")
        if not r["abstained"] and contains_gold(r["final_answer"], r["gold_answer"]) != 1.0]
out = RESULTS_DIR / "report" / "manual_review_A.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "question", "gold", "predicted", "statement", "all_gold_paragraphs_retrieved",
                "gold_string_in_context", "context_1", "context_2", "context_3", "my_label", "notes"])
    for r in rows:
        ctx = [c[:400] for c in r["contexts"]] + ["", "", ""]
        in_ctx = normalize_answer(r["gold_answer"]) in normalize_answer(" ".join(r["contexts"]))
        w.writerow([r["id"], r["question"], r["gold_answer"], r["final_answer"], r.get("statement", ""),
                    r["final_all_hit"], in_ctx, ctx[0], ctx[1], ctx[2], "", ""])
print(f"{len(rows)} wrong answers written to {out}")
print("Fill my_label with ONE of: RETRIEVAL_MISS | DISTRACTOR_MISLED | REASONING_ERROR | "
      "FORMAT_MISMATCH(actually correct) | GOLD_AMBIGUOUS | OTHER")
