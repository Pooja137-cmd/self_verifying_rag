"""Per-question diagnosis of a results file: was the abstention justified by retrieval?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import exact_match, is_abstention  # noqa: E402
from src.utils import read_jsonl  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "baseline_rag"
rows = list(read_jsonl(RESULTS_DIR / f"{name}.jsonl"))
tally = {"abstain_gold_missing": 0, "abstain_gold_present": 0, "answered_correct": 0, "answered_wrong": 0}
for r in rows:
    got = {x["doc_id"] for x in r["retrieved"]}
    gold = set(r["gold_doc_ids"])
    all_hit = gold <= got
    in_ctx = r["gold_answer"].lower() in " ".join(r["contexts"]).lower()
    ab = is_abstention(r["answer"])
    ok = exact_match(r["answer"], r["gold_answer"]) == 1.0
    if ab:
        tally["abstain_gold_present" if all_hit else "abstain_gold_missing"] += 1
    else:
        tally["answered_correct" if ok else "answered_wrong"] += 1
    print(f"{'ABSTAIN' if ab else ('OK    ' if ok else 'WRONG ')} gold_docs={len(gold & got)}/{len(gold)} "
          f"gold_answer_in_context={'Y' if in_ctx else 'N'}  pred={r['answer']!r} gold={r['gold_answer']!r}")
print(f"\nn={len(rows)}  {tally}")
print("abstain_gold_present = abstained although ALL gold paragraphs were retrieved (generator/prompt problem)")
print("abstain_gold_missing = abstained when evidence was missing (plausibly justified)")
