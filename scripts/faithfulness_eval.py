"""Post-hoc faithfulness with an NLI model INDEPENDENT of the gating verifier.
faithful = every extracted claim of the final answer's statement is entailed by the final retrieved context.
Caveats: automatic proxy; checks the 3B-generated statement (a paraphrase), not the whole answer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.agents.claim_extractor import ClaimExtractor  # noqa: E402
from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig  # noqa: E402
from src.agents.nli import NLIScorer  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.evaluation.answer_metrics import contains_gold  # noqa: E402
from src.graph.workflow import claims_from_generation  # noqa: E402
from src.utils import read_jsonl  # noqa: E402
import os  # noqa: E402

PREFIX = os.environ.get("RESULT_PREFIX", "cond_")
REPORT_DIR = "report" if PREFIX == "cond_" else "report_" + PREFIX.strip("_")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cross-encoder/nli-MiniLM2-L6-H768")
ap.add_argument("--conditions", default="ABCDEF")
args = ap.parse_args()

cv = ClaimVerifier(NLIScorer(args.model), ClaimVerifierConfig(support_threshold=0.5, premise_mode=os.environ.get("PREMISE_MODE", "joint")))
ext = ClaimExtractor()
table = []
for c in args.conditions:
    p = RESULTS_DIR / f"{PREFIX}{c}.jsonl"
    if not p.exists():
        print(f"condition {c}: NOT YET MEASURED")
        continue
    rows = [r for r in read_jsonl(p) if not r["abstained"]]
    faithful = wrong_sup = wrong_unsup = 0
    for r in rows:
        claims, _ = claims_from_generation(ext, r["question"], r["answer"], r.get("statement") or "")
        res = cv.verify(claims, r["contexts"], r["retrieved_chunk_ids"])
        ok = all(x.verdict == "SUPPORTED" for x in res)
        correct = contains_gold(r["final_answer"], r["gold_answer"]) == 1.0
        faithful += ok
        wrong_sup += (not correct) and ok
        wrong_unsup += (not correct) and not ok
    n = len(rows)
    table.append({"condition": c, "answered": n, "faithful_rate": faithful / n if n else float("nan"),
                  "wrong_but_supported_by_evidence": wrong_sup, "wrong_and_unsupported(hallucination-like)": wrong_unsup})
df = pd.DataFrame(table)
print(f"\nIndependent NLI model: {args.model}\n")
print(df.to_string(index=False))
out = RESULTS_DIR / REPORT_DIR
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "faithfulness.csv", index=False)
print("\nwrong_but_supported = retrieval/evidence-induced error; wrong_and_unsupported = closer to hallucination.")
print(f"saved {out / 'faithfulness.csv'}")
