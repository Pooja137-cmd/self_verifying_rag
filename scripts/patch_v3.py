"""Idempotent patches: result-prefix support in report scripts, revision_rate, faithfulness premise mode."""
from pathlib import Path

BLOCK = ('import os  # noqa: E402\n\nPREFIX = os.environ.get("RESULT_PREFIX", "cond_")\n'
         'REPORT_DIR = "report" if PREFIX == "cond_" else "report_" + PREFIX.strip("_")\n')
ANCHOR = "from src.utils import read_jsonl  # noqa: E402\n"

for f in ("scripts/make_report.py", "scripts/error_analysis.py", "scripts/faithfulness_eval.py",
          "scripts/random_baseline.py"):
    p = Path(f)
    t = p.read_text(encoding="utf-8")
    if "RESULT_PREFIX" not in t:
        assert ANCHOR in t, f"{f}: anchor not found"
        t = t.replace(ANCHOR, ANCHOR + BLOCK, 1)
        t = t.replace('f"cond_{c}.jsonl"', 'f"{PREFIX}{c}.jsonl"')
        t = t.replace('RESULTS_DIR / "report"', "RESULTS_DIR / REPORT_DIR")
    if f.endswith("faithfulness_eval.py") and "PREMISE_MODE" not in t:
        old = "ClaimVerifierConfig(support_threshold=0.5)"
        assert old in t, "faithfulness_eval: config call not found"
        t = t.replace(old, 'ClaimVerifierConfig(support_threshold=0.5, premise_mode=os.environ.get("PREMISE_MODE", "joint"))')
    if f.endswith("make_report.py") and '"revision": s.get' not in t:
        old = '"re_retrieval": s["re_retrieval_rate"],'
        assert old in t, "make_report: column anchor not found"
        t = t.replace(old, old + ' "revision": s.get("revision_rate"),')
    p.write_text(t, encoding="utf-8")

p = Path("src/evaluation/evaluator.py")
t = p.read_text(encoding="utf-8")
if "revision_rate" not in t:
    old = '        "re_retrieval_rate": _mean([float(r["re_retrieved"]) for r in rows]),\n'
    assert old in t, "evaluator: anchor not found"
    t = t.replace(old, old + '        "revision_rate": _mean([float(r.get("revisions", 0) > 0) for r in rows]),\n')
    p.write_text(t, encoding="utf-8")
print("patched OK")
