from pathlib import Path


def patch(path, edits):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    for old, new in edits:
        assert old in t, f"{path}: expected text not found: {old[:60]!r}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")


a = '    ap.add_argument("--use-contradiction", action="store_true")\n'
b = "    cv_cfg = ClaimVerifierConfig(support_threshold=support)\n"
patch("experiments/run_experiment.py", [
    (a, a + '    ap.add_argument("--premise-mode", default="joint", choices=["joint", "pairs"])\n'),
    (b, "    cv_cfg = ClaimVerifierConfig(support_threshold=support, premise_mode=args.premise_mode)\n")])

anchor = "from src.utils import read_jsonl  # noqa: E402\n"
block = ('import os  # noqa: E402\n\nPREFIX = os.environ.get("RESULT_PREFIX", "cond_")\n'
         'REPORT_DIR = "report" if PREFIX == "cond_" else "report_" + PREFIX.strip("_")\n')
for f in ("scripts/make_report.py", "scripts/error_analysis.py", "scripts/faithfulness_eval.py"):
    patch(f, [(anchor, anchor + block), ('f"cond_{c}.jsonl"', 'f"{PREFIX}{c}.jsonl"'),
              ('RESULTS_DIR / "report"', "RESULTS_DIR / REPORT_DIR")])
print("patched OK")
