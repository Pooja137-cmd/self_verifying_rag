"""Phase 1 audit of saved outputs. No LLM calls.  python scripts/audit_results.py
Checks per result group (prefix): same question ids across A-F, identical settings, what 'revision' really
measures, and standalone-estimated vs actual run time."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import DATA_PROCESSED, RESULTS_DIR  # noqa: E402
from src.evaluation.audit import answer_changed, correction_outcome, retrieval_triggers  # noqa: E402
from src.utils import read_json, read_jsonl  # noqa: E402

CONDS = "ABCDEF"
MUST_MATCH = ["model", "nli_model", "app_config", "risk_config", "claim_verifier_config"]
LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def diff_keys(vals: dict) -> str:
    dicts = {c: json.loads(v) for c, v in vals.items()}
    if all(isinstance(d, dict) for d in dicts.values()):
        keys = sorted({k for d in dicts.values() for k in d})
        bad = [k for k in keys if len({json.dumps(d.get(k, "<absent>"), sort_keys=True) for d in dicts.values()}) > 1]
        return "differs in: " + ", ".join(bad)
    return "values: " + "; ".join(f"{c}={v}" for c, v in vals.items())


def main() -> None:
    questions = read_json(DATA_PROCESSED / "questions.json")
    prefixes = sorted({p.stem[:-1] for p in RESULTS_DIR.glob("*A.jsonl")})
    for prefix in prefixes:
        recs = {c: list(read_jsonl(RESULTS_DIR / f"{prefix}{c}.jsonl")) for c in CONDS
                if (RESULTS_DIR / f"{prefix}{c}.jsonl").exists()}
        if len(recs) < 2 or "A" not in recs:
            continue
        out(f"\n{'=' * 78}\nGROUP '{prefix}'  conditions present: {''.join(recs)}   "
            f"(missing: {''.join(c for c in CONDS if c not in recs) or 'none'})\n{'=' * 78}")

        out("\n[1] Same questions?")
        ref = [r["id"] for r in recs["A"]]
        sA = RESULTS_DIR / f"{prefix}A_summary.json"
        offset = read_json(sA).get("offset", 0) if sA.exists() else 0
        expected = [q["id"] for q in questions[offset: offset + len(ref)]]
        out(f"  A has {len(ref)} questions; ids match questions.json[{offset}:{offset + len(ref)}]: {ref == expected}")
        for c, rows in recs.items():
            ids = [r["id"] for r in rows]
            out(f"  {c}: n={len(ids)}  same ids as A: {set(ids) == set(ref)}  same order: {ids == ref}  "
                f"duplicates: {len(ids) - len(set(ids))}")

        out("\n[2] Same settings? (from saved *_summary.json)")
        sums = {c: read_json(RESULTS_DIR / f"{prefix}{c}_summary.json") for c in recs
                if (RESULTS_DIR / f"{prefix}{c}_summary.json").exists()}
        if len(sums) < len(recs):
            out(f"  summaries missing for: {''.join(c for c in recs if c not in sums)}")
        for key in MUST_MATCH:
            vals = {c: json.dumps(s.get(key, "<absent>"), sort_keys=True) for c, s in sums.items()}
            out(f"  {key}: " + ("identical across conditions" if len(set(vals.values())) == 1 else
                                f"MISMATCH - {diff_keys(vals)}"))
        if "A" in sums:
            pcA = sums["A"].get("pipeline_config", {})
            out("  pipeline_config differences vs A (expected: these define the conditions):")
            for c, s in sums.items():
                pc = s.get("pipeline_config", {})
                out(f"    {c}: " + (str({k: v for k, v in pc.items() if pcA.get(k) != v}) or "{}"))

        out("\n[3] What 'revision' and 're-retrieval' really measure")
        rows_out, outcomes = [], {}
        for c, rows in recs.items():
            n = len(rows)
            pre = sum(retrieval_triggers(r["history"])[0] > 0 for r in rows)
            post = sum(retrieval_triggers(r["history"])[1] > 0 for r in rows)
            rev_steps = sum(any(h["step"] == "revise" for h in r["history"]) for r in rows)
            summ = sums.get(c, {})
            rows_out.append({
                "cond": c, "n": n, "revise_enabled": summ.get("pipeline_config", {}).get("revise", "absent(pre-v3)"),
                "revise_steps": rev_steps, "revision_rate_in_summary": summ.get("revision_rate", "absent"),
                "re_retrieval_before_generation": pre, "re_retrieval_after_claim_check": post,
                "answer_changed": sum(answer_changed(r) for r in rows)})
            outcomes[c] = Counter(correction_outcome(r) for r in rows)
        out(pd.DataFrame(rows_out).to_string(index=False))
        out("\n  Effect of the correction loop (first answer vs final, by gold containment):")
        out(pd.DataFrame(outcomes).fillna(0).astype(int).to_string())
        out("  'fixed_wrong' = wrong first answer became correct; 'broke_correct' = correct became wrong.")
        out("  Without Revise (pre-v3), 'answer_changed' comes from re-retrieval + regeneration, not from revision.")

        out("\n[4] Standalone-estimated vs actual time")
        lat = []
        for c, rows in recs.items():
            calls = sum(r["llm_calls"] for r in rows)
            hits = sum(r["llm_cache_hits"] for r in rows)
            lat.append({"cond": c, "avg_standalone_s": sum(r["latency_s"] for r in rows) / len(rows),
                        "avg_actual_wall_s": sum(r["wall_s"] for r in rows) / len(rows),
                        "actual_total_min": sum(r["wall_s"] for r in rows) / 60,
                        "llm_calls": calls, "served_from_cache": hits,
                        "cache_share": hits / calls if calls else 0.0})
        out(pd.DataFrame(lat).round(2).to_string(index=False))
        out("  standalone = what the condition would cost run alone (original LLM times replayed from the cache).")
        out("  actual = real wall time this run took; low values mean most calls were served from the cache.")
        out("  Use standalone time and LLM calls/tokens for cost comparisons; do not quote actual wall time.")
    p = RESULTS_DIR / "report" / "audit.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("```\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
