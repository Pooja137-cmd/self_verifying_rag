"""Condition-level summary: quality, reliability and cost from saved per-question records."""
from __future__ import annotations

import numpy as np

from src.evaluation.answer_metrics import contains_gold, exact_match, f1_score


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def summarize_condition(rows: list[dict]) -> dict:
    def contains(r):
        return contains_gold(r["final_answer"], r["gold_answer"])

    answered = [r for r in rows if not r["abstained"]]
    wrong = [float((not r["abstained"]) and contains(r) == 0.0) for r in rows]
    return {
        "n": len(rows),
        "EM": _mean([exact_match(r["final_answer"], r["gold_answer"]) for r in rows]),
        "F1": _mean([f1_score(r["final_answer"], r["gold_answer"]) for r in rows]),
        "contains_gold": _mean([contains(r) for r in rows]),
        "abstain_rate": _mean([float(r["abstained"]) for r in rows]),
        "answered_precision": _mean([contains(r) for r in answered]),
        "wrong_answer_rate": _mean(wrong),
        "safe_rate": (1.0 - _mean(wrong)) if rows else None,
        "initial_all_hit": _mean([float(r["initial_all_hit"]) for r in rows]),
        "final_all_hit": _mean([float(r["final_all_hit"]) for r in rows]),
        "re_retrieval_rate": _mean([float(r["re_retrieved"]) for r in rows]),
        "revision_rate": _mean([float(r.get("revisions", 0) > 0) for r in rows]),
        "claim_verification_rate": _mean([float(r["claim_verified"]) for r in rows]),
        "avg_latency_s": _mean([r["latency_s"] for r in rows]),
        "avg_llm_calls": _mean([r["llm_calls"] for r in rows]),
        "avg_prompt_tokens": _mean([r["prompt_tokens"] for r in rows]),
        "avg_completion_tokens": _mean([r["completion_tokens"] for r in rows]),
        "avg_total_tokens": _mean([r["total_tokens"] for r in rows]),
        "avg_nli_pairs": _mean([r["nli_pairs"] for r in rows]),
    }
