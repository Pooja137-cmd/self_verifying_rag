"""Audit helpers over saved per-question records (history-based). No LLM calls."""
from __future__ import annotations

from src.evaluation.answer_metrics import contains_gold, normalize_answer


def retrieval_triggers(history: list[dict]) -> tuple[int, int]:
    """(re-retrievals before the first generation = evidence-triggered, after = critic-triggered)."""
    seen_generate, pre, post = False, 0, 0
    for h in history:
        step = h.get("step")
        if step == "generate":
            seen_generate = True
        elif step == "re_retrieve":
            if seen_generate:
                post += 1
            else:
                pre += 1
    return pre, post


def answer_trajectory(history: list[dict]) -> list[str]:
    """Answers produced in order (initial generation, regenerations, revisions)."""
    return [h["answer"] for h in history if h.get("step") in ("generate", "revise") and h.get("answer") is not None]


def answer_changed(rec: dict) -> bool:
    traj = answer_trajectory(rec.get("history", []))
    return len(traj) > 1 and normalize_answer(traj[0]) != normalize_answer(traj[-1])


def correction_outcome(rec: dict) -> str:
    """Effect of the correction loop (critic/revise) judged by gold-answer containment."""
    hist = rec.get("history", [])
    if not any(h.get("step") in ("critic", "revise") for h in hist):
        return "no_correction"
    traj = answer_trajectory(hist)
    first_ok = bool(traj) and contains_gold(traj[0], rec["gold_answer"]) == 1.0
    if rec["abstained"]:
        return "abstained_after_correct_first" if first_ok else "abstained_after_wrong_first"
    last_ok = contains_gold(rec.get("answer") or "", rec["gold_answer"]) == 1.0
    if first_ok and last_ok:
        return "kept_correct"
    if first_ok and not last_ok:
        return "broke_correct"
    if not first_ok and last_ok:
        return "fixed_wrong"
    return "still_wrong"
