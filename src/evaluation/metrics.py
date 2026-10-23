"""Retrieval metrics. Doc-level gold labels, ranked list of retrieved chunks."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def retrieval_metrics_at_k(ranked_doc_ids: list[str], gold: Iterable[str], k: int) -> dict[str, float]:
    """ranked_doc_ids: doc_id of each retrieved chunk, best first (duplicates allowed)."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must be non-empty")
    top = ranked_doc_ids[:k]
    found = gold_set & set(top)
    return {
        "recall": len(found) / len(gold_set),                 # fraction of gold docs retrieved
        "any_hit": float(bool(found)),                        # at least one gold doc retrieved
        "all_hit": float(found == gold_set),                  # every gold doc retrieved (full support)
        "precision": (sum(d in gold_set for d in top) / len(top)) if top else 0.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}