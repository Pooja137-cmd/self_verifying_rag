"""Calibrate risk-level thresholds on questions DISJOINT from the evaluation set (retrieval only, no LLM).
Label: retrieval failure = not all gold paragraphs in top-k."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.agents.evidence_verifier import EvidenceVerifier  # noqa: E402
from src.agents.risk_controller import RiskConfig, RiskController  # noqa: E402
from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, Config  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=40, help="questions [0, eval-n) are the TEST set; excluded here")
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--questions", default=str(DATA_PROCESSED / "questions.json"))
    args = ap.parse_args()

    cfg = Config.from_env()
    store = VectorStore.load(args.index_dir, Embedder(cfg.embedding_model, cfg.query_instruction))
    pool = [q for q in read_json(args.questions)[args.eval_n:] if q["gold_doc_ids"]]
    results = store.search_many([q["question"] for q in pool], k=cfg.top_k)

    base = RiskConfig()
    rc, ev = RiskController(base), EvidenceVerifier()
    scores, fail = [], []
    for q, res in zip(pool, results):
        scores.append(rc.score(ev.assess(q["question"], res, check_contradiction=False).signals()))
        fail.append(int(not set(q["gold_doc_ids"]) <= {r.chunk.doc_id for r in res}))
    scores, fail = np.array(scores), np.array(fail)
    print(f"calibration pool: {len(pool)} questions, retrieval failures: {int(fail.sum())}")
    if fail.sum() < 5 or (fail == 0).sum() < 5:
        print("Too few failures or successes to calibrate; keeping default thresholds.")
        return
    auc = float(roc_auc_score(fail, scores))
    print(f"AUC of risk score for predicting retrieval failure: {auc:.3f}  (0.5 = no signal)")

    best_t, best_f1 = float(scores.max()), -1.0
    for t in np.unique(scores):
        pred = scores >= t
        tp = int((pred & (fail == 1)).sum())
        fp = int((pred & (fail == 0)).sum())
        fn = int((~pred & (fail == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    low = float(np.quantile(scores, 0.33))
    if low >= best_t:
        low = best_t * 0.6

    print(f"\nthresholds: LOW < {low:.3f} <= MEDIUM < {best_t:.3f} <= HIGH   (HIGH chosen to maximise failure-F1={best_f1:.3f})")
    for lvl, mask in (("LOW", scores < low), ("MEDIUM", (scores >= low) & (scores < best_t)), ("HIGH", scores >= best_t)):
        n = int(mask.sum())
        print(f"  {lvl:<6} n={n:<4} retrieval-failure rate = {fail[mask].mean():.3f}" if n else f"  {lvl:<6} n=0")
    if auc < 0.6:
        print("\nWARNING: weak signal (AUC < 0.6). Risk-gating will be close to arbitrary; report this honestly.")

    out = RiskConfig(low_threshold=low, high_threshold=best_t).to_dict()
    out["calibration"] = {"pool_size": len(pool), "failures": int(fail.sum()), "auc": auc,
                          "eval_n_excluded": args.eval_n, "criterion": "HIGH=max failure-F1; LOW=33rd percentile"}
    write_json(RESULTS_DIR / "risk_config.json", out)
    print(f"\nsaved {RESULTS_DIR / 'risk_config.json'}")


if __name__ == "__main__":
    main()
