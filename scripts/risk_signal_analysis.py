"""Do cheap retrieval signals predict retrieval failure (a gold paragraph missing from top-k)?
Label: all_hit@k. Reports AUC per feature and a cross-validated logistic regression."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.agents.evidence_verifier import question_coverage  # noqa: E402
from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, Config  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--questions", default=str(DATA_PROCESSED / "questions.json"))
    args = ap.parse_args()

    cfg = Config.from_env()
    store = VectorStore.load(args.index_dir, Embedder(cfg.embedding_model, cfg.query_instruction))
    qs = [q for q in read_json(args.questions) if q["gold_doc_ids"]]
    res = store.search_many([q["question"] for q in qs], k=args.k)

    X, y = [], []
    for q, r in zip(qs, res):
        s = [x.score for x in r]
        X.append([s[0], s[0] - s[1], float(np.mean(s)), s[-1],
                  question_coverage(q["question"], [x.chunk.text for x in r])])
        y.append(int(set(q["gold_doc_ids"]) <= {x.chunk.doc_id for x in r}))
    X, y = np.array(X), np.array(y)
    names = ["top1", "gap_top1_top2", "mean_topk", "min_topk", "coverage"]
    n_fail = int((y == 0).sum())
    print(f"questions={len(y)}  all_hit@{args.k}={y.mean():.3f}  failures={n_fail}")
    if n_fail < 5 or y.sum() < 5:
        print("Too few failures/successes for a meaningful AUC.")
        return

    print("\nAUC of each feature for predicting SUCCESS (0.5 = no signal; <0.5 = inverted):")
    aucs = {n: float(roc_auc_score(y, X[:, i])) for i, n in enumerate(names)}
    for n, a in aucs.items():
        print(f"  {n:<14} {a:.3f}")
    skf = StratifiedKFold(n_splits=min(5, n_fail, int(y.sum())), shuffle=True, random_state=cfg.seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y, p))
    print(f"\ncross-validated logistic regression AUC: {cv_auc:.3f}  (n={len(y)}, failures={n_fail}: noisy)")
    write_json(RESULTS_DIR / "risk_signal_analysis.json",
               {"k": args.k, "n": len(y), "failures": n_fail, "feature_auc": aucs, "cv_logreg_auc": cv_auc})
    print(f"saved {RESULTS_DIR / 'risk_signal_analysis.json'}")


if __name__ == "__main__":
    main()
