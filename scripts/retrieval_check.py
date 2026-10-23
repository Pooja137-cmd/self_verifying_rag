"""Measure retrieval quality of the saved index (Recall@k etc.) against gold docs."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, Config  # noqa: E402
from src.evaluation.metrics import mean_metrics, retrieval_metrics_at_k  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import Timer, read_json, write_json  # noqa: E402

KS = (1, 3, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--questions", default=str(DATA_PROCESSED / "questions.json"))
    args = ap.parse_args()

    cfg = Config.from_env()
    store = VectorStore.load(args.index_dir, Embedder(cfg.embedding_model, cfg.query_instruction))
    questions = [q for q in read_json(args.questions) if q["gold_doc_ids"]]

    with Timer() as t:
        results = store.search_many([q["question"] for q in questions], k=max(KS))

    per_k: dict[int, list] = defaultdict(list)
    per_type: dict[tuple, list] = defaultdict(list)
    for q, res in zip(questions, results):
        ranked = [r.chunk.doc_id for r in res]
        for k in KS:
            m = retrieval_metrics_at_k(ranked, q["gold_doc_ids"], k)
            per_k[k].append(m)
            per_type[(q["type"], k)].append(m)

    summary = {f"@{k}": mean_metrics(per_k[k]) for k in KS}
    by_type = {f"{ty}@{k}": mean_metrics(v) for (ty, k), v in sorted(per_type.items(), key=str)}

    print(f"questions: {len(questions)}   search time: {t.seconds:.1f}s\n")
    print(f"{'k':>3} {'recall':>8} {'any_hit':>8} {'all_hit':>8} {'precision':>10}")
    for k in KS:
        m = summary[f"@{k}"]
        print(f"{k:>3} {m['recall']:>8.3f} {m['any_hit']:>8.3f} {m['all_hit']:>8.3f} {m['precision']:>10.3f}")
    print("\n(all_hit = every gold paragraph retrieved; multi-hop questions need both)")

    out = RESULTS_DIR / "retrieval_check.json"
    write_json(out, {"index_meta": store.meta, "config": cfg.to_dict(), "n_questions": len(questions),
                     "summary": summary, "by_question_type": by_type})
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()