"""Condition A: vanilla RAG baseline. Resumable: rerun the same command to continue after a crash.

Usage: python experiments/baseline_rag.py --n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, Config  # noqa: E402
from src.evaluation.answer_metrics import contains_gold, exact_match, f1_score, is_abstention  # noqa: E402
from src.evaluation.metrics import retrieval_metrics_at_k  # noqa: E402
from src.llm.provider import get_llm  # noqa: E402
from src.pipeline.vanilla import run_vanilla_rag  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import read_json, read_jsonl, set_seed, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of questions")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--model", default=None, help="override Ollama model tag")
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--questions", default=str(DATA_PROCESSED / "questions.json"))
    ap.add_argument("--name", default="baseline_rag")
    args = ap.parse_args()

    cfg = Config.from_env(top_k=args.top_k, llm_model=args.model)
    set_seed(cfg.seed)
    store = VectorStore.load(args.index_dir, Embedder(cfg.embedding_model, cfg.query_instruction))
    llm = get_llm(cfg)
    questions = read_json(args.questions)[: args.n]

    out_jsonl = RESULTS_DIR / f"{args.name}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out_jsonl)} if out_jsonl.exists() else set()
    print(f"model={cfg.llm_model}  top_k={cfg.top_k}  questions={len(questions)}  already done={len(done)}")

    with out_jsonl.open("a", encoding="utf-8") as f:
        import json
        for i, q in enumerate(questions, 1):
            if q["id"] in done:
                continue
            rec = run_vanilla_rag(q["question"], store, llm, top_k=cfg.top_k)
            rec.update({"id": q["id"], "gold_answer": q["answer"], "gold_doc_ids": q["gold_doc_ids"],
                        "type": q.get("type")})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(questions)}] {rec['latency_s']:.1f}s  pred={rec['answer']!r}  gold={q['answer']!r}")

    rows = [r for r in read_jsonl(out_jsonl) if r["id"] in {q["id"] for q in questions}]
    ranked = lambda r: [x["doc_id"] for x in r["retrieved"]]  # noqa: E731
    ret = [retrieval_metrics_at_k(ranked(r), r["gold_doc_ids"], cfg.top_k) for r in rows if r["gold_doc_ids"]]
    summary = {
        "condition": "A_vanilla_rag", "n": len(rows), "model": cfg.llm_model, "top_k": cfg.top_k,
        "EM": float(np.mean([exact_match(r["answer"], r["gold_answer"]) for r in rows])),
        "F1": float(np.mean([f1_score(r["answer"], r["gold_answer"]) for r in rows])),
        "contains_gold": float(np.mean([contains_gold(r["answer"], r["gold_answer"]) for r in rows])),
        "abstain_rate": float(np.mean([is_abstention(r["answer"]) for r in rows])),
        "retrieval_all_hit@k": float(np.mean([m["all_hit"] for m in ret])),
        "retrieval_any_hit@k": float(np.mean([m["any_hit"] for m in ret])),
        "avg_latency_s": float(np.mean([r["latency_s"] for r in rows])),
        "avg_llm_calls": float(np.mean([r["llm_calls"] for r in rows])),
        "avg_total_tokens": float(np.mean([r["total_tokens"] for r in rows])),
        "config": cfg.to_dict(),
    }
    write_json(RESULTS_DIR / f"{args.name}_summary.json", summary)
    print("\n=== SUMMARY (measured on this run) ===")
    for k, v in summary.items():
        if k != "config":
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()