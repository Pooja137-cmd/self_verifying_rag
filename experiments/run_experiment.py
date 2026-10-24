"""Run one experimental condition (A-F) over a slice of questions. Resumable.

  python experiments/run_experiment.py --condition E --n 40 --name cond_E
Latency reported = non-LLM wall time + LLM time of the ORIGINAL (uncached) generations, i.e. standalone cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.claim_extractor import ClaimExtractor  # noqa: E402
from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig  # noqa: E402
from src.agents.evidence_verifier import EvidenceVerifier  # noqa: E402
from src.agents.nli import NLIScorer  # noqa: E402
from src.agents.planner import QueryPlanner  # noqa: E402
from src.agents.retriever_agent import RetrievalAgent  # noqa: E402
from src.agents.risk_controller import RiskConfig, RiskController  # noqa: E402
from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, ROOT, Config  # noqa: E402
from src.evaluation.evaluator import summarize_condition  # noqa: E402
from src.graph.state import make_condition  # noqa: E402
from src.graph.workflow import Pipeline  # noqa: E402
from src.llm.cache import CachedLLM  # noqa: E402
from src.llm.provider import get_llm  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import read_json, read_jsonl, set_seed, write_json  # noqa: E402

CACHE_PATH = ROOT / "evaluation" / "cache" / "llm_cache.jsonl"


def build_record(item: dict, state: dict, cond: str, delta: dict, latency: float, wall: float,
                 nli_pairs: int, cache_hits: int) -> dict:
    gold = set(item["gold_doc_ids"])
    first_docs = state["history"][0]["doc_ids"]
    results = state["results"]
    final_docs = [r.chunk.doc_id for r in results]
    return {
        "id": item["id"], "condition": cond, "question": item["question"], "gold_answer": item["answer"],
        "gold_doc_ids": item["gold_doc_ids"], "type": item.get("type"),
        "final_answer": state["final_answer"], "abstained": state["abstained"],
        "flagged_unverified": state.get("flagged_unverified", False),
        "answer": state.get("answer"), "statement": state.get("statement"), "parse_ok": state.get("parse_ok"),
        "raw_output": state.get("raw_output"), "claims": state.get("claims", []),
        "claim_fallback": state.get("claim_fallback", False), "claim_results": state.get("claim_results", []),
        "claim_risk": state.get("claim_risk"), "claim_verified": state.get("claim_verified", False),
        "critic_action": state.get("critic_action"),
        "risk_level": state.get("risk_level", "NA"), "risk_score": state.get("risk_score", 0.0),
        "evidence": state.get("evidence"),
        "rounds": state["round"], "re_retrieved": state["round"] > 0,
        "initial_all_hit": bool(gold <= set(first_docs)) if gold else False,
        "final_all_hit": bool(gold <= set(final_docs)) if gold else False,
        "retrieved_chunk_ids": [r.chunk.chunk_id for r in results],
        "contexts": [f"{r.chunk.title}: {r.chunk.text}" if r.chunk.title else r.chunk.text for r in results],
        "history": state["history"],
        "latency_s": latency, "wall_s": wall, "nli_pairs": nli_pairs, "llm_cache_hits": cache_hits, **delta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list("ABCDEF"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--support-threshold", type=float, default=None)
    ap.add_argument("--risk-config", default=str(RESULTS_DIR / "risk_config.json"))
    ap.add_argument("--claim-threshold-file", default=str(RESULTS_DIR / "claim_threshold.json"))
    ap.add_argument("--final-policy", default="abstain", choices=["abstain", "accept_flagged", "abstain_if_contradicted"])
    ap.add_argument("--use-contradiction", action="store_true")
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--questions", default=str(DATA_PROCESSED / "questions.json"))
    args = ap.parse_args()

    name = args.name or f"cond_{args.condition}"
    cfg = Config.from_env(top_k=args.top_k, llm_model=args.model)
    set_seed(cfg.seed)

    risk_path = Path(args.risk_config)
    risk_cfg = RiskConfig.load(risk_path) if risk_path.exists() else RiskConfig()
    thr_path = Path(args.claim_threshold_file)
    if args.support_threshold is not None:
        support = args.support_threshold
    elif thr_path.exists():
        support = float(read_json(thr_path)["support_threshold"])
    else:
        support = ClaimVerifierConfig().support_threshold
    cv_cfg = ClaimVerifierConfig(support_threshold=support)
    pcfg = make_condition(args.condition, final_policy=args.final_policy, use_contradiction=args.use_contradiction)

    store = VectorStore.load(args.index_dir, Embedder(cfg.embedding_model, cfg.query_instruction))
    llm = CachedLLM(get_llm(cfg), CACHE_PATH)
    nli = NLIScorer()
    pipe = Pipeline(pcfg, RetrievalAgent(store, QueryPlanner(), top_k=cfg.top_k, extra=3),
                    EvidenceVerifier(nli if args.use_contradiction else None), RiskController(risk_cfg),
                    llm, ClaimExtractor(), ClaimVerifier(nli, cv_cfg))

    questions = read_json(args.questions)[args.offset: args.offset + args.n]
    out = RESULTS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    print(f"condition={args.condition}  model={cfg.llm_model}  n={len(questions)}  done={len(done)}  "
          f"support_threshold={support}  risk=[low {risk_cfg.low_threshold:.3f}, high {risk_cfg.high_threshold:.3f}]")

    with out.open("a", encoding="utf-8") as f:
        for i, item in enumerate(questions, 1):
            if item["id"] in done:
                continue
            before, hits0, wall0_llm, pairs0 = llm.usage.snapshot(), llm.hits, llm.wall_s, nli.n_pairs
            t0 = time.perf_counter()
            state = pipe.run(item["question"])
            wall = time.perf_counter() - t0
            delta = llm.usage.delta(before)
            latency = (wall - (llm.wall_s - wall0_llm)) + delta["llm_latency_s"]
            rec = build_record(item, state, args.condition, delta, latency, wall,
                               nli.n_pairs - pairs0, llm.hits - hits0)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(questions)}] wall={wall:.0f}s standalone={latency:.0f}s calls={delta['llm_calls']} "
                  f"risk={rec['risk_level']} rr={int(rec['re_retrieved'])} cv={int(rec['claim_verified'])} "
                  f"pred={rec['final_answer']!r} gold={item['answer']!r}")

    ids = {q["id"] for q in questions}
    rows = [r for r in read_jsonl(out) if r["id"] in ids]
    summary = summarize_condition(rows)
    summary.update({"condition": args.condition, "model": cfg.llm_model, "pipeline_config": pcfg.to_dict(),
                    "risk_config": risk_cfg.to_dict(), "claim_verifier_config": cv_cfg.__dict__,
                    "app_config": cfg.to_dict(), "nli_model": nli.model_name,
                    "cache_hits_this_run": llm.hits, "cache_misses_this_run": llm.misses,
                    "offset": args.offset})
    write_json(RESULTS_DIR / f"{name}_summary.json", summary)
    print("\n=== SUMMARY (measured on this run) ===")
    for k, v in summary.items():
        if isinstance(v, (int, float, str)) or v is None:
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
