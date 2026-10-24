"""python scripts/web_smoke.py "your question" [--backend ddg|wikipedia]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.planner import QueryPlanner  # noqa: E402
from src.agents.web_retriever import WebRetrievalAgent  # noqa: E402
from src.config import Config  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.web import CachingClient, WebClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--backend", default="ddg", choices=["ddg", "wikipedia"])
ap.add_argument("--pages", type=int, default=3)
args = ap.parse_args()

cfg = Config.from_env()
client = CachingClient(WebClient(args.backend))
agent = WebRetrievalAgent(Embedder(cfg.embedding_model, cfg.query_instruction), client, QueryPlanner(),
                          top_k=cfg.top_k, n_pages=args.pages)
t0 = time.perf_counter()
results = agent.retrieve(args.question)
print(f"retrieval took {time.perf_counter() - t0:.1f}s   search log: {agent.log}")
if client.last_error:
    print("last error:", client.last_error)
for r in results:
    print(f"[{r.rank}] score={r.score:.3f}  {r.chunk.metadata.get('url')}\n     {r.chunk.text[:160]}...")
if not results:
    print("No evidence retrieved. If backend=ddg, try --backend wikipedia.")
