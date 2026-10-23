"""Chunk the corpus, embed it, build and save the FAISS index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import DATA_PROCESSED, INDEX_DIR, Config  # noqa: E402
from src.retrieval.chunking import chunk_corpus  # noqa: E402
from src.retrieval.embeddings import Embedder  # noqa: E402
from src.retrieval.ingestion import load_documents  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils import Timer, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA_PROCESSED / "corpus.jsonl"))
    ap.add_argument("--index-dir", default=str(INDEX_DIR / "hotpot_dev200"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.from_env(embedding_model=args.model, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    set_seed(cfg.seed)

    docs = load_documents(args.corpus)
    chunks = chunk_corpus(docs, cfg.chunk_size, cfg.chunk_overlap)
    words = np.array([c.n_words for c in chunks])
    print(f"docs: {len(docs)}   chunks: {len(chunks)}   "
          f"words/chunk mean={words.mean():.0f} median={np.median(words):.0f} max={words.max()}")

    embedder = Embedder(cfg.embedding_model, cfg.query_instruction)
    with Timer() as t:
        store = VectorStore.build(chunks, embedder, meta_extra={
            "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "corpus_file": str(args.corpus)})
    store.save(args.index_dir)
    cfg.save(Path(args.index_dir) / "config_snapshot.json")
    print(f"embedded + indexed in {t.seconds:.1f}s  (dim={store.meta['dim']})")
    print(f"saved to {args.index_dir}")


if __name__ == "__main__":
    main()