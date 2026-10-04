"""Environment verification. Run: python scripts/verify_env.py [--full]"""
import importlib
import importlib.metadata as md
import platform
import sys

PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
    ("pydantic", "pydantic"), ("dotenv", "python-dotenv"), ("faiss", "faiss-cpu"),
    ("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
    ("langgraph", "langgraph"), ("streamlit", "streamlit"),
    ("openai", "openai"), ("pytest", "pytest"),
]


def check_imports() -> bool:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Machine : {platform.machine()}  ({platform.platform()})")
    ok = True
    for module, dist in PACKAGES:
        try:
            importlib.import_module(module)
            print(f"  [OK]   {dist:<24} {md.version(dist)}")
        except Exception as exc:  # report every failure, not just the first
            ok = False
            print(f"  [FAIL] {dist:<24} {type(exc).__name__}: {exc}")
    return ok


def smoke_test() -> bool:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    print("\nLoading embedding model (first run downloads ~130 MB)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    docs = [
        "The Eiffel Tower is located in Paris, France.",
        "Photosynthesis converts light energy into chemical energy.",
        "The Great Wall of China is thousands of kilometres long.",
    ]
    emb = model.encode(docs, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    q = model.encode(["Where is the Eiffel Tower?"], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, 2)
    top = int(ids[0][0])
    print(f"  embedding dim={emb.shape[1]}, top hit id={top}, score={scores[0][0]:.3f}")
    emb_ok = top == 0
    print(f"  [{'OK' if emb_ok else 'FAIL'}] retrieval returns the Eiffel Tower document")

    class S(TypedDict):
        x: int

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"x": s["x"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    out = g.compile().invoke({"x": 1})
    lg_ok = out["x"] == 2
    print(f"  [{'OK' if lg_ok else 'FAIL'}] LangGraph minimal graph runs (x: 1 -> {out['x']})")
    return emb_ok and lg_ok


if __name__ == "__main__":
    imports_ok = check_imports()
    smoke_ok = smoke_test() if (imports_ok and "--full" in sys.argv) else None
    print("\nRESULT:", "imports OK" if imports_ok else "IMPORT FAILURES")
    if smoke_ok is not None:
        print("SMOKE :", "PASS" if smoke_ok else "FAIL")
    sys.exit(0 if imports_ok and smoke_ok is not False else 1)
