"""Dataset-agnostic corpus loaders."""
from __future__ import annotations

import re
from pathlib import Path

from src.retrieval.chunking import Document
from src.utils import read_jsonl


def load_documents(path: str | Path) -> list[Document]:
    """Load a JSONL corpus: one {"doc_id", "text", "title"?, "metadata"?} per line."""
    docs = [Document(**row) for row in read_jsonl(path)]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate doc_id values in corpus")
    return docs


def load_documents_from_folder(folder: str | Path, exts: tuple[str, ...] = (".txt", ".md")) -> list[Document]:
    """Use your own files as a corpus (sorted for determinism)."""
    root = Path(folder)
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(root).as_posix()
            doc_id = re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_")
            docs.append(Document(doc_id=doc_id, title=p.stem, text=p.read_text(encoding="utf-8", errors="ignore"),
                                 metadata={"source": rel}))
    return docs