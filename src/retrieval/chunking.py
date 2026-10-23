"""Sentence-aware chunking with word-based size and overlap.

Chunk size is measured in words (a cheap proxy for tokens). Sentences are never
split unless a single sentence is longer than chunk_size.
Sentence splitting is a regex heuristic; abbreviations such as "U.S. Army" can
be split incorrectly. That is acceptable for retrieval chunking.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    n_words: int
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text that gets embedded: the title gives short paragraphs context."""
        return f"{self.title}. {self.text}" if self.title else self.text


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _hard_split(sentence: str, size: int, overlap: int) -> list[str]:
    """Fallback for a single sentence longer than `size` words."""
    words = sentence.split()
    step = max(1, size - overlap)
    out: list[str] = []
    i = 0
    while True:
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += step
    return out


def chunk_document(doc: Document, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = clean_text(doc.text)
    if not text:
        return []

    units: list[str] = []
    for sent in split_sentences(text):
        if len(sent.split()) > chunk_size:
            units.extend(_hard_split(sent, chunk_size, overlap))
        else:
            units.append(sent)

    texts: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        w = len(unit.split())
        if cur and cur_words + w > chunk_size:
            texts.append(" ".join(cur))
            # carry trailing sentences (never the whole chunk) as overlap
            carry: list[str] = []
            cw = 0
            for s in reversed(cur[:-1]):
                sw = len(s.split())
                if cw + sw > overlap:
                    break
                carry.insert(0, s)
                cw += sw
            while carry and cw + w > chunk_size:
                cw -= len(carry.pop(0).split())
            cur, cur_words = carry, cw
        cur.append(unit)
        cur_words += w
    if cur:
        texts.append(" ".join(cur))

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::c{i}",
            doc_id=doc.doc_id,
            chunk_index=i,
            text=t,
            n_words=len(t.split()),
            title=doc.title,
            metadata=dict(doc.metadata),
        )
        for i, t in enumerate(texts)
    ]


def chunk_corpus(docs: list[Document], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, chunk_size, overlap))
    return chunks