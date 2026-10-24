"""Abbreviation-aware sentence splitting for claims and NLI premises.
chunking.py is left unchanged on purpose so existing indexes stay valid."""
from __future__ import annotations

import re

from src.retrieval.chunking import clean_text

_PH = "\ue000"
_ABBR = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Vol|Vols|No|Nos|Inc|Ltd|Co|Corp|vs|Prof|Mt|Gen|Col|Lt|Sgt|Rev|Hon|Ft|Gov|Sen|Rep|Capt|ca|fl|pp)\.(?=\s)")
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s)")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences_safe(text: str) -> list[str]:
    t = clean_text(text or "")
    if not t:
        return []
    t = _ABBR.sub(lambda m: m.group(1) + _PH, t)
    t = _INITIAL.sub(lambda m: m.group(1) + _PH, t)
    return [p.replace(_PH, ".").strip() for p in _SPLIT.split(t) if p.strip()]
