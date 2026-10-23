"""Small shared helpers: seeding, JSON/JSONL IO, hashing, timing."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0