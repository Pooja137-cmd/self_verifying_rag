"""Disk cache around any LLMClient. Deterministic settings (temperature 0, fixed seed) make reuse valid.
Cached responses keep their ORIGINAL tokens/latency, so usage reflects standalone cost."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.llm.base import LLMClient, LLMResponse
from src.utils import sha256_text


class CachedLLM(LLMClient):
    def __init__(self, inner: LLMClient, cache_path: str | Path) -> None:
        super().__init__()
        self.inner = inner
        self.model = inner.model
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.wall_s = 0.0   # real time spent inside this wrapper (cache lookups + real generations)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self.store[row["key"]] = row

    def _key(self, prompt: str, system: str | None, json_mode: bool, max_tokens: int) -> str:
        ident = {
            "model": self.inner.model, "temperature": getattr(self.inner, "temperature", None),
            "seed": getattr(self.inner, "seed", None), "num_ctx": getattr(self.inner, "num_ctx", None),
            "system": system, "prompt": prompt, "json": json_mode, "max_tokens": max_tokens,
        }
        return sha256_text(json.dumps(ident, sort_keys=True))

    def _generate(self, prompt: str, system: str | None, json_mode: bool, max_tokens: int) -> LLMResponse:
        t0 = time.perf_counter()
        key = self._key(prompt, system, json_mode, max_tokens)
        row = self.store.get(key)
        if row is None:
            self.misses += 1
            r = self.inner.generate(prompt, system=system, json_mode=json_mode, max_tokens=max_tokens)
            row = {"key": key, "text": r.text, "prompt_tokens": r.prompt_tokens,
                   "completion_tokens": r.completion_tokens, "latency_s": r.latency_s, "model": r.model}
            self.store[key] = row
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            self.hits += 1
        self.wall_s += time.perf_counter() - t0
        return LLMResponse(row["text"], row["prompt_tokens"], row["completion_tokens"],
                           row["latency_s"], row["model"])
