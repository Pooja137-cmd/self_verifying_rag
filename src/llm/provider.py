"""Ollama client (default, local, free) and a small factory so providers stay swappable."""
from __future__ import annotations

import time

import requests

from src.llm.base import LLMClient, LLMResponse


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", temperature: float = 0.0,
                 seed: int = 42, num_ctx: int = 4096, timeout_s: int = 900, keep_alive: str = "30m") -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive

    def _generate(self, prompt: str, system: str | None, json_mode: bool, max_tokens: int) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"

        last_exc: Exception | None = None
        for attempt in range(2):  # one retry for transient failures
            try:
                t0 = time.perf_counter()
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                return LLMResponse(
                    text=data["message"]["content"],
                    prompt_tokens=int(data.get("prompt_eval_count", 0)),
                    completion_tokens=int(data.get("eval_count", 0)),
                    latency_s=time.perf_counter() - t0,
                    model=self.model,
                )
            except requests.ConnectionError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url}. Start it (open the Ollama app or run "
                    f"'ollama serve') and check the model exists with 'ollama list'."
                ) from exc
            except Exception as exc:
                last_exc = exc
                time.sleep(1)
        raise RuntimeError(f"Ollama call failed after retry: {last_exc}")


def get_llm(cfg) -> LLMClient:
    if cfg.llm_provider == "ollama":
        return OllamaClient(cfg.llm_model, cfg.llm_base_url, cfg.llm_temperature, cfg.seed)
    raise ValueError(f"Unknown llm_provider: {cfg.llm_provider!r} (add it in src/llm/provider.py)")
    