"""LLM abstraction: a common interface plus usage accounting (the 'cost' side of the study)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


@dataclass
class UsageTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency_s += r.latency_s

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> "UsageTracker":
        return UsageTracker(self.calls, self.prompt_tokens, self.completion_tokens, self.latency_s)

    def delta(self, earlier: "UsageTracker") -> dict:
        return {
            "llm_calls": self.calls - earlier.calls,
            "prompt_tokens": self.prompt_tokens - earlier.prompt_tokens,
            "completion_tokens": self.completion_tokens - earlier.completion_tokens,
            "total_tokens": self.total_tokens - earlier.total_tokens,
            "llm_latency_s": self.latency_s - earlier.latency_s,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


class LLMClient(ABC):
    model: str

    def __init__(self) -> None:
        self.usage = UsageTracker()

    @abstractmethod
    def _generate(self, prompt: str, system: str | None, json_mode: bool, max_tokens: int) -> LLMResponse: ...

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 max_tokens: int = 256) -> LLMResponse:
        resp = self._generate(prompt, system, json_mode, max_tokens)
        self.usage.add(resp)
        return resp