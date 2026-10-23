"""Central configuration. Every experiment saves a snapshot of this."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "indexes"
RESULTS_DIR = ROOT / "evaluation" / "results"


@dataclass
class Config:
    seed: int = 42

    # retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    chunk_size: int = 200      # words
    chunk_overlap: int = 40    # words
    top_k: int = 5

    # llm (used from Phase 4 onward)
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Defaults < .env < explicit overrides (None overrides are ignored)."""
        env_map = {
            "seed": ("RANDOM_SEED", int),
            "embedding_model": ("EMBEDDING_MODEL", str),
            "llm_provider": ("LLM_PROVIDER", str),
            "llm_base_url": ("LLM_BASE_URL", str),
            "llm_model": ("LLM_MODEL", str),
        }
        values: dict[str, Any] = {}
        for field_name, (env_name, caster) in env_map.items():
            raw = os.getenv(env_name)
            if raw:
                values[field_name] = caster(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")