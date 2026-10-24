"""Reviser: one LLM call that corrects a draft using only the context and the verifier's feedback."""
from __future__ import annotations

from src.llm.prompts import parse_generation
from src.llm.revise_prompts import REVISE_SYSTEM, build_revise_prompt


class Reviser:
    def __init__(self, llm, max_tokens: int = 96) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def revise(self, question: str, contexts: list[str], answer: str, statement: str,
               claim_results: list[dict]) -> tuple[str, str, bool]:
        feedback = [f'- "{c["claim"]}" -> {c["verdict"]}' for c in claim_results if c["verdict"] != "SUPPORTED"]
        resp = self.llm.generate(build_revise_prompt(question, contexts, answer, statement, feedback),
                                 system=REVISE_SYSTEM, json_mode=True, max_tokens=self.max_tokens)
        return parse_generation(resp.text)
