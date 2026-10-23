"""Query Planner (rule-based, no LLM): initial query and a query expansion for re-retrieval."""
from __future__ import annotations


class QueryPlanner:
    def __init__(self, expand_words: int = 60) -> None:
        self.expand_words = expand_words

    def initial_query(self, question: str) -> str:
        return question

    def expand_query(self, question: str, results: list) -> str:
        """Append the top-ranked chunk so multi-hop 'bridge' entities pull in the second paragraph.
        A simple heuristic; its benefit must be measured, not assumed."""
        if not results:
            return question
        top = results[0].chunk
        snippet = " ".join(top.text.split()[: self.expand_words])
        return f"{question} {top.title}. {snippet}".strip()
