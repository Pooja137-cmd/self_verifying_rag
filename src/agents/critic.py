"""Critic: decides the next action after a failed claim verification (rule-based, no LLM call).
Order is a heuristic design choice, evaluated by ablation, not a tuned optimum:
  CONTRADICTED claim and revision budget left -> REVISE (the draft conflicts with retrieved text)
  else re-retrieval budget left               -> RE_RETRIEVE
  else revision budget left                   -> REVISE
  else final policy                           -> ABSTAIN or ACCEPT_FLAGGED"""
from __future__ import annotations


class Critic:
    def __init__(self, revise: bool = False, max_rounds: int = 1, max_revisions: int = 1,
                 final_policy: str = "abstain") -> None:
        self.revise = revise
        self.max_rounds = max_rounds
        self.max_revisions = max_revisions
        self.final_policy = final_policy

    def decide(self, claim_results: list[dict], round_: int, revisions: int) -> str:
        contradicted = any(c["verdict"] == "CONTRADICTED" for c in claim_results)
        can_revise = self.revise and revisions < self.max_revisions
        can_retrieve = round_ < self.max_rounds
        if contradicted and can_revise:
            return "REVISE"
        if can_retrieve:
            return "RE_RETRIEVE"
        if can_revise:
            return "REVISE"
        if self.final_policy == "abstain":
            return "ABSTAIN"
        if self.final_policy == "abstain_if_contradicted" and contradicted:
            return "ABSTAIN"
        return "ACCEPT_FLAGGED"
