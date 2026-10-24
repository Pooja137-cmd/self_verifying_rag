"""LangGraph workflow (v3).
retrieve -> assess -> [re_retrieve] -> generate -> [extract_claims -> verify_claims -> critic ->
   REVISE (-> extract_claims -> verify_claims) | RE_RETRIEVE (-> generate ...) | finalize] -> finalize
Active branches depend on the PipelineConfig (conditions A-F, optional revise)."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.critic import Critic
from src.agents.reviser import Reviser
from src.evaluation.answer_metrics import normalize_answer
from src.graph.state import PipelineConfig, PipelineState
from src.llm.prompts import ABSTAIN, GEN_SYSTEM, build_rag_prompt, is_abstain, parse_generation


def _ctx(r) -> str:
    c = r.chunk
    return f"{c.title}: {c.text}" if c.title else c.text


def claims_from_generation(extractor, question: str, answer: str, statement: str) -> tuple[list[str], bool]:
    claims = extractor.extract(statement) if statement else []
    if claims:
        return claims, False
    return [f'The answer to the question "{question}" is {answer}.'], True


class Pipeline:
    def __init__(self, cfg: PipelineConfig, retriever, evidence_verifier, risk, llm, extractor,
                 claim_verifier, reviser=None) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.ev = evidence_verifier
        self.risk = risk
        self.llm = llm
        self.extractor = extractor
        self.cv = claim_verifier
        self.reviser = reviser if reviser is not None else (Reviser(llm) if cfg.revise else None)
        self.critic = Critic(revise=cfg.revise, max_rounds=cfg.max_rounds, max_revisions=cfg.max_revisions,
                             final_policy=cfg.final_policy)
        self.graph = self._build()

    def run(self, question: str) -> dict:
        return self.graph.invoke({"question": question, "history": [], "revisions": 0}, {"recursion_limit": 80})

    @staticmethod
    def _h(state, event: dict) -> list:
        return list(state.get("history", [])) + [event]

    def _assess_update(self, question: str, results: list) -> dict:
        if not (self.cfg.evidence_verification or self.cfg.adaptive):
            return {"risk_level": "NA", "risk_score": 0.0}
        ev = self.ev.assess(question, results, check_contradiction=self.cfg.use_contradiction)
        score, level = self.risk.assess(ev.signals())
        return {"evidence": ev.model_dump(), "risk_score": score, "risk_level": level}

    # ---- nodes
    def n_retrieve(self, state):
        results = self.retriever.retrieve(state["question"])
        return {"results": results, "round": 0, "claim_verified": False,
                "history": self._h(state, {"step": "retrieve", "round": 0,
                                           "doc_ids": [r.chunk.doc_id for r in results],
                                           "scores": [round(r.score, 4) for r in results]})}

    def n_assess(self, state):
        upd = self._assess_update(state["question"], state["results"])
        upd["history"] = self._h(state, {"step": "assess", "risk_level": upd["risk_level"],
                                         "risk_score": round(upd["risk_score"], 4)})
        return upd

    def n_re_retrieve(self, state):
        merged, q2 = self.retriever.re_retrieve(state["question"], state["results"])
        upd = {"results": merged, "round": state["round"] + 1}
        upd.update(self._assess_update(state["question"], merged))
        upd["history"] = self._h(state, {"step": "re_retrieve", "round": state["round"] + 1,
                                         "expanded_query": q2, "doc_ids": [r.chunk.doc_id for r in merged],
                                         "risk_level": upd.get("risk_level")})
        return upd

    def n_generate(self, state):
        contexts = [_ctx(r) for r in state["results"]]
        resp = self.llm.generate(build_rag_prompt(state["question"], contexts), system=GEN_SYSTEM,
                                 json_mode=True, max_tokens=self.cfg.gen_max_tokens)
        answer, statement, ok = parse_generation(resp.text)
        return {"raw_output": resp.text, "answer": answer, "statement": statement, "parse_ok": ok,
                "history": self._h(state, {"step": "generate", "round": state["round"], "answer": answer,
                                           "statement": statement, "parse_ok": ok})}

    def n_extract_claims(self, state):
        claims, fallback = claims_from_generation(self.extractor, state["question"], state["answer"],
                                                  state.get("statement", ""))
        na = normalize_answer(state["answer"])
        in_claims = bool(na) and na in normalize_answer(" ".join(claims))
        return {"claims": claims, "claim_fallback": fallback, "answer_in_claims": in_claims,
                "history": self._h(state, {"step": "extract_claims", "claims": claims, "fallback": fallback,
                                           "answer_in_claims": in_claims})}

    def n_verify_claims(self, state):
        results = state["results"]
        cr = self.cv.verify(state["claims"], [_ctx(r) for r in results], [r.chunk.chunk_id for r in results],
                            titles=[r.chunk.title for r in results])
        dumped = [c.model_dump() for c in cr]
        n_bad = sum(c["verdict"] != "SUPPORTED" for c in dumped)
        return {"claim_results": dumped, "claim_verified": True,
                "claim_risk": (n_bad / len(dumped)) if dumped else 0.0,
                "history": self._h(state, {"step": "verify_claims", "round": state["round"],
                                           "revisions": state.get("revisions", 0),
                                           "verdicts": [c["verdict"] for c in dumped],
                                           "entailment": [round(c["entailment"], 3) for c in dumped]})}

    def n_critic(self, state):
        action = self.critic.decide(state["claim_results"], state["round"], state.get("revisions", 0))
        return {"critic_action": action,
                "history": self._h(state, {"step": "critic", "action": action, "round": state["round"],
                                           "revisions": state.get("revisions", 0),
                                           "verdicts": [c["verdict"] for c in state["claim_results"]]})}

    def n_revise(self, state):
        ans, stm, ok = self.reviser.revise(state["question"], [_ctx(r) for r in state["results"]],
                                           state["answer"], state.get("statement", ""), state["claim_results"])
        return {"answer": ans, "statement": stm, "parse_ok": ok, "revisions": state.get("revisions", 0) + 1,
                "history": self._h(state, {"step": "revise", "round": state["round"],
                                           "previous_answer": state["answer"], "answer": ans,
                                           "statement": stm, "parse_ok": ok})}

    def n_finalize(self, state):
        action = state.get("critic_action")
        abstained = is_abstain(state.get("answer", "")) or action == "ABSTAIN"
        final = ABSTAIN if abstained else state["answer"]
        return {"final_answer": final, "abstained": abstained, "flagged_unverified": action == "ACCEPT_FLAGGED",
                "history": self._h(state, {"step": "finalize", "final_answer": final, "abstained": abstained})}

    # ---- routers
    def r_after_assess(self, state):
        if state["round"] == 0 and (self.cfg.always_on or state.get("risk_level") == "HIGH"):
            return "re_retrieve"
        return "generate"

    def r_after_generate(self, state):
        if not self.cfg.claim_verification or is_abstain(state.get("answer", "")):
            return "finalize"
        if self.cfg.adaptive and not self.cfg.always_on and state.get("risk_level") == "LOW":
            return "finalize"
        return "extract_claims"

    def r_after_verify(self, state):
        ok = all(c["verdict"] == "SUPPORTED" for c in state["claim_results"])
        return "finalize" if ok else "critic"

    def r_after_critic(self, state):
        return {"REVISE": "revise", "RE_RETRIEVE": "re_retrieve"}.get(state["critic_action"], "finalize")

    def r_after_revise(self, state):
        return "finalize" if is_abstain(state.get("answer", "")) else "extract_claims"

    def _build(self):
        g = StateGraph(PipelineState)
        for name, fn in (("retrieve", self.n_retrieve), ("assess", self.n_assess), ("re_retrieve", self.n_re_retrieve),
                         ("generate", self.n_generate), ("extract_claims", self.n_extract_claims),
                         ("verify_claims", self.n_verify_claims), ("critic", self.n_critic),
                         ("revise", self.n_revise), ("finalize", self.n_finalize)):
            g.add_node(name, fn)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "assess")
        g.add_conditional_edges("assess", self.r_after_assess, {"re_retrieve": "re_retrieve", "generate": "generate"})
        g.add_edge("re_retrieve", "generate")
        g.add_conditional_edges("generate", self.r_after_generate,
                                {"extract_claims": "extract_claims", "finalize": "finalize"})
        g.add_edge("extract_claims", "verify_claims")
        g.add_conditional_edges("verify_claims", self.r_after_verify, {"critic": "critic", "finalize": "finalize"})
        g.add_conditional_edges("critic", self.r_after_critic,
                                {"revise": "revise", "re_retrieve": "re_retrieve", "finalize": "finalize"})
        g.add_conditional_edges("revise", self.r_after_revise,
                                {"extract_claims": "extract_claims", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()
