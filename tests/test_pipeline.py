from src.agents.claim_extractor import ClaimExtractor
from src.agents.claim_verifier import ClaimVerifier
from src.agents.evidence_verifier import EvidenceVerifier
from src.agents.planner import QueryPlanner
from src.agents.retriever_agent import RetrievalAgent
from src.agents.risk_controller import RiskConfig, RiskController
from src.graph.state import make_condition
from src.graph.workflow import Pipeline
from tests.test_components import ScriptLLM
from tests.test_vector_store import build

Q = "Where is the Eiffel Tower located?"
PARIS = '{"answer": "Paris", "statement": "The Eiffel Tower is in Paris."}'
LONDON = '{"answer": "London", "statement": "The Eiffel Tower is in London."}'


class KeywordNLI:
    def score(self, premises, hypotheses):
        out = []
        for p, h in zip(premises, hypotheses):
            p, h = p.lower(), h.lower()
            if "paris" in h and "paris" in p:
                out.append({"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01})
            elif "london" in h and "paris" in p:
                out.append({"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90})
            else:
                out.append({"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05})
        return out


def make(cond, reply, risk=None):
    llm = ScriptLLM(reply)
    p = Pipeline(make_condition(cond), RetrievalAgent(build(), QueryPlanner(), top_k=2, extra=3),
                 EvidenceVerifier(), RiskController(risk or RiskConfig()), llm, ClaimExtractor(),
                 ClaimVerifier(KeywordNLI()))
    return p, llm


def steps(state):
    return [h["step"] for h in state["history"]]


def test_A_vanilla_no_verification():
    p, llm = make("A", LONDON)
    s = p.run(Q)
    assert s["final_answer"] == "London" and llm.usage.calls == 1
    assert not s["claim_verified"] and s["risk_level"] == "NA" and "re_retrieve" not in steps(s)


def test_C_supported_claim_accepted():
    p, llm = make("C", PARIS)
    s = p.run(Q)
    assert s["final_answer"] == "Paris" and llm.usage.calls == 1 and s["claim_verified"]
    assert "critic" not in steps(s)


def test_C_unsupported_claim_re_retrieves_then_abstains():
    p, llm = make("C", LONDON)
    s = p.run(Q)
    assert s["abstained"] and s["final_answer"] == "insufficient information"
    assert llm.usage.calls == 2 and s["round"] == 1
    assert steps(s).count("critic") == 2 and "re_retrieve" in steps(s)


def test_B_high_risk_triggers_re_retrieval_before_generation():
    p, llm = make("B", PARIS, RiskConfig(low_threshold=0.0, high_threshold=0.0))
    s = p.run(Q)
    assert s["round"] == 1 and llm.usage.calls == 1 and s["risk_level"] == "HIGH"
    assert steps(s).index("re_retrieve") < steps(s).index("generate") and not s["claim_verified"]


def test_E_skips_claim_verification_at_low_risk_but_D_does_not():
    low = RiskConfig(low_threshold=1.01, high_threshold=2.0)
    e, llm_e = make("E", LONDON, low)
    se = e.run(Q)
    assert se["risk_level"] == "LOW" and not se["claim_verified"] and se["final_answer"] == "London"
    assert llm_e.usage.calls == 1
    d, llm_d = make("D", LONDON, low)
    sd = d.run(Q)
    assert sd["claim_verified"] and sd["abstained"] and llm_d.usage.calls == 2


def test_E_verifies_when_risk_not_low():
    mid = RiskConfig(low_threshold=0.0, high_threshold=2.0)
    e, llm = make("E", LONDON, mid)
    s = e.run(Q)
    assert s["risk_level"] == "MEDIUM" and s["claim_verified"] and s["abstained"]


def test_F_always_re_retrieves():
    p, llm = make("F", PARIS, RiskConfig(low_threshold=1.01, high_threshold=2.0))
    s = p.run(Q)
    assert s["round"] == 1 and llm.usage.calls == 1 and s["claim_verified"] and s["final_answer"] == "Paris"


def test_model_abstention_passes_through():
    p, llm = make("D", '{"answer": "insufficient information", "statement": ""}')
    s = p.run(Q)
    assert s["abstained"] and not s["claim_verified"] and llm.usage.calls == 1


def test_parse_failure_uses_fallback_claim():
    p, llm = make("C", "Paris")
    s = p.run(Q)
    assert s["claim_fallback"] and s["claim_verified"]
