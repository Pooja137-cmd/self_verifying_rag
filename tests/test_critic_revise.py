from src.agents.claim_extractor import ClaimExtractor
from src.agents.claim_verifier import ClaimVerifier
from src.agents.critic import Critic
from src.agents.evidence_verifier import EvidenceVerifier
from src.agents.planner import QueryPlanner
from src.agents.retriever_agent import RetrievalAgent
from src.agents.risk_controller import RiskConfig, RiskController
from src.graph.state import make_condition
from src.graph.workflow import Pipeline
from src.llm.base import LLMClient, LLMResponse
from tests.test_pipeline import Q, KeywordNLI
from tests.test_vector_store import build

PARIS = '{"answer": "Paris", "statement": "The Eiffel Tower is in Paris."}'
LONDON = '{"answer": "London", "statement": "The Eiffel Tower is in London."}'
BERLIN = '{"answer": "Berlin", "statement": "The Eiffel Tower is near Berlin."}'
ABST = '{"answer": "insufficient information", "statement": ""}'


class SeqLLM(LLMClient):
    model = "fake"

    def __init__(self, replies):
        super().__init__()
        self.replies, self.i = list(replies), 0

    def _generate(self, prompt, system, json_mode, max_tokens):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResponse(r, 100, 20, 0.5, "fake")


def make(replies, **over):
    llm = SeqLLM(replies)
    p = Pipeline(make_condition("C", revise=True, **over), RetrievalAgent(build(), QueryPlanner(), top_k=2, extra=3),
                 EvidenceVerifier(), RiskController(RiskConfig()), llm, ClaimExtractor(), ClaimVerifier(KeywordNLI()))
    return p, llm


def steps(s):
    return [h["step"] for h in s["history"]]


def cr(*v):
    return [{"verdict": x} for x in v]


def test_critic_rules():
    c = Critic(revise=True, max_rounds=1, max_revisions=1, final_policy="abstain")
    assert c.decide(cr("CONTRADICTED"), 0, 0) == "REVISE"
    assert c.decide(cr("INSUFFICIENT_EVIDENCE"), 0, 0) == "RE_RETRIEVE"
    assert c.decide(cr("INSUFFICIENT_EVIDENCE"), 1, 0) == "REVISE"
    assert c.decide(cr("CONTRADICTED"), 1, 1) == "ABSTAIN"
    nr = Critic(revise=False)
    assert nr.decide(cr("CONTRADICTED"), 0, 0) == "RE_RETRIEVE" and nr.decide(cr("CONTRADICTED"), 1, 0) == "ABSTAIN"
    assert Critic(final_policy="accept_flagged").decide(cr("CONTRADICTED"), 1, 0) == "ACCEPT_FLAGGED"
    assert Critic(final_policy="abstain_if_contradicted").decide(cr("INSUFFICIENT_EVIDENCE"), 1, 0) == "ACCEPT_FLAGGED"


def test_revise_fixes_contradicted_answer():
    p, llm = make([LONDON, PARIS])
    s = p.run(Q)
    assert s["final_answer"] == "Paris" and llm.usage.calls == 2 and s["revisions"] == 1
    assert "revise" in steps(s) and "re_retrieve" not in steps(s)


def test_revise_then_re_retrieve_then_success():
    p, llm = make([LONDON, LONDON, PARIS])
    s = p.run(Q)
    assert s["final_answer"] == "Paris" and llm.usage.calls == 3
    assert steps(s).index("revise") < steps(s).index("re_retrieve")


def test_everything_fails_then_abstain():
    p, llm = make([LONDON])
    s = p.run(Q)
    assert s["abstained"] and llm.usage.calls == 3 and s["revisions"] == 1 and s["round"] == 1


def test_revision_can_abstain():
    p, llm = make([LONDON, ABST])
    s = p.run(Q)
    assert s["abstained"] and llm.usage.calls == 2 and steps(s).count("verify_claims") == 1


def test_insufficient_evidence_prefers_re_retrieval():
    p, llm = make([BERLIN, PARIS])
    s = p.run(Q)
    assert s["final_answer"] == "Paris" and llm.usage.calls == 2 and "revise" not in steps(s)
