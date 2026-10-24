from src.agents.claim_extractor import ClaimExtractor
from src.agents.claim_verifier import ClaimVerifier
from src.agents.evidence_verifier import EvidenceVerifier
from src.agents.planner import QueryPlanner
from src.agents.retriever_agent import RetrievalAgent
from src.agents.risk_controller import RiskConfig, RiskController
from src.graph.state import make_condition
from src.graph.workflow import Pipeline
from tests.test_components import ScriptLLM
from tests.test_pipeline import KeywordNLI, Q
from tests.test_vector_store import build

LONDON = '{"answer": "London", "statement": "The Eiffel Tower is in London."}'
BERLIN = '{"answer": "Berlin", "statement": "The Eiffel Tower is near Berlin."}'


def run(reply):
    cfg = make_condition("C", final_policy="abstain_if_contradicted")
    p = Pipeline(cfg, RetrievalAgent(build(), QueryPlanner(), top_k=2, extra=3), EvidenceVerifier(),
                 RiskController(RiskConfig()), ScriptLLM(reply), ClaimExtractor(), ClaimVerifier(KeywordNLI()))
    return p.run(Q)


def test_contradicted_claim_abstains():
    s = run(LONDON)
    assert s["abstained"] and s["final_answer"] == "insufficient information"


def test_neutral_claim_is_accepted_but_flagged():
    s = run(BERLIN)
    assert not s["abstained"] and s["final_answer"] == "Berlin" and s["flagged_unverified"]
