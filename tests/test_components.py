import json

from src.agents.planner import QueryPlanner
from src.agents.retriever_agent import RetrievalAgent
from src.agents.risk_controller import RiskConfig, RiskController
from src.llm.base import LLMClient, LLMResponse
from src.llm.cache import CachedLLM
from src.llm.prompts import is_abstain, parse_generation
from tests.test_vector_store import build


class ScriptLLM(LLMClient):
    model = "fake"

    def __init__(self, reply="x"):
        super().__init__()
        self.reply = reply

    def _generate(self, prompt, system, json_mode, max_tokens):
        return LLMResponse(self.reply, 100, 20, 0.5, "fake")


def test_parse_generation():
    assert parse_generation('{"answer": "Paris", "statement": "It is in Paris."}') == ("Paris", "It is in Paris.", True)
    assert parse_generation('noise {"answer": "Rome", "statement": "x"} tail')[0] == "Rome"
    a, s, ok = parse_generation("Just text\nmore")
    assert (a, s, ok) == ("Just text", "", False)
    assert parse_generation("")[2] is False
    assert is_abstain("Insufficient information.")


def test_cache_hit_and_usage(tmp_path):
    inner = ScriptLLM("hello")
    c = CachedLLM(inner, tmp_path / "c.jsonl")
    r1 = c.generate("p", system="s", json_mode=True, max_tokens=10)
    r2 = c.generate("p", system="s", json_mode=True, max_tokens=10)
    assert r1.text == r2.text == "hello"
    assert inner.usage.calls == 1 and c.usage.calls == 2      # real calls vs standalone-equivalent calls
    assert c.hits == 1 and c.misses == 1
    assert c.usage.latency_s == 1.0                           # original latency preserved on the hit
    c2 = CachedLLM(ScriptLLM("other"), tmp_path / "c.jsonl")  # persisted to disk
    assert c2.generate("p", system="s", json_mode=True, max_tokens=10).text == "hello" and c2.hits == 1
    assert c.generate("different", system="s", json_mode=True, max_tokens=10).text == "hello"
    assert c.misses == 2


def test_planner_and_re_retrieve_merge():
    store = build()
    agent = RetrievalAgent(store, QueryPlanner(), top_k=1, extra=3)
    prior = agent.retrieve("Where is the Eiffel Tower?")
    assert prior[0].chunk.doc_id == "eiffel"
    merged, q2 = agent.re_retrieve("Where is the Eiffel Tower?", prior)
    ids = [r.chunk.chunk_id for r in merged]
    assert ids[0] == prior[0].chunk.chunk_id and len(ids) == len(set(ids)) > 1
    assert [r.rank for r in merged] == list(range(1, len(merged) + 1))
    assert "Eiffel Tower" in q2 and q2.startswith("Where is")


def test_risk_controller():
    rc = RiskController(RiskConfig(low_threshold=0.3, high_threshold=0.7))
    assert rc.score({"retrieval_risk": 1.0, "coverage_risk": 0.0}) == 0.5
    assert rc.level(0.1) == "LOW" and rc.level(0.5) == "MEDIUM" and rc.level(0.7) == "HIGH"
    assert rc.score({}) == 0.0
    w = RiskController(RiskConfig(w_retrieval=1.0, w_evidence=0.0))
    assert w.score({"retrieval_risk": 0.8, "coverage_risk": 0.0}) == 0.8


def test_risk_config_load_ignores_extra(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"low_threshold": 0.1, "high_threshold": 0.9, "calibration": {"auc": 0.7}}))
    c = RiskConfig.load(p)
    assert c.low_threshold == 0.1 and c.high_threshold == 0.9
