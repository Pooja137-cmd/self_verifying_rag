import pytest

from src.agents.evidence_verifier import EvidenceVerifier, question_coverage
from src.retrieval.chunking import Chunk
from src.retrieval.vector_store import SearchResult


def sr(i, text, score):
    ch = Chunk(chunk_id=f"d{i}::c0", doc_id=f"d{i}", chunk_index=0, text=text, n_words=len(text.split()))
    return SearchResult(ch, score, i + 1)


class FakeNLI:
    def __init__(self, ent=0.1, con=0.1):
        self.ent, self.con = ent, con

    def score(self, premises, hypotheses):
        return [{"entailment": self.ent, "neutral": 1 - self.ent - self.con, "contradiction": self.con}
                for _ in premises]


def test_coverage():
    assert question_coverage("Where is the Eiffel Tower located?", ["The Eiffel Tower is located in Paris."]) == 1.0
    assert question_coverage("Where is the Eiffel Tower located?", ["Photosynthesis is a process."]) < 0.5
    assert question_coverage("the of and", ["anything"]) == 1.0


def test_retrieval_risk_decreases_with_score():
    v = EvidenceVerifier()
    hi = v.assess("q words here", [sr(0, "a b c", 0.85), sr(1, "d e f", 0.6)])
    lo = v.assess("q words here", [sr(0, "a b c", 0.40), sr(1, "d e f", 0.35)])
    assert hi.retrieval_risk < lo.retrieval_risk
    assert 0.0 <= hi.retrieval_risk <= 1.0 and lo.retrieval_risk == pytest.approx(1.0)


def test_empty_results():
    a = EvidenceVerifier().assess("some question", [])
    assert a.top1_score == 0.0 and a.retrieval_risk == 1.0 and a.chunks == []


def test_contradiction_signal():
    results = [sr(0, "alpha beta", 0.8), sr(1, "gamma delta", 0.7), sr(2, "eps zeta", 0.6)]
    a = EvidenceVerifier(nli=FakeNLI(con=0.9)).assess("alpha question", results)
    assert a.max_contradiction == pytest.approx(0.9) and a.n_nli_pairs == 3
    b = EvidenceVerifier().assess("alpha question", results)
    assert b.max_contradiction == 0.0 and b.n_nli_pairs == 0
