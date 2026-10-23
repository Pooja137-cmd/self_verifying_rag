from src.agents.claim_extractor import ClaimExtractor, extract_claims_rule
from src.agents.claim_verifier import ClaimVerifier
from src.llm.base import LLMClient, LLMResponse


class FakeLLM(LLMClient):
    model = "fake"

    def __init__(self, text):
        super().__init__()
        self.text = text

    def _generate(self, prompt, system, json_mode, max_tokens):
        return LLMResponse(self.text, 10, 10, 0.0, "fake")


class RuleNLI:
    """Entails when the premise contains 'paris' and the claim mentions 'paris'; contradicts on 'london'."""

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


def test_rule_extraction():
    assert extract_claims_rule("Answer: Paris is the capital. It is large.") == ["Paris is the capital.", "It is large."]
    assert extract_claims_rule("insufficient information") == []
    assert extract_claims_rule("Rome") == []


def test_llm_atomic_and_fallback():
    ok = ClaimExtractor(FakeLLM('{"claims": ["A is B.", "C is D."]}'), atomic=True)
    assert ok.extract("A is B and C is D now.") == ["A is B.", "C is D."]
    bad = ClaimExtractor(FakeLLM("not json"), atomic=True)
    assert bad.extract("A is B and C is D now.") == ["A is B and C is D now."]
    capped = ClaimExtractor(FakeLLM('{"claims": ["a b c", "d e f", "g h i"]}'), atomic=True, max_claims=2)
    assert len(capped.extract("Some long answer sentence here.")) == 2


def test_verify_verdicts():
    cv = ClaimVerifier(RuleNLI())
    texts = ["The Eiffel Tower is in Paris.", "Photosynthesis makes sugar."]
    ids = ["d0::c0", "d1::c0"]
    r = cv.verify(["The tower is in Paris.", "The tower is in London.", "Cats are mammals here."], texts, ids)
    assert [x.verdict for x in r] == ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    assert r[0].supporting_chunk_ids and r[1].supporting_chunk_ids == []


def test_verify_no_evidence_and_no_claims():
    cv = ClaimVerifier(RuleNLI())
    assert cv.verify([], ["x"], ["i"]) == []
    assert cv.verify(["Some claim here."], [], [])[0].verdict == "INSUFFICIENT_EVIDENCE"
