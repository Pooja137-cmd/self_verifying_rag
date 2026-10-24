from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig

ENT = {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01}
CON = {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90}
SPURIOUS = {"entailment": 0.02, "neutral": 0.05, "contradiction": 0.93}   # NLI "contradiction" on unrelated text


class KW:
    def score(self, premises, hypotheses):
        out = []
        for p, h in zip(premises, hypotheses):
            p, h = p.lower(), h.lower()
            if "alpha" in h and "beta" in h and "alpha" in p and "beta" in p:
                out.append(ENT)
            elif "tower" in h and "london" in h and "paris" in p and "tower" in p:
                out.append(CON)
            else:
                out.append(SPURIOUS)
        return out


def cv():
    return ClaimVerifier(KW(), ClaimVerifierConfig(premise_mode="windows"))


def test_two_hop_claim_supported_by_cross_chunk_pair():
    texts = ["Alpha Town: The alpha thing is here. Filler sentence number one.",
             "Beta Town: The beta thing is there. Filler sentence number two."]
    r = cv().verify(["The alpha thing relates to the beta thing."], texts, ["c0", "c1"], titles=["Alpha Town", "Beta Town"])[0]
    assert r.verdict == "SUPPORTED" and set(r.supporting_chunk_ids) == {"c0", "c1"}


def test_spurious_contradiction_on_unrelated_text_is_ignored():
    texts = ["Alpha Town: The alpha thing is here. Filler sentence number one."]
    r = cv().verify(["Quantum zebras compute mathematics."], texts, ["c0"], titles=["Alpha Town"])[0]
    assert r.verdict == "INSUFFICIENT_EVIDENCE" and r.contradiction == 0.0


def test_relevant_contradiction_is_kept():
    texts = ["Eiffel Tower: The Eiffel Tower is in Paris France. It is very tall."]
    r = cv().verify(["The Eiffel Tower is in London."], texts, ["c0"], titles=["Eiffel Tower"])[0]
    assert r.verdict == "CONTRADICTED"


def test_old_modes_still_work_and_no_evidence():
    old = ClaimVerifier(KW(), ClaimVerifierConfig(premise_mode="joint"))
    assert old.verify([], ["x"], ["i"]) == []
    assert old.verify(["Some claim here."], [], [])[0].verdict == "INSUFFICIENT_EVIDENCE"
