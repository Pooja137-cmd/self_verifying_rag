from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig


class NeedsBoth:
    """Entails only when the premise contains BOTH 'alpha' and 'beta' (a two-hop claim)."""

    def score(self, premises, hypotheses):
        out = []
        for p in premises:
            ok = "alpha" in p.lower() and "beta" in p.lower()
            out.append({"entailment": 0.9 if ok else 0.05, "neutral": 0.05 if ok else 0.9, "contradiction": 0.05})
        return out


TEXTS = ["alpha fact here", "unrelated words", "beta fact here"]
IDS = ["c0", "c1", "c2"]
CLAIM = ["The alpha thing relates to the beta thing."]


def test_pairs_mode_finds_two_hop_support():
    joint = ClaimVerifier(NeedsBoth(), ClaimVerifierConfig(premise_mode="joint", joint_max_words=3))
    assert joint.verify(CLAIM, TEXTS, IDS)[0].verdict == "INSUFFICIENT_EVIDENCE"
    pairs = ClaimVerifier(NeedsBoth(), ClaimVerifierConfig(premise_mode="pairs"))
    r = pairs.verify(CLAIM, TEXTS, IDS)[0]
    assert r.verdict == "SUPPORTED" and set(r.supporting_chunk_ids) == {"c0", "c2"}


def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        ClaimVerifier(NeedsBoth(), ClaimVerifierConfig(premise_mode="nope")).verify(CLAIM, TEXTS, IDS)
