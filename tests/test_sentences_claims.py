from src.agents.claim_extractor import extract_claims_rule
from src.retrieval.sentences import split_sentences_safe


def test_abbreviations_and_initials_do_not_split():
    assert len(split_sentences_safe("Guardians of the Galaxy Vol. 2 is the movie directed by James Gunn.")) == 1
    assert len(split_sentences_safe("Robert Z. Leonard was born in 1889. Sinclair Hill was born in 1894.")) == 2
    assert split_sentences_safe("J. R. R. Tolkien wrote it. Lewis read it.") == ["J. R. R. Tolkien wrote it.", "Lewis read it."]
    assert split_sentences_safe("Dr. Smith went home. He slept.") == ["Dr. Smith went home.", "He slept."]
    assert split_sentences_safe("") == []


def test_claims_keep_full_names():
    assert extract_claims_rule("Robert Z. Leonard is younger than Sinclair Hill") == ["Robert Z. Leonard is younger than Sinclair Hill"]
    assert extract_claims_rule("Guardians of the Galaxy Vol. 2 is the movie that portrayed Ego.") == \
        ["Guardians of the Galaxy Vol. 2 is the movie that portrayed Ego."]
