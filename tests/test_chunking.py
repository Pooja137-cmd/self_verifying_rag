import pytest

from src.retrieval.chunking import Document, chunk_corpus, chunk_document, clean_text, split_sentences


def make_doc(n_sent=30, words=10, doc_id="d1"):
    sents = [" ".join(f"w{i}x{j}" for j in range(words - 1)) + "." for i in range(n_sent)]
    return Document(doc_id=doc_id, title="T", text=" ".join(sents))


def test_clean_text():
    assert clean_text("a \n\n b\t c\x00d") == "a b c d"


def test_split_sentences():
    assert split_sentences("Hello world. This is it! Really? Yes.") == ["Hello world.", "This is it!", "Really?", "Yes."]


def test_empty_document():
    assert chunk_document(Document(doc_id="e", text="   ")) == []


def test_short_doc_single_chunk():
    chunks = chunk_document(Document(doc_id="s", text="One short sentence."), 200, 40)
    assert len(chunks) == 1 and chunks[0].chunk_id == "s::c0"


def test_size_bound_and_ids():
    chunks = chunk_document(make_doc(), chunk_size=50, overlap=20)
    assert len(chunks) > 1
    assert all(c.n_words <= 50 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_shares_sentences():
    chunks = chunk_document(make_doc(), chunk_size=50, overlap=20)
    first_sentence_of_second = chunks[1].text.split(".")[0] + "."
    assert first_sentence_of_second in chunks[0].text


def test_no_overlap():
    chunks = chunk_document(make_doc(), chunk_size=50, overlap=0)
    assert chunks[1].text.split(".")[0] + "." not in chunks[0].text


def test_all_content_covered():
    doc = make_doc()
    joined = " ".join(c.text for c in chunk_document(doc, 50, 20))
    for i in range(30):
        assert f"w{i}x0" in joined


def test_long_sentence_is_hard_split():
    doc = Document(doc_id="l", text=" ".join(f"w{i}" for i in range(500)) + ".")
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert len(chunks) >= 5 and all(c.n_words <= 100 for c in chunks)


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_document(make_doc(), chunk_size=10, overlap=10)


def test_deterministic():
    a = chunk_corpus([make_doc()], 50, 20)
    b = chunk_corpus([make_doc()], 50, 20)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_embed_text_includes_title():
    c = chunk_document(Document(doc_id="x", title="Paris", text="It is a city."), 50, 10)[0]
    assert c.embed_text.startswith("Paris.")