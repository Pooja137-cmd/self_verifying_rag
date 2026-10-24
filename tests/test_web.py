from src.agents.planner import QueryPlanner
from src.agents.web_retriever import WebRetrievalAgent
from src.retrieval.web import CachingClient, extract_text, is_safe_url
from tests.test_vector_store import FakeEmbedder

HTML = """<html><head><title>Eiffel</title></head><body><nav>menu menu menu menu</nav>
<script>var x=1;</script><p>The Eiffel Tower is a wrought-iron tower in Paris France</p>
<p>It was completed in 1889 for the World's Fair</p><footer>copyright text here now</footer></body></html>"""

PAGES = {
    "https://a.example/eiffel": ("Eiffel Tower", "The Eiffel Tower is a landmark in Paris France. " * 8),
    "https://b.example/photo": ("Photosynthesis", "Photosynthesis converts light energy into chemical energy. " * 8),
    "https://c.example/wall": ("Great Wall", "The Great Wall of China is thousands of kilometres long. " * 8),
}


class FakeClient:
    def __init__(self):
        self.searches, self.fetches = 0, 0

    def search(self, query, n=6):
        self.searches += 1
        return [{"title": t, "url": u} for u, (t, _) in PAGES.items()]

    def fetch(self, hit):
        self.fetches += 1
        t, text = PAGES[hit["url"]]
        return {"url": hit["url"], "title": t, "text": text}


def test_extract_text():
    title, text = extract_text(HTML)
    assert title == "Eiffel"
    assert "wrought-iron" in text and "menu" not in text and "var x" not in text and "copyright" not in text
    assert text.count(".") >= 2


def test_is_safe_url():
    assert is_safe_url("https://example.com/a")
    assert not any(is_safe_url(u) for u in ("http://localhost/x", "http://192.168.1.5/", "http://127.0.0.1/",
                                            "ftp://x.com/a", "file:///etc/passwd"))


def test_caching_client():
    inner = FakeClient()
    c = CachingClient(inner)
    c.search("q", 4)
    c.search("q", 4)
    hit = {"url": "https://a.example/eiffel", "title": "x"}
    c.fetch(hit)
    c.fetch(hit)
    assert inner.searches == 1 and inner.fetches == 1


def test_web_agent_retrieve_and_re_retrieve():
    agent = WebRetrievalAgent(FakeEmbedder(), FakeClient(), QueryPlanner(), top_k=2, extra=3, n_pages=2,
                              chunk_size=40, overlap=10)
    prior = agent.retrieve("Where is the Eiffel Tower?")
    assert prior and prior[0].chunk.metadata["url"] == "https://a.example/eiffel"
    assert len(agent.docs) == 2
    merged, q2 = agent.re_retrieve("Where is the Eiffel Tower?", prior)
    assert len(agent.docs) == 3 and len(merged) > len(prior)
    ids = [r.chunk.chunk_id for r in merged]
    assert len(ids) == len(set(ids)) and ids[0] == prior[0].chunk.chunk_id


def test_web_agent_no_results():
    class Empty:
        def search(self, q, n=6):
            return []

        def fetch(self, h):
            return None

    agent = WebRetrievalAgent(FakeEmbedder(), Empty(), QueryPlanner())
    assert agent.retrieve("anything") == []
    assert agent.re_retrieve("anything", []) == ([], "anything")
