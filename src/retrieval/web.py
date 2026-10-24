"""Live web retrieval (free, no API key): search -> fetch -> clean text. Source URLs are preserved.
Pages are untrusted input (they can contain prompt injection); this module only extracts text."""
from __future__ import annotations

import ipaddress
import re
import time
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; research-prototype/0.1; self-verifying-rag)"
MAX_BYTES = 1_500_000
MAX_WORDS = 2500
MIN_WORDS = 40
WIKI_API = "https://en.wikipedia.org/w/api.php"


def is_safe_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return True


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]):
        tag.decompose()
    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if len(t.split()) >= 4:
            parts.append(t if t[-1] in ".!?" else t + ".")
    return title, " ".join(" ".join(parts).split()[:MAX_WORDS])


class WebClient:
    def __init__(self, backend: str = "ddg", timeout: int = 10, pause: float = 0.3) -> None:
        if backend not in ("ddg", "wikipedia"):
            raise ValueError("backend must be 'ddg' or 'wikipedia'")
        self.backend, self.timeout, self.pause = backend, timeout, pause
        self.last_error = ""

    # ---- search
    def search(self, query: str, n: int = 6) -> list[dict]:
        self.last_error = ""
        try:
            hits = self._search_ddg(query, n) if self.backend == "ddg" else self._search_wikipedia(query, n)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        return [h for h in hits if h.get("url") and is_safe_url(h["url"])]

    def _search_ddg(self, query: str, n: int) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as d:
            res = list(d.text(query, max_results=n))
        return [{"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
                 "snippet": r.get("body", "")} for r in res]

    def _search_wikipedia(self, query: str, n: int) -> list[dict]:
        r = requests.get(WIKI_API, params={"action": "query", "list": "search", "srsearch": query,
                                           "srlimit": n, "format": "json"},
                         headers={"User-Agent": UA}, timeout=self.timeout)
        r.raise_for_status()
        hits = []
        for item in r.json()["query"]["search"]:
            title = item["title"]
            hits.append({"title": title, "wiki_title": title,
                         "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                         "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", ""))})
        return hits

    # ---- fetch
    def fetch(self, hit: dict) -> dict | None:
        try:
            time.sleep(self.pause)
            return self._fetch_wikipedia(hit) if hit.get("wiki_title") else self._fetch_html(hit)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _fetch_wikipedia(self, hit: dict) -> dict | None:
        r = requests.get(WIKI_API, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                           "titles": hit["wiki_title"], "redirects": 1, "format": "json"},
                         headers={"User-Agent": UA}, timeout=self.timeout)
        r.raise_for_status()
        page = next(iter(r.json()["query"]["pages"].values()))
        text = re.sub(r"=+\s*([^=\n]+?)\s*=+", r"\1.", page.get("extract", ""))
        words = text.split()[:MAX_WORDS]
        if len(words) < MIN_WORDS:
            return None
        return {"url": hit["url"], "title": page.get("title", hit["title"]), "text": " ".join(words)}

    def _fetch_html(self, hit: dict) -> dict | None:
        url = hit["url"]
        if not is_safe_url(url):
            return None
        with requests.get(url, headers={"User-Agent": UA}, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            if not is_safe_url(r.url) or "html" not in r.headers.get("Content-Type", "").lower():
                return None
            buf = b""
            for part in r.iter_content(65536):
                buf += part
                if len(buf) > MAX_BYTES:
                    break
            html = buf.decode(r.encoding or "utf-8", errors="ignore")
        title, text = extract_text(html)
        if len(text.split()) < MIN_WORDS:
            return None
        return {"url": r.url, "title": title or hit.get("title", ""), "text": text}


class CachingClient:
    """Memoises search and fetch so several conditions are compared on identical web evidence."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self._s: dict = {}
        self._f: dict = {}

    @property
    def last_error(self) -> str:
        return getattr(self.inner, "last_error", "")

    def search(self, query: str, n: int = 6) -> list[dict]:
        key = (query, n)
        if key not in self._s:
            self._s[key] = self.inner.search(query, n)
        return self._s[key]

    def fetch(self, hit: dict) -> dict | None:
        key = hit["url"]
        if key not in self._f:
            self._f[key] = self.inner.fetch(hit)
        return self._f[key]
