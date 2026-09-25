from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict

import httpx

API_URL = "https://export.arxiv.org/api/query"
EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    primary_category: str
    categories: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


class ArxivClient:
    """arXiv API client that enforces arXiv's requested ~3s gap between requests."""

    def __init__(self, min_interval: float = 3.0) -> None:
        self._min_interval = min_interval
        self._last_request = 0.0
        self._http = httpx.Client(
            timeout=120,
            follow_redirects=True,
            headers={"User-Agent": "cosmology-rag/0.1 (research prototype)"},
        )

    def _get(self, url: str, **kwargs) -> httpx.Response:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self._http.get(url, **kwargs)
        finally:
            self._last_request = time.monotonic()
        response.raise_for_status()
        return response

    def search(self, query: str, max_results: int = 20, start: int = 0) -> list[PaperMeta]:
        response = self._get(
            API_URL,
            params={
                "search_query": query,
                "start": start,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        return parse_feed(response.text)

    def download_source(self, arxiv_id: str) -> bytes:
        return self._get(EPRINT_URL.format(arxiv_id=arxiv_id)).content


def parse_feed(xml_text: str) -> list[PaperMeta]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = _clean(entry.findtext(f"{ATOM}id"))
        arxiv_id = entry_id.rsplit("/abs/", 1)[-1]
        primary = entry.find(f"{ARXIV}primary_category")
        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=_clean(entry.findtext(f"{ATOM}title")),
                authors=[_clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
                abstract=_clean(entry.findtext(f"{ATOM}summary")),
                published=_clean(entry.findtext(f"{ATOM}published")),
                primary_category=primary.get("term", "") if primary is not None else "",
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
            )
        )
    return papers
