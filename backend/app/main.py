"""HTTP API for the cosmology RAG system.

Run: uvicorn app.main:app --port 8000   (from backend/)
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.answer import PROVIDERS, get_provider
from app.answer.base import Answer, Source, build_sources
from app.config import INDEX_DIR, PAPERS_DIR
from app.index.store import Hit, Index
from app.ingest.pipeline import paper_dir_name

PROVIDER_KEYS = {"claude": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_FIGURE_FILE_RE = re.compile(r"^fig\d+_\d+\.png$")

_state: dict = {}
_providers: dict = {}
_providers_lock = threading.Lock()


def _load_index() -> None:
    _state["index"] = Index(INDEX_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_index()
    _state["index"].search("warm up the embedding model", 1, 1)
    yield


app = FastAPI(title="Cosmology RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    provider: Literal["claude", "deepseek", "both"] = "claude"
    k_text: int = Field(default=6, ge=1, le=12)
    k_figures: int = Field(default=3, ge=0, le=6)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k_text: int = Field(default=6, ge=1, le=12)
    k_figures: int = Field(default=3, ge=0, le=6)


def _provider(name: str):
    with _providers_lock:
        if name not in _providers:
            try:
                _providers[name] = get_provider(name)
            except RuntimeError as exc:  # missing API key
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _providers[name]


def _source_json(source: Source, hit: Hit) -> dict:
    return {
        "id": source.source_id,
        "kind": source.kind,
        "arxiv_id": source.arxiv_id,
        "arxiv_url": f"https://arxiv.org/abs/{source.arxiv_id}",
        "paper_title": source.paper_title,
        "section": source.section,
        "text": source.text,
        "score": round(hit.score, 4),
        "images": [
            f"/api/figures/{paper_dir_name(source.arxiv_id)}/{path.name}" for path in source.image_paths
        ],
    }


def _answer_json(answer: Answer) -> dict:
    return {
        "provider": answer.provider,
        "model": answer.model,
        "text": answer.text,
        "citations": [{"source_id": c.source_id, "cited_text": c.cited_text} for c in answer.citations],
        "unknown_markers": answer.unknown_markers,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "latency_s": round(answer.latency_s, 2),
        "stop_reason": answer.stop_reason,
    }


def _retrieve(question: str, k_text: int, k_figures: int) -> tuple[list[Source], list[dict]]:
    text_hits, figure_hits = _state["index"].search(question, k_text, k_figures)
    sources = build_sources(text_hits, figure_hits)
    return sources, [_source_json(s, h) for s, h in zip(sources, text_hits + figure_hits)]


def _run_provider(name: str, question: str, sources: list[Source]) -> dict:
    provider = _provider(name)
    try:
        return _answer_json(provider.answer(question, sources))
    except anthropic.APIStatusError as exc:
        return {"provider": name, "error": f"Claude API error {exc.status_code}: {exc.message}"}
    except httpx.HTTPStatusError as exc:
        return {"provider": name, "error": f"DeepSeek API error {exc.response.status_code}: {exc.response.text[:300]}"}
    except (anthropic.APIConnectionError, httpx.TransportError) as exc:
        return {"provider": name, "error": f"Could not reach the {name} API: {exc}"}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict:
    records = _state["index"].records
    return {
        "papers": len({r.arxiv_id for r in records}),
        "text_chunks": sum(r.kind == "text" for r in records),
        "figures": sum(r.kind == "figure" for r in records),
        "providers": {name: bool(os.getenv(PROVIDER_KEYS[name])) for name in PROVIDERS},
    }


@app.get("/api/papers")
def papers() -> list[dict]:
    out = []
    for paper_json in sorted(PAPERS_DIR.glob("*/paper.json"), reverse=True):
        paper = json.loads(paper_json.read_text(encoding="utf-8"))
        meta = paper["meta"]
        out.append({
            "arxiv_id": meta["arxiv_id"],
            "title": meta["title"],
            "authors": meta["authors"],
            "published": meta["published"],
            "abstract": meta["abstract"],
            "figures": len(paper["figures"]),
            "source_type": paper["source_type"],
        })
    return out


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    _, sources = _retrieve(request.question, request.k_text, request.k_figures)
    return {"sources": sources}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    sources, sources_json = _retrieve(request.question, request.k_text, request.k_figures)
    names = list(PROVIDERS) if request.provider == "both" else [request.provider]
    for name in names:
        _provider(name)  # fail fast with a 400 if a key is missing, before spending on the other
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        answers = list(pool.map(lambda n: _run_provider(n, request.question, sources), names))
    return {"question": request.question, "sources": sources_json, "answers": answers}


@app.post("/api/reload-index")
def reload_index() -> dict:
    _load_index()
    return stats()


@app.get("/api/figures/{paper_dir}/{filename}")
def figure(paper_dir: str, filename: str) -> FileResponse:
    if not _SAFE_ID_RE.match(paper_dir) or not _FIGURE_FILE_RE.match(filename):
        raise HTTPException(status_code=404)
    path = PAPERS_DIR / paper_dir / "figures" / filename
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/png")
