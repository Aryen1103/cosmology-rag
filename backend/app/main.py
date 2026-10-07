"""HTTP API for the cosmology RAG system.

Run: uvicorn app.main:app --port 8000   (from backend/)

If the frontend has been built (frontend/dist, or FRONTEND_DIST), it is served at /, so a
deployment is a single container.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import AgentRun, CosmologyAgent
from app.answer import PROVIDERS, get_provider
from app.answer.base import Answer, Source, build_sources
from app.config import INDEX_DIR, PAPERS_DIR, REPO_ROOT
from app.index.store import Index
from app.ingest.pipeline import paper_dir_name, paper_summaries

PROVIDER_KEYS = {"claude": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_FIGURE_FILE_RE = re.compile(r"^fig\d+_\d+\.png$")

_state: dict = {}
_providers: dict = {}
_providers_lock = threading.Lock()


class AskLimiter:
    """Caps model calls on a public deployment, where every question spends the owner's API credit.

    Limits are per client IP per hour and across all clients per UTC day; 0 disables a limit.
    In-memory, so it resets on restart and is per-replica - enough for a single small container.
    """

    def __init__(self, per_ip_per_hour: int, per_day: int) -> None:
        self.per_ip_per_hour = per_ip_per_hour
        self.per_day = per_day
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day = dt.date.min
        self._day_count = 0

    def check(self, client: str, calls: int) -> None:
        now = time.time()
        today = dt.datetime.now(dt.timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day, self._day_count = today, 0
            recent = self._recent[client]
            while recent and recent[0] <= now - 3600:
                recent.popleft()
            if self.per_ip_per_hour and len(recent) >= self.per_ip_per_hour:
                raise HTTPException(429, f"Limit of {self.per_ip_per_hour} questions per hour reached. Try again later, or use Search only.")
            if self.per_day and self._day_count + calls > self.per_day:
                raise HTTPException(429, "This demo's daily question budget is used up. Search only still works.")
            recent.append(now)
            self._day_count += calls


_limiter = AskLimiter(
    per_ip_per_hour=int(os.getenv("ASK_LIMIT_PER_IP_PER_HOUR", "0")),
    per_day=int(os.getenv("ASK_LIMIT_PER_DAY", "0")),
)


def _client_ip(request: Request) -> str:
    # Behind a cloud load balancer the real client is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _load_index() -> None:
    _state["index"] = Index(INDEX_DIR)
    _state.pop("papers", None)


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


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k_text: int = Field(default=6, ge=1, le=12)
    k_figures: int = Field(default=3, ge=0, le=6)


def _cached(name: str, factory):
    with _providers_lock:
        if name not in _providers:
            try:
                _providers[name] = factory()
            except RuntimeError as exc:  # missing API key
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _providers[name]


def _provider(name: str):
    return _cached(name, lambda: get_provider(name))


def _agent() -> CosmologyAgent:
    return _cached("agent", CosmologyAgent)


def _source_json(source: Source, score: float) -> dict:
    return {
        "id": source.source_id,
        "kind": source.kind,
        "arxiv_id": source.arxiv_id,
        "arxiv_url": f"https://arxiv.org/abs/{source.arxiv_id}",
        "paper_title": source.paper_title,
        "section": source.section,
        "text": source.text,
        "score": round(score, 4),
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
    return sources, [_source_json(s, h.score) for s, h in zip(sources, text_hits + figure_hits)]


def _guarded(name: str, call) -> dict:
    """Run a model call, turning API failures into an error answer the UI can show."""
    try:
        return call()
    except anthropic.APIStatusError as exc:
        return {"provider": name, "error": f"Claude API error {exc.status_code}: {exc.message}"}
    except httpx.HTTPStatusError as exc:
        return {"provider": name, "error": f"DeepSeek API error {exc.response.status_code}: {exc.response.text[:300]}"}
    except (anthropic.APIConnectionError, httpx.TransportError) as exc:
        return {"provider": name, "error": f"Could not reach the {name} API: {exc}"}


def _run_provider(name: str, question: str, sources: list[Source]) -> dict:
    provider = _provider(name)
    return _guarded(name, lambda: _answer_json(provider.answer(question, sources)))


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
    if "papers" not in _state:  # reading every paper.json per request is slow; cached until reload
        _state["papers"] = paper_summaries(PAPERS_DIR)
    return _state["papers"]


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    _, sources = _retrieve(request.question, request.k_text, request.k_figures)
    return {"sources": sources}


@app.post("/api/ask")
def ask(request: AskRequest, http_request: Request) -> dict:
    names = list(PROVIDERS) if request.provider == "both" else [request.provider]
    for name in names:
        _provider(name)  # fail fast with a 400 if a key is missing, before spending on the other
    _limiter.check(_client_ip(http_request), calls=len(names))
    sources, sources_json = _retrieve(request.question, request.k_text, request.k_figures)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        answers = list(pool.map(lambda n: _run_provider(n, request.question, sources), names))
    return {"question": request.question, "sources": sources_json, "answers": answers}


def _agent_json(question: str, run: AgentRun) -> dict:
    return {
        "question": question,
        "sources": [_source_json(s, score) for s, score in zip(run.sources, run.scores)],
        "answers": [_answer_json(run.answer)],
        "steps": [
            {"tool": s.tool, "input": s.input, "output": s.output, "is_error": s.is_error} for s in run.steps
        ],
    }


@app.post("/api/agent")
def ask_agent(request: AgentRequest, http_request: Request) -> dict:
    """Agent mode: the model runs its own searches, views figures and calls lcdm_calculator."""
    agent = _agent()  # 400 if the Anthropic key is missing
    _limiter.check(_client_ip(http_request), calls=1)  # one question, however many turns it takes
    index, paper_list = _state["index"], papers()
    result = _guarded("agent", lambda: _agent_json(request.question, agent.run(request.question, index, paper_list)))
    if "error" in result:
        return {"question": request.question, "sources": [], "answers": [result], "steps": []}
    return result


@app.post("/api/reload-index")
def reload_index(http_request: Request) -> dict:
    token = os.getenv("ADMIN_TOKEN")
    if token and http_request.headers.get("x-admin-token") != token:
        raise HTTPException(status_code=403)
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


# Last, so every /api route above takes precedence over the static files.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
