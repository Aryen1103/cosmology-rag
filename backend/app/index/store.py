"""Build and query the local retrieval index (text chunks + figures).

Brute-force cosine search over numpy arrays: a few hundred papers is tens of
thousands of vectors, well within what a matrix-vector product handles in
milliseconds, and it avoids another compiled dependency (faiss) on this machine.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.index.embeddings import DIM, embed_passages, embed_query
from app.index.embeddings import MODEL_NAME as EMBEDDING_MODEL
from app.index.lexical import BM25
from app.index.rerank import Reranker

CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 25
MAX_FIGURE_CONTEXT_PARAGRAPHS = 2
_BOILERPLATE_RE = re.compile(
    r"acknowledg|data availability|author contribution|software|funding|"
    r"conflicts? of interest|competing interest|code availability|author list|contributors",
    re.I,
)


@dataclass
class Record:
    kind: str  # "text" | "figure"
    arxiv_id: str
    paper_title: str
    section: str
    text: str
    figure_id: str | None = None
    label: str | None = None
    image_files: list[str] | None = None


@dataclass
class Hit:
    record: Record
    score: float


def chunk_section(text: str) -> list[str]:
    """Split a section into ~CHUNK_WORDS-word chunks, preferring paragraph boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in text.split("\n\n"):
        words = paragraph.split()
        if current and len(current) + len(words) > CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP_WORDS:]
        while len(words) > CHUNK_WORDS:
            room = CHUNK_WORDS - len(current)
            current += words[:room]
            words = words[room:]
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP_WORDS:]
        current += words
    if len(current) >= MIN_CHUNK_WORDS or (current and not chunks):
        chunks.append(" ".join(current))
    return chunks


def _figure_text(fig: dict) -> str:
    context = fig["citing_paragraphs"][:MAX_FIGURE_CONTEXT_PARAGRAPHS]
    context = [re.sub(r"\[\[REF:[^\]]+\]\]", "this figure", p) for p in context]
    return fig["caption"] + ("\n\nDiscussed in the text: " + "\n\n".join(context) if context else "")


def is_boilerplate(section_path: list[str]) -> bool:
    # These carry no science but still match queries through the shared paper title.
    return bool(_BOILERPLATE_RE.search(section_path[-1])) if section_path else False


def records_for_paper(paper: dict) -> list[Record]:
    meta = paper["meta"]
    records = [
        Record("text", meta["arxiv_id"], meta["title"], "Abstract", meta["abstract"])
    ]
    for section in paper["sections"]:
        if is_boilerplate(section["path"]):
            continue
        path = " > ".join(section["path"])
        for chunk in chunk_section(section["text"]):
            records.append(Record("text", meta["arxiv_id"], meta["title"], path, chunk))
    for fig in paper["figures"]:
        records.append(
            Record(
                kind="figure",
                arxiv_id=meta["arxiv_id"],
                paper_title=meta["title"],
                section="Figure",
                text=_figure_text(fig),
                figure_id=fig["figure_id"],
                label=fig["label"],
                image_files=[img["file"] for img in fig["images"] if img["file"]],
            )
        )
    return records


def _embedding_input(record: Record) -> str:
    # Title and section give short chunks the context that pure passage text lacks.
    return f"{record.paper_title} | {record.section}\n{record.text}"


def _paper_embeddings(paper_dir: Path, records: list[Record]) -> tuple[np.ndarray, bool]:
    """Per-paper embedding cache, keyed on the exact embedding inputs.

    Embedding runs ~0.3s/chunk on CPU, so re-embedding the whole corpus on every
    rebuild doesn't scale; only new papers or changed chunking pay the cost.
    """
    inputs = [_embedding_input(r) for r in records]
    key = hashlib.sha256(f"{EMBEDDING_MODEL}\n".encode() + "\x00".join(inputs).encode()).hexdigest()
    cache_npy, cache_key = paper_dir / "embeddings.npy", paper_dir / "embeddings.key"
    if cache_npy.exists() and cache_key.exists() and cache_key.read_text() == key:
        return np.load(cache_npy), True
    embeddings = embed_passages(inputs)
    np.save(cache_npy, embeddings)
    cache_key.write_text(key)
    return embeddings, False


def build_index(papers_dir: Path, index_dir: Path) -> dict[str, int]:
    records: list[Record] = []
    parts: list[np.ndarray] = []
    embedded = 0
    for paper_json in sorted(papers_dir.glob("*/paper.json")):
        paper_records = records_for_paper(json.loads(paper_json.read_text(encoding="utf-8")))
        embeddings, cached = _paper_embeddings(paper_json.parent, paper_records)
        records += paper_records
        parts.append(embeddings)
        embedded += not cached

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.concatenate(parts) if parts else np.zeros((0, DIM), dtype=np.float32)
    np.save(index_dir / "embeddings.npy", embeddings)
    with open(index_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return {
        "text_chunks": sum(r.kind == "text" for r in records),
        "figures": sum(r.kind == "figure" for r in records),
        "papers_embedded": embedded,
        "papers_cached": len(parts) - embedded,
    }


METHODS = ("dense", "bm25", "hybrid")
RRF_K = 60  # standard reciprocal-rank-fusion constant
FUSION_POOL = 100  # candidates taken from each ranking before fusing
RERANK_DEPTH = {"text": 30, "figure": 10}  # candidates the cross-encoder reorders


def rrf_order(rankings: list[np.ndarray], k: int = RRF_K) -> tuple[np.ndarray, np.ndarray]:
    """Fuse ranked lists of row ids by reciprocal rank: sum of 1/(k + rank) over lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking.tolist(), start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    rows = sorted(fused, key=lambda r: -fused[r])
    return np.array(rows, dtype=int), np.array([fused[r] for r in rows], dtype=np.float32)


class Index:
    def __init__(self, index_dir: Path, method: str = "dense", reranker: Reranker | None = None) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "records.jsonl", encoding="utf-8") as f:
            self.records = [Record(**json.loads(line)) for line in f]
        kinds = np.array([r.kind for r in self.records])
        self._rows = {"text": np.flatnonzero(kinds == "text"), "figure": np.flatnonzero(kinds == "figure")}
        self.method = method
        self.reranker = reranker
        # BM25 sees the same title | section + text as the embedding model.
        self._bm25 = BM25([_embedding_input(r) for r in self.records]) if method != "dense" else None

    def _ranked(self, query: str, kind: str, k: int, dense: np.ndarray | None,
                lexical: np.ndarray | None) -> list[Hit]:
        rows = self._rows[kind]
        if k <= 0 or rows.size == 0:
            return []
        depth = max(k, RERANK_DEPTH[kind]) if self.reranker else k
        if self.method == "hybrid":
            by_dense = rows[np.argsort(-dense[rows])[:FUSION_POOL]]
            by_lexical = rows[np.argsort(-lexical[rows])[:FUSION_POOL]]
            order, scores = rrf_order([by_dense, by_lexical])
            order, scores = order[:depth], scores[:depth]
        else:
            raw = dense if self.method == "dense" else lexical
            order = rows[np.argsort(-raw[rows])[:depth]]
            scores = raw[order]
        if self.reranker:
            n = min(RERANK_DEPTH[kind], len(order))
            head = self.reranker.scores(query, [_embedding_input(self.records[i]) for i in order[:n]])
            resorted = np.argsort(-head)
            order = np.concatenate([order[:n][resorted], order[n:]])
            scores = np.concatenate([head[resorted], scores[n:]])
        return [Hit(self.records[i], float(s)) for i, s in zip(order[:k], scores[:k])]

    def search(self, query: str, k_text: int = 6, k_figures: int = 3) -> tuple[list[Hit], list[Hit]]:
        dense = self.embeddings @ embed_query(query) if self.method != "bm25" else None
        lexical = self._bm25.scores(query) if self._bm25 else None
        return (self._ranked(query, "text", k_text, dense, lexical),
                self._ranked(query, "figure", k_figures, dense, lexical))


def open_index(index_dir: Path | None = None) -> Index:
    """The index as the app runs it, with RETRIEVAL_METHOD and RERANKER from the environment."""
    from app.config import INDEX_DIR, RERANKER, RETRIEVAL_METHOD

    return Index(index_dir or INDEX_DIR, method=RETRIEVAL_METHOD,
                 reranker=Reranker(RERANKER) if RERANKER else None)
