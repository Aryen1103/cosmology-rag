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


class Index:
    def __init__(self, index_dir: Path) -> None:
        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "records.jsonl", encoding="utf-8") as f:
            self.records = [Record(**json.loads(line)) for line in f]
        kinds = np.array([r.kind for r in self.records])
        self._text_rows = np.flatnonzero(kinds == "text")
        self._figure_rows = np.flatnonzero(kinds == "figure")

    def _top(self, scores: np.ndarray, rows: np.ndarray, k: int) -> list[Hit]:
        if k <= 0 or rows.size == 0:
            return []
        best = rows[np.argsort(-scores[rows])[:k]]
        return [Hit(self.records[i], float(scores[i])) for i in best]

    def search(self, query: str, k_text: int = 6, k_figures: int = 3) -> tuple[list[Hit], list[Hit]]:
        scores = self.embeddings @ embed_query(query)
        return self._top(scores, self._text_rows, k_text), self._top(scores, self._figure_rows, k_figures)
