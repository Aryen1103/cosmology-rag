"""Retrieval evaluation: does the passage or figure that answers a question get retrieved?

Gold labels are tied to the paper, not to index rows, so they survive changes to
chunking or embedding:
- a text question names its paper and a short evidence quote; any text record
  from that paper whose text contains the quote is relevant;
- a figure question names its paper and figure_id.

Nothing here imports torch: the runner passes in the ranked hits.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.index.store import Hit, Record

TEXT_KS = (1, 3, 6, 20)  # 6 = passages the answer step actually sends
FIGURE_KS = (1, 3, 10)  # 3 = figures the answer step actually sends
MAX_TEXT_K = max(TEXT_KS)
MAX_FIGURE_K = max(FIGURE_KS)
SPLIT_SEED = "cosmology-rag-eval-v1"


@dataclass
class Question:
    id: str
    question: str
    kind: str  # "text" | "figure"
    arxiv_id: str
    evidence: str | None = None  # text questions: quote that must appear in the retrieved chunk
    figure_id: str | None = None  # figure questions
    tags: list[str] = field(default_factory=list)


def split_papers(arxiv_ids: list[str], n_test: int, seed: str = SPLIT_SEED) -> tuple[list[str], list[str]]:
    """Deterministic paper-level split: (train, test).

    Hash order, so each added paper displaces at most one test paper. The split
    used for the eval is frozen in eval/split.json; read that rather than recomputing.
    """
    ranked = sorted(set(arxiv_ids), key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).hexdigest())
    return sorted(ranked[n_test:]), sorted(ranked[:n_test])


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(question: Question, record: Record) -> bool:
    if record.arxiv_id != question.arxiv_id:
        return False
    if question.kind == "figure":
        return record.kind == "figure" and record.figure_id == question.figure_id
    return record.kind == "text" and normalise(question.evidence or "") in normalise(record.text)


def first_relevant_rank(question: Question, hits: list[Hit]) -> int | None:
    """1-based rank of the first relevant hit, or None if none was retrieved."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(question, hit.record):
            return rank
    return None


def load_questions(path: Path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def check_questions(questions: list[Question], records: list[Record], test_ids: set[str]) -> list[str]:
    """Problems that would make a question unscorable or misleading. Empty list = all good."""
    problems: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.arxiv_id not in test_ids:
            problems.append(f"{q.id}: paper {q.arxiv_id} is not in the test split")
        if q.kind not in ("text", "figure"):
            problems.append(f"{q.id}: unknown kind {q.kind!r}")
            continue
        if q.kind == "text" and not (q.evidence and len(q.evidence.split()) >= 3):
            problems.append(f"{q.id}: text question needs an evidence quote of 3+ words")
            continue
        if q.kind == "figure" and not q.figure_id:
            problems.append(f"{q.id}: figure question needs figure_id")
            continue
        relevant = [r for r in records if is_relevant(q, r)]
        if not relevant:
            problems.append(f"{q.id}: gold not found in the index")
        if q.kind == "text":
            needle = normalise(q.evidence)
            elsewhere = {r.arxiv_id for r in records if r.kind == "text" and r.arxiv_id != q.arxiv_id
                         and needle in normalise(r.text)}
            if elsewhere:
                problems.append(f"{q.id}: evidence also appears in {sorted(elsewhere)}")
    return problems


def summarise(ranks: list[int | None], ks: tuple[int, ...], max_k: int) -> dict[str, float]:
    """hit@k for each k plus MRR@max_k, as fractions."""
    if not ranks:
        return {}
    out = {f"hit@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in ks}
    out[f"mrr@{max_k}"] = float(np.mean([1 / r if r is not None and r <= max_k else 0.0 for r in ranks]))
    return out


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of per-question values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
