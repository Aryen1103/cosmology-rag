"""Training pairs for fine-tuning the embedding model, built from the papers themselves.

No model or API writes them. Two kinds, both "inverse cloze" style: a sentence is
taken out of an indexed record and used as the query, and the record with that
sentence removed is the positive passage.

- text: a sentence from a text chunk -> the rest of that chunk;
- figure: the first sentence of a caption -> the figure record without it
  (rest of the caption plus the paragraphs that cite the figure).

Only papers in the train side of eval/split.json are used, so nothing from the
eval's test papers is trained on. No torch import here (CI runs without it).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.index.store import Record, _embedding_input, records_for_paper

SEED = 13
MIN_QUERY_WORDS = 8
MAX_QUERY_WORDS = 40
MIN_PASSAGE_WORDS = 40
MAX_DOLLARS = 4  # skip sentences that are mostly inline math
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass
class Pair:
    kind: str  # "text" | "figure"
    arxiv_id: str
    query: str
    passage: str  # the full embedding input, "title | section\ntext", as the index sees it


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def usable_query(sentence: str) -> bool:
    words = len(sentence.split())
    return (MIN_QUERY_WORDS <= words <= MAX_QUERY_WORDS and sentence.count("$") <= MAX_DOLLARS
            and "[[REF" not in sentence and "[cite:" not in sentence and "\\begin" not in sentence)


def _without(text: str, sentence: str) -> str:
    return re.sub(r"\s+", " ", text.replace(sentence, " ", 1)).strip()


def pair_for_record(record: Record, rng: random.Random) -> Pair | None:
    """At most one pair per record, so no passage dominates a batch."""
    if record.kind == "text":
        if record.section == "Abstract":
            return None
        candidates = [s for s in sentences(record.text) if usable_query(s)]
    else:
        caption = record.text.split("\n\nDiscussed in the text:")[0]
        first = sentences(caption)[:1]
        candidates = [s for s in first if usable_query(s)]
    if not candidates:
        return None
    query = rng.choice(candidates)
    rest = _without(record.text, query)
    if len(rest.split()) < MIN_PASSAGE_WORDS:
        return None
    return Pair(record.kind, record.arxiv_id, query, _embedding_input(replace(record, text=rest)))


def build_pairs(papers_dir: Path, train_ids: set[str], test_ids: set[str], seed: int = SEED) -> list[Pair]:
    """Every usable pair from the train papers, in a seeded shuffled order."""
    if train_ids & test_ids:
        raise ValueError("train and test papers overlap")
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for paper_json in sorted(papers_dir.glob("*/paper.json")):
        paper = json.loads(paper_json.read_text(encoding="utf-8"))
        arxiv_id = paper["meta"]["arxiv_id"]
        if arxiv_id not in train_ids:
            continue
        for record in records_for_paper(paper):
            pair = pair_for_record(record, rng)
            if pair:
                pairs.append(pair)
    rng.shuffle(pairs)
    if {p.arxiv_id for p in pairs} & test_ids:
        raise AssertionError("a test paper leaked into the training pairs")
    return pairs


def write_pairs(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_pairs(path: Path) -> list[Pair]:
    with open(path, encoding="utf-8") as f:
        return [Pair(**json.loads(line)) for line in f if line.strip()]
