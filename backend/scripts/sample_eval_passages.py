"""Split papers into train/test and sample passages + figures to write eval questions from.

The question writer only sees what this script picks, so the eval isn't biased
toward passages that happen to be easy to retrieve.

Usage:
  python scripts/sample_eval_passages.py --out sampled.md
Writes eval/split.json (repo root) and a Markdown dump of the sampled records.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import INDEX_DIR, REPO_ROOT  # noqa: E402
from app.eval.retrieval import SPLIT_SEED, split_papers  # noqa: E402
from app.index.store import Record  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
N_TEST = 50
TEXT_PER_PAPER = 2
FIGURES_PER_PAPER = 1
MIN_WORDS = 80


def sample(records: list[Record], arxiv_id: str) -> list[tuple[int, Record]]:
    rng = random.Random(f"{SPLIT_SEED}:{arxiv_id}")
    texts = [(i, r) for i, r in records if r.kind == "text" and r.section != "Abstract"
             and len(r.text.split()) >= MIN_WORDS]
    figures = [(i, r) for i, r in records if r.kind == "figure"]
    with_images = [(i, r) for i, r in figures if r.image_files]
    rng.shuffle(texts)
    picked: list[tuple[int, Record]] = []
    sections: set[str] = set()
    for i, r in texts:  # prefer different sections, fall back to any
        if len(picked) < TEXT_PER_PAPER and r.section not in sections:
            picked.append((i, r))
            sections.add(r.section)
    for i, r in texts:
        if len(picked) < TEXT_PER_PAPER and (i, r) not in picked:
            picked.append((i, r))
    pool = with_images or figures
    return picked + rng.sample(pool, min(FIGURES_PER_PAPER, len(pool)))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(INDEX_DIR / "records.jsonl", encoding="utf-8") as f:
        records = [Record(**json.loads(line)) for line in f]
    by_paper: dict[str, list[tuple[int, Record]]] = {}
    for i, r in enumerate(records):
        by_paper.setdefault(r.arxiv_id, []).append((i, r))

    train, test = split_papers(list(by_paper), N_TEST)
    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / "split.json").write_text(
        json.dumps({"seed": SPLIT_SEED, "test": test, "train": train}, indent=2) + "\n", encoding="utf-8")

    lines = []
    for arxiv_id in test:
        title = by_paper[arxiv_id][0][1].paper_title
        lines.append(f"# {arxiv_id} - {title}\n")
        for i, r in sample(by_paper[arxiv_id], arxiv_id):
            what = f"figure {r.figure_id}" if r.kind == "figure" else f"text, section: {r.section}"
            lines.append(f"## row {i} ({what})\n\n{r.text}\n")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"split: {len(train)} train / {len(test)} test -> {EVAL_DIR / 'split.json'}")
    print(f"sampled records -> {args.out}")


if __name__ == "__main__":
    main()
