"""Embed ingested papers into an index.

Usage:
  python scripts/build_index.py                                   # all papers, default model -> data/index/
  python scripts/build_index.py --model data/models/ft --index-dir data/index-ft --papers test
--papers test|train restricts the index to one side of eval/split.json (for quick experiments).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import INDEX_DIR, PAPERS_DIR, REPO_ROOT  # noqa: E402
from app.index import embeddings  # noqa: E402
from app.index.store import build_index  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=embeddings.DEFAULT_MODEL, help="Hugging Face id or local model directory")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--papers", choices=["all", "test", "train"], default="all")
    args = parser.parse_args()
    if args.index_dir.resolve() == INDEX_DIR.resolve() and (args.model != embeddings.DEFAULT_MODEL or args.papers != "all"):
        sys.exit(f"refusing to overwrite the app's index {INDEX_DIR} with a partial or non-default build; pass --index-dir")
    only = None
    if args.papers != "all":
        only = set(json.loads((REPO_ROOT / "eval" / "split.json").read_text(encoding="utf-8"))[args.papers])
    embeddings.use_model(args.model)
    started = time.monotonic()
    counts = build_index(PAPERS_DIR, args.index_dir, only)
    print(f"indexed {counts['text_chunks']} text chunks + {counts['figures']} figures "
          f"({counts['papers_embedded']} papers embedded, {counts['papers_cached']} from cache) "
          f"with {args.model} in {time.monotonic() - started:.1f}s -> {args.index_dir}")
