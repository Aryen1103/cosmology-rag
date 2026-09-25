"""Embed all ingested papers into data/index/. Usage: python scripts/build_index.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import INDEX_DIR, PAPERS_DIR  # noqa: E402
from app.index.store import build_index  # noqa: E402

if __name__ == "__main__":
    started = time.monotonic()
    counts = build_index(PAPERS_DIR, INDEX_DIR)
    print(f"indexed {counts['text_chunks']} text chunks + {counts['figures']} figures "
          f"({counts['papers_embedded']} papers embedded, {counts['papers_cached']} from cache) "
          f"in {time.monotonic() - started:.1f}s -> {INDEX_DIR}")
