"""Ask a question over the indexed corpus.

Usage:
  python scripts/ask.py "What value of sigma_8 do they find?" --provider claude
  python scripts/ask.py "..." --provider both
  python scripts/ask.py "..." --retrieval-only     # no API call, no key needed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer import PROVIDERS, get_provider  # noqa: E402
from app.answer.base import build_sources  # noqa: E402
from app.config import INDEX_DIR  # noqa: E402
from app.index.store import Index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--provider", choices=[*PROVIDERS, "both"], default="claude")
    parser.add_argument("--k-text", type=int, default=6)
    parser.add_argument("--k-figures", type=int, default=3)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    text_hits, figure_hits = Index(INDEX_DIR).search(args.question, args.k_text, args.k_figures)
    sources = build_sources(text_hits, figure_hits)

    print("== Retrieved sources ==")
    for source, hit in zip(sources, text_hits + figure_hits):
        images = f" [{len(source.image_paths)} image(s)]" if source.kind == "figure" else ""
        print(f"{hit.score:.3f}  {source.heading}{images}")
    if args.retrieval_only:
        return

    for name in PROVIDERS if args.provider == "both" else [args.provider]:
        answer = get_provider(name).answer(args.question, sources)
        print(f"\n== {answer.provider} ({answer.model}) - {answer.latency_s:.1f}s, "
              f"{answer.input_tokens} in / {answer.output_tokens} out ==")
        print(answer.text)
        print("\nCitations:")
        for c in answer.citations:
            quote = f': "{c.cited_text.strip()[:120]}"' if c.cited_text else ""
            print(f"  [{c.source_id}]{quote}")
        if answer.unknown_markers:
            print(f"  WARNING - cited sources that don't exist: {answer.unknown_markers}")


if __name__ == "__main__":
    main()
