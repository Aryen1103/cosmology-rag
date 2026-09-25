"""Fetch recent arXiv papers and ingest them into data/papers/.

Usage: python scripts/ingest.py --query "cat:astro-ph.CO" --max 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.arxiv_client import ArxivClient  # noqa: E402
from app.ingest.pipeline import fetch_and_ingest, paper_dir_name  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="cat:astro-ph.CO")
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()

    papers_dir = DATA_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    client = ArxivClient()

    report = []
    for meta in client.search(args.query, max_results=args.max):
        if (papers_dir / paper_dir_name(meta.arxiv_id) / "paper.json").exists():
            print(f"skip   {meta.arxiv_id} (already ingested)")
            continue
        try:
            record = fetch_and_ingest(client, meta, papers_dir)
        except Exception as exc:
            print(f"FAIL   {meta.arxiv_id}: {type(exc).__name__}: {exc}")
            report.append({"arxiv_id": meta.arxiv_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        statuses = Counter(
            img["status"].split(":")[0] for fig in record["figures"] for img in fig["images"]
        )
        entry = {
            "arxiv_id": meta.arxiv_id,
            "source_type": record["source_type"],
            "sections": len(record["sections"]),
            "chars": sum(len(s["text"]) for s in record["sections"]),
            "figures": len(record["figures"]),
            "image_statuses": dict(statuses),
        }
        report.append(entry)
        print(f"ok     {meta.arxiv_id} [{entry['source_type']}] sections={entry['sections']} "
              f"figures={entry['figures']} images={dict(statuses)}")

    (DATA_DIR / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
