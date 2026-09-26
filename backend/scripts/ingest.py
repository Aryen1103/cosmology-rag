"""Fetch recent arXiv papers and ingest them into data/papers/.

Usage: python scripts/ingest.py --query "cat:astro-ph.CO" --max 20
       python scripts/ingest.py --ids 2609.19969 2609.27282   # specific papers, any category
       python scripts/ingest.py --retry-failed                  # papers that failed in the last report
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
    parser.add_argument("--ids", nargs="+", metavar="ARXIV_ID", help="ingest these papers instead of searching")
    parser.add_argument("--retry-failed", action="store_true", help="re-try the failures in ingest_report.json")
    parser.add_argument("--force", action="store_true", help="re-ingest papers that are already ingested")
    args = parser.parse_args()

    papers_dir = DATA_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    report_path = DATA_DIR / "ingest_report.json"
    client = ArxivClient()

    ids = list(args.ids or [])
    if args.retry_failed:
        ids += [e["arxiv_id"] for e in json.loads(report_path.read_text(encoding="utf-8")) if "error" in e]
    if ids:
        metas = []
        for arxiv_id in dict.fromkeys(ids):
            found = client.search(f"id:{arxiv_id}", max_results=1)
            if not found:
                print(f"FAIL   {arxiv_id}: not found on arXiv")
                continue
            metas.append(found[0])
    else:
        metas = client.search(args.query, max_results=args.max)

    report = []
    for meta in metas:
        if not args.force and (papers_dir / paper_dir_name(meta.arxiv_id) / "paper.json").exists():
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

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
