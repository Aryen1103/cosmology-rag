"""Retrieval eval: where does the passage or figure that answers each question rank?

No API calls; the only cost is embedding the questions locally.

Usage:
  python scripts/eval_retrieval.py --check                  # validate questions against the index
  python scripts/eval_retrieval.py --name baseline          # score the current index (dense search)
  python scripts/eval_retrieval.py --name hybrid --method hybrid [--rerank minilm|bge-base]
  python scripts/eval_retrieval.py --compare baseline other # paired difference between two runs
Questions live in eval/retrieval_questions.jsonl, runs in eval/runs/<name>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.config import INDEX_DIR, REPO_ROOT  # noqa: E402
from app.eval.retrieval import (  # noqa: E402
    FIGURE_KS,
    MAX_FIGURE_K,
    MAX_TEXT_K,
    TEXT_KS,
    bootstrap_ci,
    check_questions,
    first_relevant_rank,
    load_questions,
    summarise,
)
from app.index.embeddings import MODEL_NAME  # noqa: E402
from app.index.rerank import RERANKERS, Reranker  # noqa: E402
from app.index.store import METHODS, Hit, Index, Record  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
QUESTIONS = EVAL_DIR / "retrieval_questions.jsonl"
RUNS_DIR = EVAL_DIR / "runs"
# Text: the 6 passages the answer step receives. Figures: hit@3 (what it receives) was 100% at
# baseline, so it can't show improvements; rank 1 still has headroom.
HEADLINE = {"text": "hit@6", "figure": "hit@1"}


def _brief(hit: Hit) -> dict:
    r = hit.record
    return {"arxiv_id": r.arxiv_id, "where": r.figure_id or r.section, "score": round(hit.score, 4)}


def _paper_rank(arxiv_id: str, hits: list[Hit]) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if h.record.arxiv_id == arxiv_id), None)


def run(name: str, method: str, rerank: str | None) -> None:
    questions = load_questions(QUESTIONS)
    index = Index(INDEX_DIR, method=method, reranker=Reranker(rerank) if rerank else None)
    index.search("warm up the models", 1, 1)  # keep model loading out of the timings
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in questions:
        started = time.perf_counter()
        text_hits, figure_hits = index.search(q.question, k_text=MAX_TEXT_K, k_figures=MAX_FIGURE_K)
        elapsed = time.perf_counter() - started
        hits = figure_hits if q.kind == "figure" else text_hits
        rows.append({
            "id": q.id, "kind": q.kind, "tags": q.tags, "arxiv_id": q.arxiv_id,
            "rank": first_relevant_rank(q, hits),
            "paper_rank": _paper_rank(q.arxiv_id, hits),
            "search_s": round(elapsed, 4),
            "top": [_brief(h) for h in hits[:6]],
        })
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"name": name, "method": method, "reranker": RERANKERS[rerank] if rerank else None,
               "embedding_model": MODEL_NAME, "index_records": len(index.records),
               "questions": len(rows), "median_search_s": float(np.median([r["search_s"] for r in rows])),
               "metrics": {}}
    for kind, ks, max_k in (("text", TEXT_KS, MAX_TEXT_K), ("figure", FIGURE_KS, MAX_FIGURE_K)):
        subset = [r for r in rows if r["kind"] == kind]
        ranks = [r["rank"] for r in subset]
        m = summarise(ranks, ks, max_k)
        head = HEADLINE[kind]
        k = int(head.split("@")[1])
        m[f"{head}_ci95"] = bootstrap_ci([float(r is not None and r <= k) for r in ranks])
        m[f"paper_{head}"] = float(np.mean([r["paper_rank"] is not None and r["paper_rank"] <= k
                                             for r in subset]))
        m["n"] = len(subset)
        summary["metrics"][kind] = m
        for tag in sorted({t for r in subset for t in r["tags"]} - {kind}):
            tagged = [r["rank"] for r in subset if tag in r["tags"]]
            summary["metrics"][f"{kind}:{tag}"] = {**summarise(tagged, ks, max_k), "n": len(tagged)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    print(f"\nper-question results -> {out_dir / 'results.jsonl'}")


def _print_summary(summary: dict) -> None:
    print(f"{summary['name']}: {summary['questions']} questions, method={summary['method']}, "
          f"reranker={summary['reranker']}, {summary['embedding_model']}, "
          f"{summary['index_records']} index records, median search {summary['median_search_s'] * 1000:.0f} ms")
    for group, m in summary["metrics"].items():
        cols = "  ".join(f"{k}={v:.3f}" for k, v in m.items()
                         if isinstance(v, float) and not k.startswith("paper_"))
        line = f"  {group:<18} n={m['n']:<4} {cols}"
        kind = group.split(":")[0]
        head = HEADLINE[kind]
        if f"{head}_ci95" in m:
            lo, hi = m[f"{head}_ci95"]
            line += f"  ({head} 95% CI {lo:.3f}-{hi:.3f}; right paper {m[f'paper_{head}']:.3f})"
        print(line)


def compare(a: str, b: str) -> None:
    """Paired difference b - a on the headline metric and MRR, with a paired bootstrap CI."""
    def load(name: str) -> dict[str, dict]:
        with open(RUNS_DIR / name / "results.jsonl", encoding="utf-8") as f:
            return {row["id"]: row for row in map(json.loads, f)}

    ra, rb = load(a), load(b)
    shared = sorted(ra.keys() & rb.keys())
    if len(shared) != len(ra) or len(shared) != len(rb):
        print(f"warning: runs differ in questions; comparing the {len(shared)} they share")
    print(f"{b} vs {a} (paired, {len(shared)} questions)")
    for kind, max_k in (("text", MAX_TEXT_K), ("figure", MAX_FIGURE_K)):
        ids = [i for i in shared if ra[i]["kind"] == kind]
        k = int(HEADLINE[kind].split("@")[1])

        def hit(r: dict) -> float:
            return float(r["rank"] is not None and r["rank"] <= k)

        def rr(r: dict) -> float:
            return 1 / r["rank"] if r["rank"] is not None and r["rank"] <= max_k else 0.0

        for label, fn in ((HEADLINE[kind], hit), (f"mrr@{max_k}", rr)):
            diffs = [fn(rb[i]) - fn(ra[i]) for i in ids]
            lo, hi = bootstrap_ci(diffs)
            print(f"  {kind:<6} {label:<7} {np.mean([fn(ra[i]) for i in ids]):.3f} -> "
                  f"{np.mean([fn(rb[i]) for i in ids]):.3f}  diff {np.mean(diffs):+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
        gained = sum(hit(rb[i]) > hit(ra[i]) for i in ids)
        lost = sum(hit(rb[i]) < hit(ra[i]) for i in ids)
        print(f"  {kind:<6} questions gained {gained}, lost {lost}")


def check() -> None:
    questions = load_questions(QUESTIONS)
    with open(INDEX_DIR / "records.jsonl", encoding="utf-8") as f:
        records = [Record(**json.loads(line)) for line in f]
    split = json.loads((EVAL_DIR / "split.json").read_text(encoding="utf-8"))
    problems = check_questions(questions, records, set(split["test"]))
    kinds = {k: sum(q.kind == k for q in questions) for k in ("text", "figure")}
    print(f"{len(questions)} questions ({kinds['text']} text, {kinds['figure']} figure) "
          f"from {len({q.arxiv_id for q in questions})} papers")
    for p in problems:
        print("  PROBLEM", p)
    if problems:
        sys.exit(1)
    print("all gold passages and figures found in the index; no evidence quote matches another paper")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--name")
    group.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument("--rerank", choices=sorted(RERANKERS))
    args = parser.parse_args()
    if args.check:
        check()
    elif args.compare:
        compare(*args.compare)
    else:
        run(args.name, args.method, args.rerank)


if __name__ == "__main__":
    main()
