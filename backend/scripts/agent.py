"""Ask the tool-calling agent a question, printing each tool call as it happens.

Usage:
  python scripts/agent.py "What H0 does the TRGB paper find, and what age of the universe does that imply?"
  python scripts/agent.py "..." --max-turns 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import CosmologyAgent, ToolStep  # noqa: E402
from app.agent.agent import MAX_TURNS  # noqa: E402
from app.config import INDEX_DIR, PAPERS_DIR  # noqa: E402
from app.index.store import Index  # noqa: E402
from app.ingest.pipeline import paper_summaries  # noqa: E402


def print_step(step: ToolStep) -> None:
    flag = "  ERROR" if step.is_error else ""
    print(f"-> {step.tool}({json.dumps(step.input, ensure_ascii=False)}){flag}\n   {step.output}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # answers contain dashes, ×, Greek; the Windows console default mangles them
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = parser.parse_args()

    agent = CosmologyAgent(max_turns=args.max_turns)
    print("== Tool calls ==")
    run = agent.run(args.question, Index(INDEX_DIR), paper_summaries(PAPERS_DIR), on_step=print_step)

    answer = run.answer
    print(f"\n== {answer.provider} ({answer.model}) - {answer.latency_s:.1f}s, {len(run.steps)} tool calls, "
          f"{answer.input_tokens} in / {answer.output_tokens} out ==")
    print(answer.text)
    print("\nCited sources:")
    cited = {c.source_id for c in answer.citations}
    for source in run.sources:
        if source.source_id in cited:
            print(f"  {source.heading}")
    if answer.unknown_markers:
        print(f"  WARNING - cited sources that don't exist: {answer.unknown_markers}")


if __name__ == "__main__":
    main()
