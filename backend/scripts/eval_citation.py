"""Claude vs DeepSeek citation eval on the same retrieved sources.

  python scripts/eval_citation.py --name main [--limit 3] [--providers claude deepseek]
  python scripts/eval_citation.py --name main --report

Each question is retrieved once (the app's hybrid search, 6 passages + 3 figures)
and both models answer from those identical sources. DeepSeek goes through the
app's DeepSeekProvider. Claude runs through the Claude Code CLI in headless
mode (`claude -p`, the user's subscription; no API key), with the same system
prompt and sources, in an empty working directory holding only the figure
images, and only the Read tool, so it can view figures and nothing else. Both
therefore cite with [S#]/[F#] markers; Claude's API-native citations are not
used here. Answers are judged by DeepSeek with provider names removed.

Rows are written as they complete; rerunning skips (question, provider) pairs
already answered or judged. Output: eval/citation/runs/<name>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer.base import SYSTEM_PROMPT, build_sources, figure_intro, marker_citations  # noqa: E402
from app.config import REPO_ROOT  # noqa: E402
from app.eval.citation import (  # noqa: E402
    CLAIM_CITATION_PROMPT, JUDGE_PROMPT, TIERS, cited_correctly, load_questions, parse_claim_citations,
    parse_judgment, score_citations,
)
from app.index.store import open_index  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval" / "citation"
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_MAX_TURNS = 6  # up to three figure reads, then the answer
CLAUDE_TIMEOUT_S = 600
PROVIDERS = ("claude", "deepseek")


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _sources_prompt(sources, image_names: dict[str, list[str]] | None = None) -> str:
    passages = "\n\n".join(f"{s.heading}\n{s.text}" for s in sources if s.kind == "text")
    parts = [f"Text sources:\n\n{passages}"]
    for s in sources:
        if s.kind == "figure":
            intro = figure_intro(s)
            if image_names and image_names.get(s.source_id):
                intro += "\nImage file(s) in the current folder: " + ", ".join(image_names[s.source_id])
            parts.append(intro)
    return "\n\n".join(parts)


def ask_claude(question: str, sources) -> dict:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("Claude Code CLI not found (npm install -g @anthropic-ai/claude-code)")
    with tempfile.TemporaryDirectory(prefix="citation_eval_") as tmp:
        work = Path(tmp)
        names: dict[str, list[str]] = {}
        for s in sources:
            for k, path in enumerate(s.image_paths, start=1):
                name = f"{s.source_id}_{k}.png"
                shutil.copyfile(path, work / name)
                names.setdefault(s.source_id, []).append(name)
        prompt = _sources_prompt(sources, names)
        if names:
            prompt += ("\n\nThe figure images are the PNG files named above, in the current folder. "
                       "Use the Read tool to look at any figure the question depends on before answering.")
        prompt += f"\n\nQuestion: {question}"
        (work / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
        cmd = [exe, "-p", "--output-format", "json", "--model", CLAUDE_MODEL,
               "--system-prompt-file", "system_prompt.txt", "--tools", "Read", "--allowedTools", "Read",
               "--max-turns", str(CLAUDE_MAX_TURNS), "--no-session-persistence", "--strict-mcp-config",
               "--disable-slash-commands", "--setting-sources", "project"]
        started = time.monotonic()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              cwd=work, timeout=CLAUDE_TIMEOUT_S)
        latency = time.monotonic() - started
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude -p returned no JSON (exit {proc.returncode}): {proc.stderr[-500:]}") from e
    if data.get("is_error") or data.get("subtype") != "success":
        raise RuntimeError(f"claude -p failed: {data.get('subtype')} {str(data.get('result'))[:300]}")
    models = list(data.get("modelUsage", {}))
    if CLAUDE_MODEL not in models:
        raise RuntimeError(f"served models {models} do not include {CLAUDE_MODEL}")
    usage = data.get("usage", {})
    return {"text": data.get("result", ""), "model": CLAUDE_MODEL, "latency_s": round(latency, 1),
            "turns": data.get("num_turns"), "input_tokens": usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "list_price_usd": data.get("total_cost_usd")}  # an estimate; the subscription is not billed per call


def ask_deepseek(question: str, sources) -> dict:
    from app.answer.deepseek import DeepSeekProvider

    a = DeepSeekProvider().answer(question, sources)
    return {"text": a.text, "model": a.model, "latency_s": round(a.latency_s, 1), "turns": 1,
            "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "list_price_usd": None}


def judge(question: str, sources, answer_text: str, prompt: str = JUDGE_PROMPT, parse=parse_judgment) -> dict:
    """DeepSeek judges claim support, seeing the sources (with images) but not which model answered."""
    import httpx

    from app.answer.base import image_b64
    from app.config import DEEPSEEK_API_URL, DEEPSEEK_MODEL, require_env

    content = [{"type": "text", "text": _sources_prompt(sources)}]
    for s in sources:
        for path in s.image_paths:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64(path)}",
                                                                "detail": "high"}})
    content.append({"type": "text", "text": f"Question: {question}\n\n<answer>\n{answer_text}\n</answer>"})
    r = httpx.post(DEEPSEEK_API_URL, timeout=180,
                   headers={"Authorization": f"Bearer {require_env('DEEPSEEK_API_KEY')}"},
                   # The judge reasons before answering; 4,000 tokens ran out mid-reasoning on long answers.
                   json={"model": DEEPSEEK_MODEL, "max_tokens": 16000, "response_format": {"type": "json_object"},
                         "messages": [{"role": "system", "content": prompt},
                                      {"role": "user", "content": content}]})
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError(f"judge stopped with finish_reason={choice.get('finish_reason')}")
    out = parse(choice["message"]["content"])
    out["judge_model"] = data.get("model", DEEPSEEK_MODEL)
    return out


def run(name: str, providers: list[str], limit: int | None, ids: list[str] | None = None) -> None:
    out = EVAL_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    answers_path, errors_path, judged_path = out / "answers.jsonl", out / "errors.jsonl", out / "judgments.jsonl"
    done = {(r["id"], r["provider"]) for r in _rows(answers_path)}
    judged = {(r["id"], r["provider"]) for r in _rows(judged_path)}
    questions = load_questions(EVAL_DIR / "questions.jsonl")[:limit]
    if ids:
        questions = [q for q in questions if q.id in ids]
    index = open_index()
    for q in questions:
        text_hits, figure_hits = index.search(q.question, 6, 3)
        sources = build_sources(text_hits, figure_hits)
        records = {s.source_id: h.record for s, h in zip(sources, text_hits + figure_hits)}
        for provider in providers:
            if (q.id, provider) not in done:
                try:
                    answer = (ask_claude if provider == "claude" else ask_deepseek)(q.question, sources)
                except Exception as e:  # recorded, not scored; a rerun retries it
                    _append(errors_path, {"id": q.id, "provider": provider, "error": f"{type(e).__name__}: {e}"[:1000]})
                    print(f"{q.id} {provider}: ERROR {e}"[:300])
                    continue
                cited, unknown = marker_citations(answer["text"], sources)
                ids = {c.source_id for c in cited}
                row = {"id": q.id, "tier": q.tier, "provider": provider, **answer,
                       "cited": sorted(ids), "unknown_markers": unknown,
                       "sources": {k: {"arxiv_id": r.arxiv_id, "kind": r.kind, "where": r.figure_id or r.section}
                                   for k, r in records.items()},
                       **score_citations(q, records, ids)}
                _append(answers_path, row)
                done.add((q.id, provider))
                print(f"{q.id} {provider}: cited {sorted(ids)} -> correct={cited_correctly(row)} "
                      f"({answer['latency_s']}s, {answer['input_tokens']} in / {answer['output_tokens']} out)")
            if (q.id, provider) in done and (q.id, provider) not in judged:
                answer_row = next(r for r in _rows(answers_path) if r["id"] == q.id and r["provider"] == provider)
                try:
                    j = judge(q.question, sources, answer_row["text"])
                except Exception as e:
                    _append(errors_path, {"id": q.id, "provider": provider, "stage": "judge",
                                          "error": f"{type(e).__name__}: {e}"[:1000]})
                    continue
                _append(judged_path, {"id": q.id, "provider": provider, **j})
                judged.add((q.id, provider))


def claim_citations(name: str) -> None:
    """Second judge pass, added after the first rubric saturated: is each claim's own citation right?"""
    out = EVAL_DIR / "runs" / name
    path = out / "claim_citations.jsonl"
    done = {(r["id"], r["provider"]) for r in _rows(path)}
    answers = _rows(out / "answers.jsonl")
    questions = {q.id: q for q in load_questions(EVAL_DIR / "questions.jsonl")}
    index = open_index()
    for a in answers:
        if (a["id"], a["provider"]) in done:
            continue
        q = questions[a["id"]]
        sources = build_sources(*index.search(q.question, 6, 3))
        try:
            j = judge(q.question, sources, a["text"], CLAIM_CITATION_PROMPT, parse_claim_citations)
        except Exception as e:
            _append(out / "errors.jsonl", {"id": a["id"], "provider": a["provider"], "stage": "claim_citations",
                                           "error": f"{type(e).__name__}: {e}"[:1000]})
            continue
        _append(path, {"id": a["id"], "provider": a["provider"], **j})
        print(f"{a['id']} {a['provider']}: {j['n_cited_correct']}/{j['n_cited']} cited claims correct, {j['n_uncited']} uncited")


def report(name: str) -> None:
    out = EVAL_DIR / "runs" / name
    answers = {(r["id"], r["provider"]): r for r in _rows(out / "answers.jsonl")}
    judged = {(r["id"], r["provider"]): r for r in _rows(out / "judgments.jsonl")}
    questions = {q.id: q for q in load_questions(EVAL_DIR / "questions.jsonl")}
    summary: dict = {"providers": {}}
    for provider in PROVIDERS:
        rows = [(questions[i], a, judged.get((i, p))) for (i, p), a in answers.items() if p == provider]
        if not rows:
            continue

        def rate(values):
            vals = [v for v in values if v is not None]
            return {"pass": sum(vals), "n": len(vals), "rate": round(sum(vals) / len(vals), 3) if vals else None}

        per = {"cited_correctly": rate(cited_correctly(a) for _, a, _ in rows),
               "correct_paper": rate(a["correct_paper"] for _, a, _ in rows),
               "correct_section": rate(a["correct_section"] for _, a, _ in rows),
               "figure_referenced": rate(a["figure_referenced"] for _, a, _ in rows),
               "no_unsupported_claims": rate((j["n_unsupported"] == 0) if j else None for _, _, j in rows),
               "declined_unanswerable": rate(j["declines"] if j else None for q, _, j in rows if not q.answerable),
               "invalid_markers": sum(len(a["unknown_markers"]) for _, a, _ in rows),
               "by_tier": {t: rate(cited_correctly(a) for q, a, _ in rows if q.tier == t) for t in TIERS},
               "median_latency_s": sorted(a["latency_s"] for _, a, _ in rows)[len(rows) // 2],
               "input_tokens": sum(a["input_tokens"] for _, a, _ in rows),
               "output_tokens": sum(a["output_tokens"] for _, a, _ in rows),
               "answers": len(rows)}
        summary["providers"][provider] = per
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    ap.add_argument("--limit", type=int, help="only the first N questions (pilot)")
    ap.add_argument("--ids", nargs="+", help="only these question ids (pilot)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--claim-citations", action="store_true", help="second judge pass on existing answers")
    args = ap.parse_args()
    if args.report:
        report(args.name)
    elif args.claim_citations:
        claim_citations(args.name)
    else:
        run(args.name, args.providers, args.limit, args.ids)


if __name__ == "__main__":
    main()
