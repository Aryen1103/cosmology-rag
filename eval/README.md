# Retrieval evaluation

Measures whether the passage or figure that answers a question gets retrieved,
so changes to retrieval (hybrid search, reranking, a fine-tuned embedding
model, new chunking) can be compared on the same questions. It makes no API
calls; a run only embeds the questions locally and takes seconds.

## Files

| File | What it is |
|---|---|
| `split.json` | Frozen paper-level split: 50 test papers (eval questions) and 151 train papers (reserved for fine-tuning data). Read this file rather than recomputing the split, because adding papers can shift a recomputed one. |
| `retrieval_questions.jsonl` | 138 questions (90 text, 48 figure) from the 50 test papers. |
| `runs/<name>/` | `summary.json` (metrics) and `results.jsonl` (rank and top hits per question) for each run. |

## How the questions were made

`backend/scripts/sample_eval_passages.py` picks two text chunks (from different
sections where possible, at least 80 words, no abstracts) and one figure from
each test paper, using a seeded random choice. Questions were written only from
those samples, so the set isn't biased toward passages that happen to be easy
to retrieve. Passages that were pure equations or tables, or that many papers
would answer equally well, were skipped.

Each question is phrased the way a researcher might ask it, paraphrasing rather
than copying the passage. Tags mark `conceptual`, `fact` and `numeric`
questions.

## How a question is graded

Gold labels name the paper, not an index row, so they survive re-chunking:

- **Text:** a retrieved text chunk counts if it comes from the right paper and
  contains the question's `evidence` quote (case- and whitespace-insensitive).
- **Figure:** a retrieved figure counts if it is the right paper's `figure_id`.

Text and figure rankings are scored separately, as the app retrieves them.
`--check` confirms every gold item exists in the current index and that no
evidence quote also appears in another paper.

**Headline metrics:** text **hit@6** (the answer is among the 6 passages the
model receives) and figure **hit@1**. Figure hit@3, which is what the model
receives, was already 100% at baseline, so it cannot show improvements.

## Baseline (bge-small-en-v1.5, dense cosine search)

| | n | hit@1 | hit@3 | hit@6 | hit@20 | MRR |
|---|---|---|---|---|---|---|
| Text | 90 | 0.444 | 0.611 | **0.778** (95% CI 0.689–0.856) | 0.944 | 0.567 |
| Figure | 48 | **0.792** (95% CI 0.667–0.896) | 1.000 | – | 1.000 | 0.882 |

The right paper is in the top 6 text hits for every question: every text miss
is the right paper with the wrong passage, usually ranked 7–20. That points at
reranking within the candidates rather than at finding the paper.

## Limitations

- Questions were written while reading the passage, so they share more
  vocabulary with it, and name specific objects more often, than a real user's
  question would. Absolute scores are optimistic; comparisons between variants
  on the same set are the main use.
- With 90 text questions, a single run's hit@6 is uncertain by about ±8 points.
  `--compare` uses a paired bootstrap, which is tighter, but differences of a
  few points are still within noise.
- Choosing many settings by repeatedly scoring against this set would overfit
  it. Prefer deciding variants on design grounds and measuring each once.

## Usage

From `backend/`:

```bash
python scripts/eval_retrieval.py --check                    # validate questions against the index
python scripts/eval_retrieval.py --name baseline            # score the current index into eval/runs/baseline/
python scripts/eval_retrieval.py --compare baseline hybrid  # paired difference between two runs
```
