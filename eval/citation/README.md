# Citation evaluation: Claude vs DeepSeek

**Question.** Given the same retrieved sources, does each model cite the right
paper, section and figure, and avoid claims the sources do not support?

**Headline.** Both models cited the right paper, section and figure on all 16
answerable questions, and both rejected both false-premise questions. Where a
claim carried a citation, the cited source supported it 97% of the time for
Claude and 99% for DeepSeek, a difference within noise. The clear difference
is coverage: Claude made 75% more claims per answer and left 26% of them
uncited, against 10% for DeepSeek (paired 95% CI for the gap: +7 to +26 points).

## Setup

- **Questions** (`questions.jsonl`): 18 across three tiers. *Simple* (6): one
  fact from one passage. *Medium* (6): needs reading a figure. *Complex* (6):
  four cross-paper syntheses (2–3 papers each) and two false-premise questions
  whose answer is not in the corpus (absence checked against every indexed chunk).
- **Same evidence for both.** Each question is retrieved once with the app's
  hybrid search (6 passages + 3 figures) and both models answer from those
  sources, with the app's system prompt.
- **Models.** DeepSeek (`deepseek-flash`) through the app's provider, images
  inline. Claude Opus 5.5 through the Claude Code CLI in headless mode,
  viewing figure images with a read-only file tool in an isolated folder. Both
  therefore cite with `[S#]`/`[F#]` markers; the app's API-native Claude
  citations were not used, so this compares the models, not the app's two
  integrations.
- **Rubric per answer.**

| Item | How it is checked |
|---|---|
| Correct paper cited | every retrieved gold paper has at least one cited source (automatic) |
| Correct section | a cited passage contains the gold evidence quote, or the gold figure is cited (automatic) |
| Figure referenced | for figure questions, the gold figure is cited (automatic) |
| No unsupported claims | a judge lists the answer's claims and marks each supported or not |

  A second judge pass, added after the first three items saturated, checks
  each claim's *own* citation: does the source attached to that claim support it?
- **Judge.** DeepSeek, with sources and images, provider names removed,
  JSON output. Calibrated before use on planted answers: it accepted a correct
  answer, flagged an invented claim and a wrong-paper attribution, accepted a
  correct refusal, and on the per-claim pass told a right citation from a wrong
  one and from a missing one.

## Results

| | Claude Opus 5.5 | DeepSeek |
|---|---|---|
| Cited correctly (paper + section + figure), answerable questions | **16/16** | **16/16** |
| &nbsp;&nbsp;Simple / Medium / Complex | 6/6 · 6/6 · 4/4 | 6/6 · 6/6 · 4/4 |
| Figure cited on figure questions | 6/6 | 6/6 |
| Answers with no unsupported claim | 16/18 | 14/18 |
| False premise rejected | 2/2 | 2/2 |
| Claims whose cited source supports them | **97.1%** (190 of 196) | **98.6%** (143 of 145) |
| Claims with no citation | **26%** | **10%** |
| Claims per answer | 15.6 | 8.9 |
| Sources cited per answer | 4.2 | 3.2 |
| Invalid markers (citing a source that does not exist) | 0 | 0 |
| Median latency | 12.8 s | 4.9 s |

Gold coverage: retrieval delivered the gold passage or figure for all 12
Simple and Medium questions, but only some of the papers for the cross-paper
questions (1 of 3 for the LZ question). Complex citation items are scored
against the gold papers that were retrieved.

## What this means

- **Citation correctness does not separate the models.** Both pick the right
  sources and rarely attach a citation to a claim it does not support. The
  rubric's automatic items saturate: an answer that cites most of what it was
  given passes them, so per-claim checking is the more informative measure.
- **The errors are mostly figure readings.** Of the 8 wrongly cited and 6
  unsupported claims, most are values read off plots ("the 68% band bottoms out
  around 0.5–0.6"). Here the judge's own reading of the image may be what is
  wrong, so these are the least certain judgements.
- **Claude writes more and cites less of it.** Longer answers with more claims,
  a quarter of them unmarked, often framing or summary sentences, but some
  factual. For a citation-first product, the marker instruction needs to be
  stricter, or the API's native citations (which the app uses for Claude) are
  the better route; this run did not test them.
- **Both handled false premises well**, saying the sources did not contain the
  answer. The judge's "declines" flag missed one of each, because both answers
  went on to summarise what the paper does cover; the table reports the answers
  as read.

## Limitations

18 questions and one run per model: differences of a few answers are noise
(only the uncited-claims gap has an interval excluding zero). Questions and
gold labels were written by one person. The judge is one of the two models
under test and may favour its own answers; it did not obviously do so here
(it flagged more DeepSeek answers than Claude ones). The Claude answers ran
through a different harness from the app's.

## Reproduce

From `backend/` (the Claude half needs the Claude Code CLI signed in; DeepSeek
needs `DEEPSEEK_API_KEY`):

```bash
python scripts/eval_citation.py --name main                    # answer + judge, resumable
python scripts/eval_citation.py --name main --claim-citations  # per-claim citation pass
python scripts/eval_citation.py --name main --report           # summary.json
```

Answers, judgments and per-claim checks for every question are in `runs/main/`.
