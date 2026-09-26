# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Multimodal RAG over arXiv `astro-ph.CO` papers: it retrieves text passages *and* figures, then a vision-language model (Claude or DeepSeek) answers with citations. `README.md` holds the full design rationale and results.

## Commands

Backend commands run from `backend/`, using the virtualenv at `backend/venv/` (Windows: `./venv/Scripts/activate`). `pytest.ini` sets `pythonpath = .`, so imports use `app.*`.

```bash
python -m pytest                                          # all tests (no network or API keys needed)
python -m pytest tests/test_latex.py::test_name           # single test
python scripts/ingest.py --max 20                         # fetch + parse newest papers into data/papers/
python scripts/ingest.py --query "cat:astro-ph.CO AND abs:BAO" --max 50
python scripts/build_index.py                             # embed into data/index/ (only new/changed papers)
python scripts/ask.py "question" --retrieval-only         # retrieval without an API call
python scripts/ask.py "question" --provider claude|deepseek|both
uvicorn app.main:app --port 8000                          # HTTP API (restart after rebuilding the index)
```

Frontend commands run from `frontend/`:

```bash
npm run dev      # http://localhost:5173, proxies /api to 127.0.0.1:8000
npm run build    # tsc -b && vite build
npm run lint     # oxlint
```

Docker, from the repo root: `docker build -t cosmology-rag .` then `docker run -p 8000:8000 --env-file backend/.env cosmology-rag`. The image bakes in `data/index` and each paper's `paper.json` + `figures/` (see `.dockerignore`) and the bge model, and runs with `HF_HUB_OFFLINE=1`. If `MODEL_NAME` in `app/index/embeddings.py` changes, update the Dockerfile too. CI (`.github/workflows/ci.yml`) runs pytest **without torch**, so nothing may import torch or transformers at module level.

Install: CPU-only torch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `pip install -r requirements-dev.txt`.

API keys (`ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`) and optional `CLAUDE_MODEL` / `DEEPSEEK_MODEL` overrides are read from `backend/.env` (`ENV_FILE` in `app/config.py`). `data/`, by contrast, lives at the repo root and is generated and gitignored.

## Architecture

The pipeline has three stages, which communicate only through files under `data/`:

1. **Ingest** (`app/ingest/`): `arxiv_client` downloads the LaTeX e-print, not the PDF, and enforces a ~3 s delay between arXiv requests. `pipeline.unpack_source` handles a tarball, a gzipped single file or a PDF. `latex.py` finds the `\documentclass` file, inlines `\input`/`\include`, expands zero-argument macros, keeps math as raw LaTeX, and links figures to the paragraphs that `\ref` them. `figures.py` converts images to PNG (at most 1568 px on the long edge; PDFs through PyMuPDF). Output: `data/papers/<paper_dir_name(arxiv_id)>/paper.json` plus `figures/figN_M.png`. If a paper has no source, ingest falls back to extracting text from the PDF.
2. **Index** (`app/index/`): `store.records_for_paper` turns sections into ~220-word chunks (40-word overlap) and figures into caption plus citing paragraphs, and skips boilerplate sections (acknowledgements, funding, …). The embedding input is `"{title} | {section}\n{text}"`. `embeddings.py` runs `BAAI/bge-small-en-v1.5` through plain `transformers` with CLS pooling. Embeddings are cached per paper in `embeddings.npy` + `embeddings.key`, keyed on a SHA-256 of the model name and the exact inputs, so any change to chunking or embedding text re-embeds everything. `build_index` concatenates the per-paper arrays into `data/index/embeddings.npy` + `records.jsonl`.
3. **Search + answer**: `Index.search` is a brute-force numpy dot product that returns text hits and figure hits **ranked separately** (`k_text`, `k_figures`), so the far more numerous chunks can't crowd out figures. `answer/base.build_sources` labels the hits `[S#]`/`[F#]`. Both providers receive identical sources and instructions so the comparison is fair:
   - `ClaudeProvider` sends passages as citable documents (native API citations, quoted spans) plus image blocks.
   - `DeepSeekProvider` sends `image_url` parts; `marker_citations` checks the `[S#]`/`[F#]` markers against the real sources and reports unknown ones in `unknown_markers`.
   - Providers are loaded lazily through `app/answer/__init__.get_provider`, which raises `RuntimeError` when a key is missing.

`app/main.py` (FastAPI) wraps stage 3. It loads the index and warms the embedding model in `lifespan`, caches providers, and runs `provider="both"` in parallel threads. It returns a 400 error for a missing key before spending on either provider. The figure route only serves files matching `fig\d+_\d+\.png`, which guards against path traversal. CORS allows the Vite dev server on `:5173`. If `frontend/dist` exists (or `FRONTEND_DIST`), it is mounted at `/` after all `/api` routes. `/api/ask` goes through `AskLimiter`, an in-memory limit per IP (first `X-Forwarded-For` entry) and per day, configured by `ASK_LIMIT_PER_IP_PER_HOUR` / `ASK_LIMIT_PER_DAY`; `/api/search` is never limited. The server loads the index only at startup, so rebuilding it needs a restart or `POST /api/reload-index`. `tests/test_api.py` skips the lifespan by not using `with TestClient(...)`, and monkeypatches `main._state["index"]` and `main._providers`.

The frontend (`frontend/`, React + TypeScript + Vite) talks only to `/api` and holds no state beyond the current result. Two things there are easy to break:

- `src/markers.ts` must match the backend's marker regex in `app/answer/base.py`, including grouped markers like `[S2, F1]`. It rewrites markers as `#cite-` links that `Markdown.tsx` renders as chips. It also converts `\(…\)` / `\[…\]` to `$` delimiters, because models switch between the two styles and `remark-math` only understands `$`.
- `katex` is pinned to the same minor version that `rehype-katex` depends on (0.16). KaTeX 0.18 renamed CSS classes such as `.sizing`, so a mismatched stylesheet silently breaks subscripts. After changing dependencies, restart Vite with `node_modules/.vite` cleared, because it caches the old CSS.

## Constraints

- **Do not add `sentence-transformers`, scikit-learn or a vector database.** The development machine's security policy blocks some compiled Python extensions (a scikit-learn DLL, for one). This is why embedding uses plain `transformers` and search uses numpy.
- TikZ figures have no image. They are kept as caption-only figures and marked `inline_drawing`. EPS figures are skipped unless Ghostscript is installed.
- LaTeX parsing is intentionally "good enough": macros that take arguments are not expanded, and unknown packages degrade to plain text rather than failing the paper.
