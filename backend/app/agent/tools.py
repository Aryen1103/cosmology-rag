"""The agent's tools: their schemas, and a Toolbox that runs them for one question.

The Toolbox owns the source registry. Every passage or figure the agent sees gets a
stable [S#]/[F#] label for the whole run, so a later search that returns the same
chunk reuses its label and the final answer's markers can be checked like the
single-shot providers' are.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.agent.cosmology import flat_lcdm
from app.answer.base import Source, figure_intro, image_b64, source_from_hit
from app.config import PAPERS_DIR
from app.index.store import Hit
from app.ingest.pipeline import paper_dir_name

MAX_K_TEXT = 8
MAX_K_FIGURES = 4
MAX_PAPER_MATCHES = 15
_FIGURE_ID_RE = re.compile(r"^F\d+$")


class ToolError(Exception):
    """Bad tool input; the message goes back to the model as an is_error result."""


def _schema(properties: dict) -> dict:
    # strict: True needs every property required and no extras.
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Semantic search over the indexed papers. Returns the best-matching text passages, labelled [S#], "
            "and figures, labelled [F#], ranked separately. Figures come back as caption plus the text that "
            "discusses them; call view_figure to see the image. Search again with different wording, or with "
            "a paper's title in the query, if the first results miss."
        ),
        "strict": True,
        "input_schema": _schema({
            "query": {"type": "string", "description": "What to look for, phrased like the text you hope to find."},
            "k_text": {"type": "integer", "description": f"Passages to return, 1-{MAX_K_TEXT}. 5 is a good default."},
            "k_figures": {"type": "integer", "description": f"Figures to return, 0-{MAX_K_FIGURES}."},
        }),
    },
    {
        "name": "view_figure",
        "description": (
            "Show the image of a figure already returned by search_papers. Use it when the answer depends on "
            "what the plot shows: values read off an axis, trends, contours, which curve is which."
        ),
        "strict": True,
        "input_schema": _schema({"figure_id": {"type": "string", "description": "A figure label such as F2."}}),
    },
    {
        "name": "find_papers",
        "description": (
            "Find papers whose title or author list contains the given text (case-insensitive). Returns arXiv "
            f"id, date, title and first authors for up to {MAX_PAPER_MATCHES} papers, newest first. Use it for "
            "questions about specific authors or papers by name, which semantic search handles poorly."
        ),
        "strict": True,
        "input_schema": _schema({"text": {"type": "string", "description": "An author surname or words from a title."}}),
    },
    {
        "name": "get_paper",
        "description": "Title, authors, date, abstract and section outline of one paper, by arXiv id.",
        "strict": True,
        "input_schema": _schema({"arxiv_id": {"type": "string", "description": "For example 2609.14606v1."}}),
    },
    {
        "name": "lcdm_calculator",
        "description": (
            "Background quantities in flat ΛCDM (radiation neglected): H(z), comoving, luminosity and angular "
            "diameter distances in Mpc, lookback time, age at z and age today in Gyr. Use it for any derived "
            "number rather than computing it yourself, with H0 and Om0 taken from the sources when they give them."
        ),
        "strict": True,
        "input_schema": _schema({
            "z": {"type": "number", "description": "Redshift, 0-10000."},
            "H0": {"type": "number", "description": "Hubble constant in km/s/Mpc."},
            "Om0": {"type": "number", "description": "Matter density today, between 0 and 1."},
        }),
    },
]


def _source_key(hit: Hit) -> tuple:
    r = hit.record
    return (r.kind, r.arxiv_id, r.section, r.text)


class Toolbox:
    def __init__(self, index, papers: list[dict], papers_dir: Path = PAPERS_DIR) -> None:
        self._index = index
        self._papers = {p["arxiv_id"]: p for p in papers}
        self._papers_dir = papers_dir
        self.sources: list[Source] = []
        self.scores: list[float] = []  # best retrieval score per source, parallel to `sources`
        self._by_key: dict[tuple, int] = {}

    def run(self, name: str, tool_input: dict) -> tuple[str | list[dict], bool]:
        """Run one tool call. Returns (tool_result content, is_error)."""
        handler = getattr(self, f"_{name}", None) if name in {t["name"] for t in TOOLS} else None
        if handler is None:
            return f"Unknown tool {name!r}.", True
        try:
            return handler(**tool_input), False
        except (ToolError, ValueError, TypeError) as exc:
            return f"Error: {exc}", True

    def _register(self, hit: Hit) -> tuple[Source, bool]:
        key = _source_key(hit)
        if key in self._by_key:
            i = self._by_key[key]
            self.scores[i] = max(self.scores[i], hit.score)
            return self.sources[i], False
        prefix = "S" if hit.record.kind == "text" else "F"
        number = sum(s.source_id[0] == prefix for s in self.sources) + 1
        self._by_key[key] = len(self.sources)
        self.sources.append(source_from_hit(hit, f"{prefix}{number}"))
        self.scores.append(hit.score)
        return self.sources[-1], True

    def _search_papers(self, query: str, k_text: int, k_figures: int) -> str:
        if not query.strip():
            raise ToolError("query is empty")
        k_text = min(max(int(k_text), 1), MAX_K_TEXT)
        k_figures = min(max(int(k_figures), 0), MAX_K_FIGURES)
        text_hits, figure_hits = self._index.search(query, k_text, k_figures)
        if not text_hits and not figure_hits:
            return "No results."
        blocks = []
        for hit in text_hits + figure_hits:
            source, new = self._register(hit)
            if not new:
                blocks.append(f"{source.heading} (already shown earlier)")
            elif source.kind == "text":
                blocks.append(f"{source.heading}\n{source.text}")
            else:
                images = f"{len(source.image_paths)} image(s)" if source.image_paths else "no image"
                blocks.append(f"{figure_intro(source)}\n({images}; view_figure(\"{source.source_id}\") to see it)")
        return "\n\n".join(blocks)

    def _view_figure(self, figure_id: str) -> list[dict]:
        figure_id = figure_id.strip().strip("[]").upper()
        source = next((s for s in self.sources if s.source_id == figure_id), None)
        if not _FIGURE_ID_RE.match(figure_id) or source is None:
            raise ToolError(f"{figure_id!r} is not a figure from an earlier search_papers result")
        content: list[dict] = [{"type": "text", "text": figure_intro(source)}]
        for path in source.image_paths:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64(path)},
            })
        return content

    def _find_papers(self, text: str) -> str:
        needle = text.strip().lower()
        if not needle:
            raise ToolError("text is empty")
        matches = [
            p for p in self._papers.values()
            if needle in p["title"].lower() or any(needle in a.lower() for a in p["authors"])
        ]
        if not matches:
            return f"No paper has {text!r} in its title or authors."
        lines = []
        for p in matches[:MAX_PAPER_MATCHES]:
            authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
            lines.append(f"arXiv:{p['arxiv_id']} ({p['published'][:10]}) {p['title']} - {authors}")
        more = len(matches) - MAX_PAPER_MATCHES
        return "\n".join(lines) + (f"\n... and {more} more; use a more specific text." if more > 0 else "")

    def _get_paper(self, arxiv_id: str) -> str:
        # Only ids from the corpus reach the filesystem, so model input can't name another path.
        meta = self._papers.get(arxiv_id.strip().removeprefix("arXiv:"))
        if meta is None:
            raise ToolError(f"No paper {arxiv_id!r} in the library; use find_papers or search_papers to find ids")
        paper = json.loads(
            (self._papers_dir / paper_dir_name(meta["arxiv_id"]) / "paper.json").read_text(encoding="utf-8")
        )
        outline = "\n".join("  " * (len(s["path"]) - 1) + s["path"][-1] for s in paper["sections"])
        return (
            f"arXiv:{meta['arxiv_id']} ({meta['published'][:10]})\n{meta['title']}\n"
            f"Authors: {', '.join(meta['authors'])}\n\nAbstract: {meta['abstract']}\n\n"
            f"Sections:\n{outline}\n\nFigures: {meta['figures']}"
        )

    def _lcdm_calculator(self, z: float, H0: float, Om0: float) -> str:  # noqa: N803 - matches the schema
        result = flat_lcdm(float(z), float(H0), float(Om0))
        lines = [f"Flat ΛCDM with H0 = {H0} km/s/Mpc, Om0 = {Om0}, at z = {z}:"]
        lines += [f"  {name} = {value:.5g}" for name, value in result.items()]
        if z > 10:
            lines.append("Warning: radiation is neglected, so values at z > 10 are increasingly inaccurate "
                         "(at z ~ 1100 the age is ~25% too high). Say so if you use them.")
        return "\n".join(lines)
