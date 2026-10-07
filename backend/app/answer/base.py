"""Provider-neutral pieces of the answer step: sources, prompt, result shape.

Both providers get identical sources, numbering and instructions so their
outputs can be compared fairly; only the wire format differs.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PAPERS_DIR
from app.index.store import Hit
from app.ingest.pipeline import paper_dir_name

MAX_IMAGES_PER_FIGURE = 2

SYSTEM_PROMPT = """You answer questions about research papers (mostly cosmology) using only the sources provided.

Sources are text passages labelled [S1], [S2], ... and figures labelled [F1], [F2], ... . Figures come with their caption and, when available, the image itself - read values, trends and labels directly from the image when the question needs them.

Rules:
- Base every claim on the sources. If they don't contain the answer, say so plainly instead of drawing on outside knowledge.
- Cite the source label right after each claim it supports, e.g. "... σ8 = 0.81 [S3]." Refer to figures as [F2].
- When sources disagree or come from different papers, say which paper says what.
- Keep LaTeX math as-is where it helps precision.
- Be concise: answer the question first, then only the supporting detail that matters."""

# Single markers "[S1]" and grouped ones "[S2, F1]" / "[S3; S6]", which DeepSeek often emits.
_MARKER_GROUP_RE = re.compile(r"\[([SF]\d+(?:\s*[,;]\s*[SF]\d+)*)\]")
_MARKER_RE = re.compile(r"[SF]\d+")


@dataclass
class Source:
    source_id: str  # "S1" / "F1"
    kind: str
    arxiv_id: str
    paper_title: str
    section: str
    text: str
    image_paths: list[Path] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"[{self.source_id}] arXiv:{self.arxiv_id} - {self.paper_title} - {self.section}"


@dataclass
class Citation:
    source_id: str
    cited_text: str | None = None  # exact quoted span, when the provider returns one


@dataclass
class Answer:
    provider: str
    model: str
    text: str
    citations: list[Citation]
    unknown_markers: list[str]
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str | None = None


def source_from_hit(hit: Hit, source_id: str) -> Source:
    r = hit.record
    if r.kind == "text":
        return Source(source_id, "text", r.arxiv_id, r.paper_title, r.section, r.text)
    paper_dir = PAPERS_DIR / paper_dir_name(r.arxiv_id)
    images = [paper_dir / f for f in (r.image_files or [])][:MAX_IMAGES_PER_FIGURE]
    label = f"Figure ({r.label})" if r.label else "Figure"
    return Source(source_id, "figure", r.arxiv_id, r.paper_title, label, r.text, images)


def build_sources(text_hits: list[Hit], figure_hits: list[Hit]) -> list[Source]:
    return [source_from_hit(h, f"S{n}") for n, h in enumerate(text_hits, start=1)] + [
        source_from_hit(h, f"F{n}") for n, h in enumerate(figure_hits, start=1)
    ]


def figure_intro(source: Source) -> str:
    note = "" if source.image_paths else "\n(No image available for this figure - caption and text only.)"
    return f"{source.heading}\nCaption and context: {source.text}{note}"


def image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def marker_citations(answer_text: str, sources: list[Source]) -> tuple[list[Citation], list[str]]:
    """Citations from [S#]/[F#] markers, plus markers that name no real source."""
    known = {s.source_id for s in sources}
    seen: list[str] = []
    for group in _MARKER_GROUP_RE.findall(answer_text):
        for marker in _MARKER_RE.findall(group):
            if marker not in seen:
                seen.append(marker)
    return [Citation(m) for m in seen if m in known], [m for m in seen if m not in known]
