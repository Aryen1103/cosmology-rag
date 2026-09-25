"""Download, parse and store one arXiv paper as data/papers/<id>/paper.json + figures/."""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
from pathlib import Path

import pymupdf

from app.ingest.arxiv_client import ArxivClient, PaperMeta
from app.ingest.figures import STATUS_INLINE_DRAWING, STATUS_NOT_FOUND, convert_to_png, resolve_graphic
from app.ingest.latex import parse_latex_tree

SOURCE_LATEX = "latex"
SOURCE_PDF = "pdf"


def paper_dir_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def unpack_source(raw: bytes, dest: Path) -> str:
    """Unpack an arXiv e-print into dest; returns SOURCE_LATEX or SOURCE_PDF.

    An e-print is a tarball, a single gzipped .tex file, or (for PDF-only
    submissions) the PDF itself.
    """
    if raw.startswith(b"%PDF"):
        (dest / "paper.pdf").write_bytes(raw)
        return SOURCE_PDF
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            tar.extractall(dest, filter="data")
        return SOURCE_LATEX
    except tarfile.ReadError:
        pass
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    if data.startswith(b"%PDF"):
        (dest / "paper.pdf").write_bytes(data)
        return SOURCE_PDF
    (dest / "main.tex").write_bytes(data)
    return SOURCE_LATEX


def _pdf_sections(pdf_path: Path) -> list[dict]:
    with pymupdf.open(pdf_path) as doc:
        return [
            {"path": [f"Page {i + 1}"], "text": page.get_text(sort=True).strip()}
            for i, page in enumerate(doc)
            if page.get_text().strip()
        ]


def ingest_paper(meta: PaperMeta, raw_source: bytes, papers_dir: Path) -> dict:
    out_dir = papers_dir / paper_dir_name(meta.arxiv_id)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        src_root = Path(tmp)
        source_type = unpack_source(raw_source, src_root)

        if source_type == SOURCE_PDF:
            sections, figures = _pdf_sections(src_root / "paper.pdf"), []
        else:
            parsed = parse_latex_tree(src_root)
            sections = [{"path": s.path, "text": s.text} for s in parsed.sections]
            figures = []
            for fig in parsed.figures:
                images = []
                for n, graphic in enumerate(fig.graphics, start=1):
                    src = resolve_graphic(parsed.root, graphic, parsed.graphics_paths)
                    if src is None:
                        images.append({"source": graphic, "file": None, "status": STATUS_NOT_FOUND})
                        continue
                    png_name = f"{fig.figure_id}_{n}.png"
                    status = convert_to_png(src, figures_dir / png_name)
                    images.append({
                        "source": graphic,
                        "file": f"figures/{png_name}" if status == "ok" else None,
                        "status": status,
                    })
                if fig.inline_drawing:
                    images.append({"source": "inline drawing", "file": None, "status": STATUS_INLINE_DRAWING})
                figures.append({
                    "figure_id": fig.figure_id,
                    "label": fig.label,
                    "caption": fig.caption,
                    "citing_paragraphs": fig.citing_paragraphs,
                    "images": images,
                })

    record = {
        "meta": meta.to_dict(),
        "source_type": source_type,
        "sections": sections,
        "figures": figures,
    }
    (out_dir / "paper.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def fetch_and_ingest(client: ArxivClient, meta: PaperMeta, papers_dir: Path) -> dict:
    return ingest_paper(meta, client.download_source(meta.arxiv_id), papers_dir)
