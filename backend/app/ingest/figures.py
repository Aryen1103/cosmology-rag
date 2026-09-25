"""Resolve \\includegraphics paths and convert figures to PNG for the vision model."""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

# Claude downsizes images whose long edge exceeds ~1568px, so larger renders only cost tokens.
MAX_EDGE = 1568
MAX_PDF_ZOOM = 4.0
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg"}
CANDIDATE_SUFFIXES = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps")

STATUS_OK = "ok"
STATUS_NOT_FOUND = "not_found"
STATUS_INLINE_DRAWING = "inline_drawing"
# No Ghostscript on the dev machine, so EPS/PS can't be rasterised.
STATUS_UNSUPPORTED = "unsupported_format"


def resolve_graphic(root: Path, name: str, graphics_paths: list[str]) -> Path | None:
    for directory in ["", *graphics_paths]:
        for suffix in CANDIDATE_SUFFIXES:
            candidate = root / directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def convert_to_png(src: Path, dst: Path) -> str:
    suffix = src.suffix.lower()
    try:
        if suffix in RASTER_SUFFIXES:
            with Image.open(src) as image:
                if image.mode in ("RGBA", "LA", "P"):
                    image = image.convert("RGBA")
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")
                image.thumbnail((MAX_EDGE, MAX_EDGE))
                image.save(dst, "PNG")
            return STATUS_OK
        if suffix == ".pdf":
            with pymupdf.open(src) as doc:
                page = doc[0]
                zoom = min(MAX_EDGE / max(page.rect.width, page.rect.height), MAX_PDF_ZOOM)
                page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(dst)
            return STATUS_OK
        return STATUS_UNSUPPORTED
    except Exception as exc:  # corrupt or exotic figure files shouldn't sink the whole paper
        return f"error: {type(exc).__name__}: {exc}"
