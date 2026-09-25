from __future__ import annotations

import time

import anthropic

from app.answer.base import (
    SYSTEM_PROMPT,
    Answer,
    Citation,
    Source,
    figure_intro,
    image_b64,
    marker_citations,
)
from app.config import CLAUDE_MODEL, require_env

MAX_TOKENS = 16000


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL) -> None:
        require_env("ANTHROPIC_API_KEY")
        self.model = model
        self._client = anthropic.Anthropic()

    def _content(self, question: str, sources: list[Source]) -> list[dict]:
        # Text passages go in as citable documents: the API then returns the exact
        # quoted span behind each claim, which the UI can show for verification.
        content: list[dict] = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": s.text},
                "title": s.heading,
                "citations": {"enabled": True},
            }
            for s in sources
            if s.kind == "text"
        ]
        for s in sources:
            if s.kind != "figure":
                continue
            content.append({"type": "text", "text": figure_intro(s)})
            for path in s.image_paths:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_b64(path)},
                })
        content.append({"type": "text", "text": f"Question: {question}"})
        return content

    def answer(self, question: str, sources: list[Source]) -> Answer:
        started = time.monotonic()
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": self._content(question, sources)}],
        )
        latency = time.monotonic() - started

        if response.stop_reason == "refusal":
            text = "The model declined to answer this request."
            citations: list[Citation] = []
        else:
            text_sources = [s for s in sources if s.kind == "text"]
            text = "".join(b.text for b in response.content if b.type == "text")
            citations = []
            for block in response.content:
                if block.type != "text":
                    continue
                for c in block.citations or []:
                    index = getattr(c, "document_index", None)
                    if index is not None and index < len(text_sources):
                        citations.append(Citation(text_sources[index].source_id, c.cited_text))

        marked, unknown = marker_citations(text, sources)
        cited_ids = {c.source_id for c in citations}
        citations += [c for c in marked if c.source_id not in cited_ids]

        return Answer(
            provider=self.name,
            model=response.model,
            text=text,
            citations=citations,
            unknown_markers=unknown,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            stop_reason=response.stop_reason,
        )
