from __future__ import annotations

import time

import httpx

from app.answer.base import (
    SYSTEM_PROMPT,
    Answer,
    Source,
    figure_intro,
    image_b64,
    marker_citations,
)
from app.config import DEEPSEEK_API_URL, DEEPSEEK_MODEL, require_env

MAX_TOKENS = 8000


class DeepSeekProvider:
    """DeepSeek chat completions with image input (deepseek-flash).

    No native citation API, so grounding relies on the [S#]/[F#] markers the
    system prompt asks for, checked against the real source list afterwards.
    """

    name = "deepseek"

    def __init__(self, model: str = DEEPSEEK_MODEL) -> None:
        self.model = model
        self._http = httpx.Client(
            timeout=180,
            headers={"Authorization": f"Bearer {require_env('DEEPSEEK_API_KEY')}"},
        )

    def _content(self, question: str, sources: list[Source]) -> list[dict]:
        passages = "\n\n".join(f"{s.heading}\n{s.text}" for s in sources if s.kind == "text")
        content: list[dict] = [{"type": "text", "text": f"Text sources:\n\n{passages}"}]
        for s in sources:
            if s.kind != "figure":
                continue
            content.append({"type": "text", "text": figure_intro(s)})
            for path in s.image_paths:
                content.append({
                    "type": "image_url",
                    # "high" keeps full resolution; tick labels and legends in plots are small.
                    "image_url": {"url": f"data:image/png;base64,{image_b64(path)}", "detail": "high"},
                })
        content.append({"type": "text", "text": f"Question: {question}"})
        return content

    def answer(self, question: str, sources: list[Source]) -> Answer:
        started = time.monotonic()
        response = self._http.post(
            DEEPSEEK_API_URL,
            json={
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._content(question, sources)},
                ],
            },
        )
        latency = time.monotonic() - started
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        citations, unknown = marker_citations(text, sources)
        usage = data.get("usage", {})
        return Answer(
            provider=self.name,
            model=data.get("model", self.model),
            text=text,
            citations=citations,
            unknown_markers=unknown,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=latency,
            stop_reason=choice.get("finish_reason"),
        )
