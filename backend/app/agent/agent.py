"""A small tool-calling agent over the paper library.

Unlike the single-shot providers, which answer from one fixed retrieval, the agent
decides what to search for, can search again, look at figure images, look papers up
by author or title, and compute derived numbers with lcdm_calculator.

The loop is written by hand rather than with the SDK's beta tool runner: view_figure
returns images, the last turn must switch tools off, and tests drive it with a fake
client.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic

from app.agent.tools import TOOLS, Toolbox
from app.answer.base import Answer, Source, marker_citations
from app.config import AGENT_MODEL, require_env

MAX_TOKENS = 16000
MAX_TURNS = 6  # model calls per question; the last one has tools switched off
EFFORT = "medium"  # explicit: Opus 5.5's default is medium, other models' is high

SYSTEM_PROMPT = """You are a research assistant for a library of recent arXiv cosmology papers (astro-ph.CO). Answer questions by gathering evidence from those papers with your tools.

How to work:
- Search before answering. If the results miss, search again with different wording; split questions that compare papers or quantities into separate searches.
- Look at a figure with view_figure when the answer depends on what the plot shows.
- Use lcdm_calculator for derived quantities (distances, ages, H(z)) instead of computing them yourself, and state the parameters you used and where they came from.
- Stop searching once you have enough to answer; most questions need two or three tool calls.

Answering:
- Base every claim about the papers on the sources you retrieved. If they don't contain the answer, say so plainly instead of drawing on outside knowledge.
- Cite the source label right after each claim it supports, e.g. "... σ8 = 0.81 [S3]." Refer to figures as [F2]. Only cite labels that your tools returned.
- When sources disagree or come from different papers, say which paper says what.
- Keep LaTeX math as-is where it helps precision.
- Be concise: answer the question first, then only the supporting detail that matters."""

_WRAP_UP = "Tool budget used up: answer now from the sources you already have."


@dataclass
class ToolStep:
    tool: str
    input: dict
    output: str  # first line of the result, for traces and the UI
    is_error: bool


@dataclass
class AgentRun:
    answer: Answer
    sources: list[Source]
    scores: list[float]
    steps: list[ToolStep]


def _summary(content: str | list[dict]) -> str:
    text = content if isinstance(content, str) else content[0]["text"]
    if not isinstance(content, str):
        text += f" (+{len(content) - 1} image(s))"
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:200]


class CosmologyAgent:
    name = "agent"

    def __init__(self, model: str = AGENT_MODEL, client: anthropic.Anthropic | None = None,
                 max_turns: int = MAX_TURNS) -> None:
        if client is None:
            require_env("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic()
        self.model = model
        self.max_turns = max_turns
        self._client = client

    def run(self, question: str, index, papers: list[dict],
            on_step: Callable[[ToolStep], None] | None = None) -> AgentRun:
        toolbox = Toolbox(index, papers)
        messages: list[dict] = [{"role": "user", "content": question}]
        steps: list[ToolStep] = []
        input_tokens = output_tokens = 0
        started = time.monotonic()

        for turn in range(1, self.max_turns + 1):
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice={"type": "none" if turn == self.max_turns else "auto"},
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT},
                cache_control={"type": "ephemeral"},  # each turn resends the whole history
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
            usage = response.usage
            input_tokens += (usage.input_tokens + (usage.cache_creation_input_tokens or 0)
                             + (usage.cache_read_input_tokens or 0))
            output_tokens += usage.output_tokens
            # Append the full content, thinking blocks included, unchanged: the history must stay append-only.
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break

            results: list[dict] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                content, is_error = toolbox.run(block.name, block.input)
                step = ToolStep(block.name, dict(block.input), _summary(content), is_error)
                steps.append(step)
                if on_step:
                    on_step(step)
                result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            if turn + 1 == self.max_turns:
                results.append({"type": "text", "text": _WRAP_UP})
            messages.append({"role": "user", "content": results})

        if response.stop_reason == "refusal":
            text = "The model declined to answer this request."
        else:
            # Only the last turn's text: on Opus 5.5 notes between tool calls arrive as thinking blocks.
            text = "".join(b.text for b in response.content if b.type == "text")
        citations, unknown = marker_citations(text, toolbox.sources)

        answer = Answer(
            provider=self.name,
            model=response.model,
            text=text,
            citations=citations,
            unknown_markers=unknown,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=time.monotonic() - started,
            stop_reason=response.stop_reason,
        )
        return AgentRun(answer, toolbox.sources, toolbox.scores, steps)
