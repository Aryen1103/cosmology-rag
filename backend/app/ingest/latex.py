"""Parse an arXiv LaTeX source tree into sections and figures.

Aims for a robust "good enough" parse rather than a faithful LaTeX
implementation: real arXiv sources use arbitrary packages and macros, so
anything unrecognised degrades to plain text instead of failing the paper.
Math is kept as raw LaTeX, since the answering model reads it directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pylatexenc import latex2text, latexwalker
from pylatexenc.macrospec import MacroSpec

MAX_INPUT_DEPTH = 10
MACRO_EXPANSION_PASSES = 3
MAX_CITING_PARAGRAPHS = 3

# Gaps in pylatexenc's defaults:
# - The parser doesn't know \href takes two arguments, but the stock text handler indexes both, so any
#   paper with a hyperlink crashed with IndexError.
# - These text macros render as "" and silently drop their content, e.g. \texttt{CLASS}.
_KEEP_CONTENT_MACROS = ("texttt", "textsf", "textup", "textmd", "mbox")
_parse_context = latexwalker.get_default_latex_context_db()
_parse_context.add_context_category(
    "cosmology-rag",
    macros=[MacroSpec("href", "{{"), *(MacroSpec(m, "{") for m in _KEEP_CONTENT_MACROS)],
    prepend=True,
)
_text_context = latex2text.get_default_latex_context_db()
_text_context.add_context_category(
    "cosmology-rag",
    macros=[
        latex2text.MacroTextSpec(
            "href",
            simplify_repl=lambda n, l2tobj: f"{l2tobj.node_arg_to_text(n, 1)} <{l2tobj.node_arg_to_text(n, 0)}>",
        ),
        *(latex2text.MacroTextSpec(m, simplify_repl=lambda n, l2tobj: l2tobj.node_arg_to_text(n, 0))
          for m in _KEEP_CONTENT_MACROS),
    ],
    prepend=True,
)
_converter = latex2text.LatexNodes2Text(math_mode="verbatim", latex_context=_text_context)

_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")
_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{")
_FIGURE_ENV_RE = re.compile(r"\\begin\{(figure\*?|wrapfigure)\}(.*?)\\end\{\1\}", re.S)
_REF_RE = re.compile(r"\\(?:ref|autoref|cref|Cref|eqref|Autoref)\*?\s*\{([^}]+)\}")
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")
_NEWCOMMAND_RE = re.compile(
    r"\\(?:re)?newcommand\*?\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(\[\d\])?\s*\{"
)
_DEF_RE = re.compile(r"\\def\s*\\([a-zA-Z]+)\s*\{")
_GRAPHICS_RE = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{")
_AAS_PLOT_RE = re.compile(r"\\plot(?:one|two)\s*\{")

FRONT_MATTER_COMMANDS = (
    "title", "author", "affiliation", "affil", "altaffiliation", "email", "thanks",
    "keywords", "date", "correspondingauthor", "institute", "shorttitle", "shortauthors",
    "received", "accepted", "journalinfo", "submitjournal", "orcid", "collaboration",
    "pacs", "maketitle", "tableofcontents", "bibliography", "bibliographystyle",
)
DROP_ENVIRONMENTS = ("abstract", "thebibliography", "comment", "acknowledgments", "acknowledgements")

REF_MARKER = "[[REF:{}]]"


@dataclass
class Section:
    path: list[str]
    text: str


@dataclass
class Figure:
    figure_id: str
    label: str | None
    caption: str
    graphics: list[str]
    # Drawn in-source (TikZ/pgfplots); there is no image file to render without a TeX install.
    inline_drawing: bool = False
    citing_paragraphs: list[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    root: Path
    sections: list[Section]
    figures: list[Figure]
    graphics_paths: list[str]


def read_braced(s: str, open_idx: int) -> tuple[str, int]:
    """Return the content of the brace group opening at s[open_idx], and the index after it."""
    depth = 0
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : i], i + 1
        i += 1
    return s[open_idx + 1 :], len(s)


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def read_tex_file(path: Path) -> str:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n")


def _resolve_tex(root: Path, name: str) -> Path | None:
    for candidate in (root / name, root / f"{name}.tex"):
        if candidate.is_file():
            return candidate
    return None


def flatten_inputs(tex: str, root: Path, depth: int = 0) -> str:
    def replace(match: re.Match) -> str:
        path = _resolve_tex(root, match.group(1).strip())
        if path is None or depth >= MAX_INPUT_DEPTH:
            return ""
        return flatten_inputs(strip_comments(read_tex_file(path)), root, depth + 1)

    return _INPUT_RE.sub(replace, tex)


def find_main_file(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("*.tex"):
        text = strip_comments(read_tex_file(path))
        if "\\documentclass" in text:
            candidates.append((("\\begin{document}" in text), len(text), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def collect_simple_macros(tex: str) -> dict[str, str]:
    """Zero-argument \\newcommand/\\def definitions, e.g. \\hmpc -> h^{-1}\\,{\\rm Mpc}."""
    macros: dict[str, str] = {}
    for pattern in (_NEWCOMMAND_RE, _DEF_RE):
        for match in pattern.finditer(tex):
            if pattern is _NEWCOMMAND_RE and match.group(2):
                continue
            body, _ = read_braced(tex, match.end() - 1)
            macros[match.group(1)] = body
    return macros


def expand_macros(tex: str, macros: dict[str, str]) -> str:
    if not macros:
        return tex
    pattern = re.compile(r"\\(" + "|".join(re.escape(m) for m in macros) + r")(?![a-zA-Z])")
    for _ in range(MACRO_EXPANSION_PASSES):
        expanded = pattern.sub(lambda m: macros[m.group(1)], tex)
        if expanded == tex:
            break
        tex = expanded
    return tex


def remove_commands(tex: str, names: tuple[str, ...]) -> str:
    pattern = re.compile(r"\\(" + "|".join(names) + r")\*?(?![a-zA-Z])\s*(?:\[[^\]]*\])?")
    out = []
    pos = 0
    for match in pattern.finditer(tex):
        if match.start() < pos:
            continue
        out.append(tex[pos : match.start()])
        pos = match.end()
        rest = tex[pos:]
        stripped = rest.lstrip()
        if stripped.startswith("{"):
            _, end = read_braced(tex, pos + (len(rest) - len(stripped)))
            pos = end
    out.append(tex[pos:])
    return "".join(out)


def remove_environments(tex: str, names: tuple[str, ...]) -> str:
    for name in names:
        tex = re.sub(
            r"\\begin\{" + name + r"\*?\}.*?\\end\{" + name + r"\*?\}", "", tex, flags=re.S
        )
    return tex


def _command_args(tex: str, pattern: re.Pattern) -> list[str]:
    args = []
    for match in pattern.finditer(tex):
        content, _ = read_braced(tex, match.end() - 1)
        args.append(content.strip())
    return args


def _first_arg(tex: str, command: str) -> str | None:
    match = re.search(r"\\" + command + r"\*?\s*(?:\[[^\]]*\])?\s*\{", tex)
    if not match:
        return None
    content, _ = read_braced(tex, match.end() - 1)
    return content


def _graphics_in(env: str) -> list[str]:
    paths = _command_args(env, _GRAPHICS_RE)
    for match in _AAS_PLOT_RE.finditer(env):
        pos = match.end() - 1
        for _ in range(2 if "plottwo" in match.group(0) else 1):
            content, pos = read_braced(env, pos)
            paths.append(content.strip())
            while pos < len(env) and env[pos].isspace():
                pos += 1
            if pos >= len(env) or env[pos] != "{":
                break
    return paths


def to_text(tex: str) -> str:
    text = _converter.latex_to_text(tex, latex_context=_parse_context).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _prepare_references(tex: str) -> str:
    tex = _REF_RE.sub(lambda m: REF_MARKER.format(m.group(1).strip()), tex)
    return _CITE_RE.sub(lambda m: f"[cite: {', '.join(k.strip() for k in m.group(1).split(','))}]", tex)


def _extract_figures(body: str) -> tuple[str, list[Figure]]:
    figures = []

    def replace(match: re.Match) -> str:
        env = match.group(2)
        label = _first_arg(env, "label")
        caption_tex = _first_arg(env, "caption") or ""
        figures.append(
            Figure(
                figure_id=f"fig{len(figures) + 1}",
                label=label.strip() if label else None,
                caption=to_text(_prepare_references(remove_commands(caption_tex, ("label",)))),
                graphics=_graphics_in(env),
                inline_drawing=bool(re.search(r"\\begin\{(tikzpicture|pspicture|picture)\}", env)),
            )
        )
        return "\n\n"

    return _FIGURE_ENV_RE.sub(replace, body), figures


def _split_sections(body: str) -> list[Section]:
    appendix_at = body.find("\\appendix")
    headings = list(_SECTION_RE.finditer(body))
    sections: list[Section] = []
    path: list[str] = []
    levels = {"section": 0, "subsection": 1, "subsubsection": 2}

    first_start = headings[0].start() if headings else len(body)
    preamble = to_text(body[:first_start])
    if preamble:
        # Letters (e.g. PRL) often have no \section at all, so this can be the whole paper.
        sections.append(Section(path=["Main text"], text=preamble))

    for i, match in enumerate(headings):
        title_tex, content_start = read_braced(body, match.end() - 1)
        content_end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        level = levels[match.group(1)]
        title = to_text(title_tex) or "(untitled)"
        if level == 0 and 0 <= appendix_at < match.start():
            title = f"Appendix: {title}"
        path = path[:level] + [title]
        text = to_text(body[content_start:content_end].replace("\\appendix", ""))
        if text:
            sections.append(Section(path=list(path), text=text))
    return sections


def _attach_citing_paragraphs(sections: list[Section], figures: list[Figure]) -> None:
    paragraphs = [p for s in sections for p in s.text.split("\n\n")]
    for figure in figures:
        if not figure.label:
            continue
        marker = REF_MARKER.format(figure.label)
        figure.citing_paragraphs = [p for p in paragraphs if marker in p][:MAX_CITING_PARAGRAPHS]


def parse_latex_tree(root: Path) -> ParsedPaper:
    main = find_main_file(root)
    if main is None:
        raise ValueError("No LaTeX main file (with \\documentclass) found")

    tex = flatten_inputs(strip_comments(read_tex_file(main)), main.parent)
    graphics_paths = re.findall(r"\{([^{}]+)\}", _first_arg(tex, "graphicspath") or "")

    begin = tex.find("\\begin{document}")
    end = tex.find("\\end{document}")
    preamble = tex[:begin] if begin >= 0 else ""
    body = tex[begin + len("\\begin{document}") : end if end > begin else len(tex)] if begin >= 0 else tex

    body = expand_macros(body, collect_simple_macros(preamble + body))
    body = remove_environments(body, DROP_ENVIRONMENTS)
    body = remove_commands(body, FRONT_MATTER_COMMANDS)
    body, figures = _extract_figures(body)
    body = _prepare_references(body)

    sections = _split_sections(body)
    _attach_citing_paragraphs(sections, figures)
    return ParsedPaper(root=main.parent, sections=sections, figures=figures, graphics_paths=graphics_paths)
