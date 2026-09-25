import gzip
import io
import tarfile

import pymupdf
from PIL import Image

from app.ingest.figures import STATUS_OK, STATUS_UNSUPPORTED, convert_to_png, resolve_graphic
from app.ingest.latex import expand_macros, collect_simple_macros, parse_latex_tree, read_braced
from app.ingest.pipeline import SOURCE_LATEX, SOURCE_PDF, unpack_source

MAIN_TEX = r"""
\documentclass{aastex631}
\newcommand{\hmpc}{h^{-1}\,{\rm Mpc}}
\newcommand{\vect}[1]{\mathbf{#1}}
\graphicspath{{figs/}}
\begin{document}
\title{A Test Paper}
\author{Someone}
\affiliation{Some University}
\begin{abstract}Abstract text that metadata already provides.\end{abstract}
\maketitle
\input{sections/intro}
\section{Results}
Clustering on scales of 10 \hmpc{} is shown in Figure~\ref{fig:pk}. % a comment \ref{fig:hidden}
We use $\sigma_8 = 0.81$ from \citep{Planck2018, DESI2024}.

Unrelated paragraph.
\begin{figure}
  \includegraphics[width=0.5\textwidth]{power_spectrum}
  \caption{The matter power spectrum $P(k)$ with {nested {braces}} at 10 \hmpc.}
  \label{fig:pk}
\end{figure}
\appendix
\section{Extra}
Appendix content.
\begin{thebibliography}{}
\bibitem{Planck2018} Planck Collaboration.
\end{thebibliography}
\end{document}
"""

INTRO_TEX = r"""
\section{Introduction}
\subsection{Motivation}
Dark energy motivates this work.
"""


def _write_tree(root):
    (root / "sections").mkdir()
    (root / "figs").mkdir()
    (root / "main.tex").write_text(MAIN_TEX)
    (root / "sections" / "intro.tex").write_text(INTRO_TEX)
    Image.new("RGB", (40, 30), "blue").save(root / "figs" / "power_spectrum.png")


def test_read_braced_handles_nesting_and_escapes():
    s = r"\caption{a {b {c}} \} d} tail"
    content, end = read_braced(s, s.index("{"))
    assert content == r"a {b {c}} \} d"
    assert s[end:] == " tail"


def test_expand_macros_skips_macros_with_arguments():
    macros = collect_simple_macros(r"\newcommand{\hmpc}{h^{-1}\,{\rm Mpc}}\newcommand{\vect}[1]{\mathbf{#1}}")
    assert set(macros) == {"hmpc"}
    assert expand_macros(r"10 \hmpc and \hmpcx", macros) == r"10 h^{-1}\,{\rm Mpc} and \hmpcx"


def test_parse_latex_tree_sections(tmp_path):
    _write_tree(tmp_path)
    parsed = parse_latex_tree(tmp_path)
    paths = [s.path for s in parsed.sections]
    assert ["Introduction", "Motivation"] in paths
    assert ["Results"] in paths
    assert ["Appendix: Extra"] in paths

    all_text = "\n".join(s.text for s in parsed.sections)
    assert "Some University" not in all_text
    assert "Abstract text" not in all_text
    assert "Planck Collaboration" not in all_text
    assert "fig:hidden" not in all_text

    results = next(s.text for s in parsed.sections if s.path == ["Results"])
    assert "10 h^-1 Mpc" in results
    assert "\r" not in results
    assert r"$\sigma_8 = 0.81$" in results
    assert "[cite: Planck2018, DESI2024]" in results


def test_parse_latex_tree_figures(tmp_path):
    _write_tree(tmp_path)
    parsed = parse_latex_tree(tmp_path)
    assert len(parsed.figures) == 1
    fig = parsed.figures[0]
    assert fig.label == "fig:pk"
    assert fig.graphics == ["power_spectrum"]
    assert "$P(k)$" in fig.caption and "nested braces" in fig.caption
    assert "10 h^-1 Mpc" in fig.caption
    assert len(fig.citing_paragraphs) == 1
    assert "Clustering on scales" in fig.citing_paragraphs[0]
    assert parsed.graphics_paths == ["figs/"]


def test_resolve_and_convert_figures(tmp_path):
    _write_tree(tmp_path)
    src = resolve_graphic(tmp_path, "power_spectrum", ["figs/"])
    assert src == tmp_path / "figs" / "power_spectrum.png"
    assert convert_to_png(src, tmp_path / "out.png") == STATUS_OK

    doc = pymupdf.open()
    doc.new_page(width=200, height=100)
    doc.save(tmp_path / "vector.pdf")
    assert convert_to_png(tmp_path / "vector.pdf", tmp_path / "vector.png") == STATUS_OK
    with Image.open(tmp_path / "vector.png") as img:
        assert max(img.size) <= 1568

    (tmp_path / "old.eps").write_text("%!PS")
    assert convert_to_png(tmp_path / "old.eps", tmp_path / "old.png") == STATUS_UNSUPPORTED


def test_unpack_source_variants(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = MAIN_TEX.encode()
        info = tarfile.TarInfo("main.tex")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (tmp_path / "a").mkdir()
    assert unpack_source(buf.getvalue(), tmp_path / "a") == SOURCE_LATEX
    assert (tmp_path / "a" / "main.tex").exists()

    (tmp_path / "b").mkdir()
    assert unpack_source(gzip.compress(MAIN_TEX.encode()), tmp_path / "b") == SOURCE_LATEX
    assert (tmp_path / "b" / "main.tex").exists()

    (tmp_path / "c").mkdir()
    assert unpack_source(b"%PDF-1.7 fake", tmp_path / "c") == SOURCE_PDF


def test_inline_tikz_figure_is_flagged(tmp_path):
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\begin{document}\section{A}Text."
        r"\begin{figure}\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}"
        r"\caption{A cartoon.}\label{fig:c}\end{figure}\end{document}"
    )
    fig = parse_latex_tree(tmp_path).figures[0]
    assert fig.inline_drawing and fig.graphics == [] and fig.caption == "A cartoon."
