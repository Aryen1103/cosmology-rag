from app.answer.base import Source, marker_citations
from app.index.store import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, chunk_section, records_for_paper


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_section_is_one_chunk():
    assert chunk_section("A short section.") == ["A short section."]


def test_long_section_chunks_overlap_and_cover_everything():
    text = _words(1000)
    chunks = chunk_section(text)
    assert all(len(c.split()) <= CHUNK_WORDS for c in chunks)
    assert chunks[0].split()[-CHUNK_OVERLAP_WORDS:] == chunks[1].split()[:CHUNK_OVERLAP_WORDS]
    assert set(" ".join(chunks).split()) == set(text.split())


def test_chunks_prefer_paragraph_boundaries():
    text = _words(150, "a") + "\n\n" + _words(150, "b")
    first = chunk_section(text)[0].split()
    assert first[-1] == "a149"


def test_records_include_abstract_chunks_and_figures():
    paper = {
        "meta": {"arxiv_id": "2609.00001v1", "title": "T", "abstract": "Abs."},
        "sections": [{"path": ["Intro"], "text": "Some intro text."}],
        "figures": [{
            "figure_id": "fig1", "label": "fig:a", "caption": "Cap.",
            "citing_paragraphs": ["As shown in [[REF:fig:a]], it rises."],
            "images": [{"file": "figures/fig1_1.png"}, {"file": None}],
        }],
    }
    records = records_for_paper(paper)
    assert [r.kind for r in records] == ["text", "text", "figure"]
    fig = records[2]
    assert fig.image_files == ["figures/fig1_1.png"]
    assert "As shown in this figure" in fig.text and "[[REF" not in fig.text


def test_boilerplate_sections_are_not_indexed():
    paper = {
        "meta": {"arxiv_id": "x", "title": "T", "abstract": "Abs."},
        "sections": [
            {"path": ["Results"], "text": "Science."},
            {"path": ["Acknowledgements"], "text": "We thank our funders."},
            {"path": ["Data Availability"], "text": "Data on request."},
            {"path": ["Methods", "Software"], "text": "We used numpy."},
            {"path": ["Appendix: Author List"], "text": "A. Author; B. Author."},
        ],
        "figures": [],
    }
    assert [r.section for r in records_for_paper(paper)] == ["Abstract", "Results"]


def test_marker_citations_flags_unknown_sources():
    sources = [Source("S1", "text", "x", "t", "s", "a"), Source("F1", "figure", "x", "t", "s", "b")]
    citations, unknown = marker_citations("Rises [S1][F1], again [S1], and [S7].", sources)
    assert [c.source_id for c in citations] == ["S1", "F1"]
    assert unknown == ["S7"]


def test_marker_citations_reads_grouped_markers():
    sources = [Source(i, "text", "x", "t", "s", "a") for i in ("S1", "S2", "F1")]
    citations, unknown = marker_citations("Tight [S2, F1] and loose [S1; S9], not [see S1].", sources)
    assert [c.source_id for c in citations] == ["S2", "F1", "S1"]
    assert unknown == ["S9"]
