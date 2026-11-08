import pytest

from app.eval.citation import (
    CitationQuestion, GoldItem, cited_correctly, parse_judgment, score_citations,
)
from app.index.store import Record

TEXT_A = Record("text", "A", "Paper A", "Results", "We infer H0 = 71.6 from the discs.")
TEXT_A2 = Record("text", "A", "Paper A", "Intro", "Background on megamasers.")
TEXT_B = Record("text", "B", "Paper B", "Results", "Unrelated passage.")
FIG_A = Record("figure", "A", "Paper A", "Figure", "Caption.", figure_id="fig5")
SOURCES = {"S1": TEXT_A, "S2": TEXT_A2, "S3": TEXT_B, "F1": FIG_A}

TEXT_Q = CitationQuestion("t", "simple", "H0?", True, False, [GoldItem("A", evidence="we infer H0 = 71.6")])
FIG_Q = CitationQuestion("f", "medium", "Figure?", True, True, [GoldItem("A", figure_id="fig5")])


def test_right_paper_wrong_passage_fails_only_the_section_item():
    row = score_citations(TEXT_Q, SOURCES, {"S2"})
    assert (row["correct_paper"], row["correct_section"], row["figure_referenced"]) == (True, False, None)
    assert cited_correctly(row) is False
    assert cited_correctly(score_citations(TEXT_Q, SOURCES, {"S1", "S3"})) is True


def test_figure_question_needs_the_figure_cited():
    assert score_citations(FIG_Q, SOURCES, {"S1"})["figure_referenced"] is False
    row = score_citations(FIG_Q, SOURCES, {"F1"})
    assert row["figure_referenced"] is True and cited_correctly(row) is True


def test_gold_not_retrieved_is_excluded_not_failed():
    missing = CitationQuestion("m", "simple", "?", True, False, [GoldItem("Z", evidence="x y z")])
    row = score_citations(missing, SOURCES, {"S1"})
    assert row["gold_papers_retrieved"] == 0 and cited_correctly(row) is None


def test_multi_paper_question_needs_every_retrieved_gold_paper():
    q = CitationQuestion("c", "complex", "?", True, False,
                         [GoldItem("A", evidence="we infer H0 = 71.6"), GoldItem("B", evidence="unrelated passage")])
    assert score_citations(q, SOURCES, {"S1"})["correct_paper"] is False
    assert cited_correctly(score_citations(q, SOURCES, {"S1", "S3"})) is True


def test_unanswerable_question_has_no_citation_items():
    q = CitationQuestion("u", "complex", "?", False, False, [])
    assert cited_correctly(score_citations(q, SOURCES, {"S1"})) is None


def test_unknown_labels_are_ignored():
    assert score_citations(TEXT_Q, SOURCES, {"S9"})["correct_paper"] is False


def test_parse_judgment_counts_unsupported_claims():
    text = 'Here: {"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}], "declines_to_answer": false}'
    j = parse_judgment(text)
    assert (j["n_claims"], j["n_unsupported"], j["unsupported"], j["declines"]) == (2, 1, ["b"], False)
    with pytest.raises(ValueError):
        parse_judgment("no json here")
