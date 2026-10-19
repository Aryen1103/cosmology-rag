from app.eval.retrieval import (
    Question,
    bootstrap_ci,
    check_questions,
    first_relevant_rank,
    is_relevant,
    split_papers,
    summarise,
)
from app.index.store import Hit, Record


def _text(arxiv_id: str, text: str) -> Record:
    return Record("text", arxiv_id, "Title", "Results", text)


def _figure(arxiv_id: str, figure_id: str) -> Record:
    return Record("figure", arxiv_id, "Title", "Figure", "caption", figure_id=figure_id)


TEXT_Q = Question("q1", "What is H0?", "text", "A", evidence="we find H0 = 70")
FIG_Q = Question("q2", "Which plot?", "figure", "A", figure_id="fig2")


def test_text_relevance_matches_evidence_ignoring_case_and_whitespace():
    assert is_relevant(TEXT_Q, _text("A", "Overall,  We find\nH0 = 70 km/s."))
    assert not is_relevant(TEXT_Q, _text("B", "we find H0 = 70"))  # right words, wrong paper
    assert not is_relevant(TEXT_Q, _text("A", "we find H0 = 67"))
    assert not is_relevant(TEXT_Q, _figure("A", "fig2"))


def test_figure_relevance_needs_paper_and_figure_id():
    assert is_relevant(FIG_Q, _figure("A", "fig2"))
    assert not is_relevant(FIG_Q, _figure("A", "fig3"))
    assert not is_relevant(FIG_Q, _figure("B", "fig2"))


def test_oracle_ranks_first_and_null_never_hits():
    oracle = [Hit(_text("A", "we find H0 = 70"), 0.9), Hit(_text("B", "other"), 0.5)]
    null = [Hit(_text("B", "other"), 0.9), Hit(_text("C", "more"), 0.5)]
    assert first_relevant_rank(TEXT_Q, oracle) == 1
    assert first_relevant_rank(TEXT_Q, null) is None
    assert summarise([1, 1, 1], (1, 6), 20) == {"hit@1": 1.0, "hit@6": 1.0, "mrr@20": 1.0}
    assert summarise([None, None], (1, 6), 20) == {"hit@1": 0.0, "hit@6": 0.0, "mrr@20": 0.0}


def test_summarise_counts_ranks_beyond_k_as_misses():
    m = summarise([1, 4, None, 30], (1, 6), 20)
    assert m["hit@1"] == 0.25
    assert m["hit@6"] == 0.5
    assert m["mrr@20"] == (1 + 0.25) / 4  # rank 30 is past the MRR cutoff


def test_split_is_deterministic_and_stable_when_papers_are_added():
    ids = [f"p{i}" for i in range(40)]
    train, test = split_papers(ids, 10)
    assert len(test) == 10 and not set(train) & set(test)
    assert split_papers(list(reversed(ids)), 10) == (train, test)
    # A new paper displaces at most one old test paper, so eval/split.json (frozen) is the source of truth.
    _, test_more = split_papers(ids + ["new"], 10)
    assert len(set(test) - set(test_more)) <= 1


def test_check_questions_flags_unscorable_questions():
    records = [_text("A", "we find H0 = 70"), _text("B", "also we find H0 = 70"), _figure("A", "fig2")]
    questions = [
        TEXT_Q,
        FIG_Q,
        Question("q2", "dup id", "figure", "A", figure_id="fig2"),
        Question("q3", "missing gold", "figure", "A", figure_id="fig9"),
        Question("q4", "not in test split", "text", "Z", evidence="some three words"),
        Question("q5", "short quote", "text", "A", evidence="H0"),
    ]
    problems = "\n".join(check_questions(questions, records, test_ids={"A"}))
    assert "q1: evidence also appears in ['B']" in problems
    assert "q2: duplicate id" in problems
    assert "q3: gold not found" in problems
    assert "q4: paper Z is not in the test split" in problems
    assert "q5: text question needs an evidence quote" in problems
    assert check_questions([FIG_Q], records, test_ids={"A"}) == []


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([1.0] * 30 + [0.0] * 70)
    assert lo < 0.3 < hi
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
