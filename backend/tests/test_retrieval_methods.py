import json
from dataclasses import asdict

import numpy as np
import pytest

from app.index import store
from app.index.lexical import BM25, tokenize
from app.index.store import Index, Record, rrf_order


def test_tokenize_keeps_identifiers_and_numbers_whole():
    assert tokenize("The GN-z11 slope is 0.778 for PKS 0903-57.") == ["gn-z11", "slope", "0.778", "pks", "0903-57"]


def test_bm25_prefers_rare_matching_terms_and_ignores_unknown_ones():
    bm25 = BM25(["galaxy clusters lensing", "galaxy GN-z11 accretion", "galaxy survey galaxy"])
    scores = bm25.scores("GN-z11 galaxy")
    assert int(np.argmax(scores)) == 1
    assert bm25.scores("unseenword").sum() == 0


def test_rrf_rewards_agreement_between_rankings():
    order, scores = rrf_order([np.array([1, 2, 3]), np.array([3, 2, 1]), np.array([2, 9])])
    assert order[0] == 2  # ranked 2nd, 2nd and 1st beats any single first place
    assert list(scores) == sorted(scores, reverse=True)


def _write_index(tmp_path, records, embeddings):
    np.save(tmp_path / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    with open(tmp_path / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


@pytest.fixture
def toy_index(tmp_path, monkeypatch):
    records = [
        Record("text", "A", "Paper A", "Intro", "general cosmology overview"),
        Record("text", "A", "Paper A", "Results", "the GN-z11 slope measurement"),
        Record("text", "B", "Paper B", "Results", "weak lensing of clusters"),
        Record("figure", "A", "Paper A", "Figure", "slope versus mass", figure_id="fig1"),
    ]
    # Dense favours row 0, keywords favour row 1.
    _write_index(tmp_path, records, [[1, 0], [0.6, 0.8], [0, 1], [0.5, 0.5]])
    monkeypatch.setattr(store, "embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))
    return tmp_path


def test_dense_bm25_and_hybrid_rank_differently(toy_index):
    query = "GN-z11 slope"
    first = {m: Index(toy_index, method=m).search(query, 3, 1)[0][0].record.section for m in store.METHODS}
    assert first["dense"] == "Intro"
    assert first["bm25"] == "Results"
    text, figures = Index(toy_index, method="hybrid").search(query, 3, 1)
    # Each of the two Paper A chunks is 1st in one ranking and 2nd in the other; Paper B matches neither.
    assert {h.record.section for h in text[:2]} == {"Intro", "Results"}
    assert text[2].record.arxiv_id == "B"
    assert figures[0].record.figure_id == "fig1"


def test_reranker_reorders_candidates(toy_index):
    class FakeReranker:
        def scores(self, query, passages):
            return np.array([float("lensing" in p) for p in passages])

    text, _ = Index(toy_index, method="dense", reranker=FakeReranker()).search("anything", 2, 1)
    assert text[0].record.text == "weak lensing of clusters"
    assert len(text) == 2


def test_unknown_method_is_rejected(toy_index):
    with pytest.raises(ValueError):
        Index(toy_index, method="magic")
