import json
import random

import numpy as np
import pytest

from app.finetune.pairs import build_pairs, pair_for_record, sentences, usable_query
from app.index import embeddings as emb
from app.index import store
from app.index.store import Index, Record, build_index

BODY = ("We measure the clustering of luminous red galaxies across three redshift bins in the survey. "
        "The acoustic peak is detected at high significance in every bin of the sample. "
        "Systematic weights for stellar density are applied before the correlation function is estimated. ") * 3


def _paper(arxiv_id: str) -> dict:
    return {
        "meta": {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "abstract": "Short abstract."},
        "sections": [{"path": ["Results"], "text": BODY}],
        "figures": [{
            "figure_id": "fig1", "label": "fig:a",
            "caption": "The measured correlation function of the luminous red galaxy sample in three bins. Errors are jackknife.",
            "citing_paragraphs": [BODY],
            "images": [{"file": "figures/fig1_1.png"}],
        }],
    }


def _write_papers(tmp_path, ids):
    for i in ids:
        (tmp_path / i).mkdir()
        (tmp_path / i / "paper.json").write_text(json.dumps(_paper(i)), encoding="utf-8")


def test_sentence_filter_skips_math_references_and_short_sentences():
    assert sentences("First one here. Second (two) here. third stays joined.") == [
        "First one here.", "Second (two) here. third stays joined."]
    assert usable_query("The acoustic peak is detected at high significance in every bin.")
    assert not usable_query("Too short.")
    assert not usable_query("As shown in [[REF:fig:a]] the peak is detected at high significance here.")
    assert not usable_query("We find $a$ and $b$ and $c$ with the peak detected at high significance.")


def test_query_is_removed_from_its_passage():
    record = Record("text", "A", "Paper A", "Results", BODY)
    pair = pair_for_record(record, random.Random(0))
    assert pair.kind == "text"
    assert pair.passage.startswith("Paper A | Results\n")
    assert BODY.count(pair.query) == 3 and pair.passage.count(pair.query) == 2  # one copy removed


def test_figure_pair_uses_first_caption_sentence():
    fig = store.records_for_paper(_paper("A"))[-1]
    pair = pair_for_record(fig, random.Random(0))
    assert pair.kind == "figure"
    assert pair.query.startswith("The measured correlation function")
    assert pair.query not in pair.passage and "Discussed in the text" in pair.passage


def test_pairs_come_only_from_train_papers(tmp_path):
    _write_papers(tmp_path, ["train1", "train2", "test1"])
    pairs = build_pairs(tmp_path, {"train1", "train2"}, {"test1"})
    assert pairs and {p.arxiv_id for p in pairs} == {"train1", "train2"}
    assert {p.kind for p in pairs} == {"text", "figure"}
    with pytest.raises(ValueError):
        build_pairs(tmp_path, {"train1", "test1"}, {"test1"})


def test_other_models_get_their_own_cache_and_the_index_remembers_its_model(tmp_path, monkeypatch):
    papers, index_dir = tmp_path / "papers", tmp_path / "index"
    papers.mkdir()
    _write_papers(papers, ["A", "B"])
    monkeypatch.setattr(store, "embed_passages", lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    try:
        build_index(papers, index_dir, only={"A"})
        assert (papers / "A" / "embeddings.npy").exists() and not (papers / "B" / "embeddings.npy").exists()
        emb.use_model("data/models/my-ft")
        build_index(papers, index_dir, only={"A"})
        cached = sorted(p.name for p in (papers / "A").glob("embeddings*.npy"))
        assert cached[0] == "embeddings-my-ft-" + cached[0].split("-")[-1] and "embeddings.npy" in cached
        assert json.loads((index_dir / "meta.json").read_text())["embedding_model"] == "data/models/my-ft"
        emb.use_model(emb.DEFAULT_MODEL)
        assert Index(index_dir).embedding_model == "data/models/my-ft" == emb.MODEL_NAME
    finally:
        emb.use_model(emb.DEFAULT_MODEL)
