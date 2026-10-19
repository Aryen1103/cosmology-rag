"""BM25 keyword search, in plain Python + numpy (no compiled extras on this machine).

Dense embeddings blur exact tokens such as object names (GN-z11, PKS 0903-57),
survey names and numbers; BM25 matches them literally.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

K1 = 1.2
B = 0.75
# Keep decimals and hyphenated identifiers whole: "0.778", "gn-z11", "0903-57".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in into is it its of on or "
    "that the their them then there these this those to was were what when where which while "
    "who why will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: list[str]) -> None:
        tokenized = [tokenize(d) for d in docs]
        self.n_docs = len(tokenized)
        lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        avg = float(lengths.mean()) if self.n_docs else 1.0
        # Per-doc length normalisation, folded in once: K1 * (1 - B + B * len / avg).
        self._norm = K1 * (1 - B + B * lengths / max(avg, 1e-9))
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                tfs.append(tf)
        self._postings = {
            term: (np.array(ids), np.array(tfs, dtype=np.float32)) for term, (ids, tfs) in postings.items()
        }

    def _idf(self, df: int) -> float:
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            ids, tfs = self._postings[term]
            out[ids] += self._idf(len(ids)) * tfs * (K1 + 1) / (tfs + self._norm[ids])
        return out
