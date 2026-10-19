"""Cross-encoder reranking of retrieved candidates.

A bi-encoder embeds question and passage separately; a cross-encoder reads them
together and scores the pair, which is much better at telling apart passages
from the same paper. It is too slow to run on the whole index, so it only
reorders the top candidates. Plain transformers, loaded lazily (no torch at
import time, same as embeddings.py).
"""
from __future__ import annotations

import numpy as np

RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~90 MB, fast, trained on web search
    "bge-base": "BAAI/bge-reranker-base",  # ~1.1 GB, stronger, slower on CPU
}
MAX_TOKENS = 512
BATCH_SIZE = 16


class Reranker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model_name = RERANKERS[name]
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        return self._tokenizer, self._model

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        import torch

        if not passages:
            return np.zeros(0, dtype=np.float32)
        tokenizer, model = self._load()
        out = []
        for start in range(0, len(passages), BATCH_SIZE):
            batch = passages[start : start + BATCH_SIZE]
            inputs = tokenizer([query] * len(batch), batch, padding=True, truncation="only_second",
                               max_length=MAX_TOKENS, return_tensors="pt")
            with torch.no_grad():
                out.append(model(**inputs).logits[:, 0].float().numpy())
        return np.concatenate(out)
