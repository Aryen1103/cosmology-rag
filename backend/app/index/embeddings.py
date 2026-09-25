"""Local text embeddings with BAAI/bge-small-en-v1.5 via plain transformers.

Deliberately not `sentence-transformers`: it pulls in scikit-learn, whose
compiled extension is blocked by this machine's Application Control policy.
bge uses the [CLS] token as the sentence embedding, L2-normalised.
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
MAX_TOKENS = 512
BATCH_SIZE = 32
# bge's recommended instruction for queries in asymmetric (query -> passage) retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).eval()
    return _tokenizer, _model


def _encode(texts: list[str]) -> np.ndarray:
    import torch

    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    tokenizer, model = _load()
    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(
            texts[start : start + BATCH_SIZE],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            cls = model(**batch).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(cls, p=2, dim=1).numpy())
    return np.concatenate(out).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    return _encode(texts)


def embed_query(query: str) -> np.ndarray:
    return _encode([QUERY_INSTRUCTION + query])[0]
