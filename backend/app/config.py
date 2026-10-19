from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
ENV_FILE = BACKEND_DIR / ".env"
DATA_DIR = REPO_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"

load_dotenv(ENV_FILE)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# Retrieval: "dense", "bm25" or "hybrid" (see eval/README.md for how they compare), and an
# optional cross-encoder reranker ("minilm" or "bge-base"; off by default because it is slow on CPU).
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "hybrid")
RERANKER = os.getenv("RERANKER") or None


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set - add it to {ENV_FILE}")
    return value
