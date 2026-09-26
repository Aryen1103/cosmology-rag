# One container serving the API and the built web app on $PORT (default 8000).
#
#   docker build -t cosmology-rag .
#   docker run -p 8000:8000 --env-file backend/.env cosmology-rag
#
# The corpus is baked in: run ingest.py and build_index.py first so data/ exists.
# API keys are never part of the image; pass them at runtime.

FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

# CPU-only torch: the CUDA build is several GB and this only embeds short queries.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image (must match MODEL_NAME in app/index/embeddings.py),
# then forbid Hub access so startup never depends on the network.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
m = 'BAAI/bge-small-en-v1.5'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY backend/app ./app
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY data/index /app/data/index
COPY data/papers /app/data/papers

RUN useradd --create-home --uid 1000 app
USER app

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
