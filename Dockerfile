FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install CPU-only torch FIRST before anything else
# This prevents sentence-transformers from pulling GPU version
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Now install everything else
RUN uv pip install --system --no-cache \
    chromadb \
    fastapi \
    uvicorn \
    langchain \
    langchain-chroma \
    langchain-community \
    langchain-groq \
    langchain-huggingface \
    langchain-text-splitters \
    rank-bm25 \
    sentence-transformers \
    python-dotenv \
    google-cloud-firestore \
    google-cloud-trace \
    opentelemetry-sdk \
    opentelemetry-exporter-gcp-trace \
    opentelemetry-api

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/
COPY bm25_index.pkl .
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]