# 1. Create and enter your project folder
mkdir rag-pipeline && cd rag-pipeline

# 2. Initialize — this creates pyproject.toml automatically
uv init

# 3. Create your virtual environment
uv venv myvenv

# 4. Activate it
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 5. Add dependencies (uv updates pyproject.toml for you)
uv add langchain chromadb faiss-cpu
uv add langchain-groq python-dotenv langgraph
# 6. Open in VS Code
code .



L2
``
User query
    ↓
├── ChromaDB retriever  (semantic)  → top 5 chunks
├── BM25 retriever      (keyword)   → top 5 chunks
└── EnsembleRetriever combines and reranks both results
    ↓

create a rewriter.py file
test it : python -m src.rewriter

BM25Retriever contains:
├── all chunk texts        (the raw words)
├── word frequency table   (how often each word appears in each chunk)
└── k = 5                  (how many chunks to return at query time)

modifying retriever.py file by adding  ensemble method
python -m src.retriever
python -m src.chain


L3
Skip for now ❌
├── Hallucination detection
├── Ground truth evals
└── Confidence scoring

Do now ✅
├── Dockerfile
├── Deploy to Cloud Run
├── Secret Manager
└── Cloud Logging (basic)

create a looger .py file
update chain.py with log

what logs would look like on cloud run due to the structured logging(logger.py)
{"timestamp": "2025-03-14T10:23:01Z", "invocation_id": "a3f9b2c1", "event": "request_started", "level": "INFO", "question": "how long do i have to return something"}
{"timestamp": "2025-03-14T10:23:02Z", "invocation_id": "a3f9b2c1", "event": "query_rewritten", "level": "INFO", "original_query": "how long do i have to return something", "rewritten_query": "What is the return eligibility window?"}
{"timestamp": "2025-03-14T10:23:02Z", "invocation_id": "a3f9b2c1", "event": "chunks_retrieved", "level": "INFO", "chunk_count": 7, "sections": ["1. Return Eligibility Window", "Overview"]}
{"timestamp": "2025-03-14T10:23:03Z", "invocation_id": "a3f9b2c1", "event": "request_completed", "level": "INFO", "answer_length": 142, "latency_ms": 1823}

app.py
@asynccontextmanager:gives the function the ability to pause and resume

uv add fastapi uvicorn


Test it locally before Docker:
uvicorn app:app --reload --port 8080

uvicorn app:app --reload --port 8080
```

This is actually the correct workflow going forward:
```
First time / after deleting:   python -m src.ingestion → uvicorn app:app
Every restart after that:      uvicorn app:app  (loads existing chroma_db)

uvicorn app:app --reload --port 8080

test it with fast api auto-docs that has its own complete ui to test endpoint
http://localhost:8080/docs

after that create Dockerfile and .dockerignore 
open docker desktop or ensure its running(docker --version)
build docker image

Step 1 — Test the image locally first:
docker run -p 8080:8080 --env-file .env novabuy-support

```
✅ Docker image built
✅ Tested locally

⬜ Create GCP project
⬜ Enable Cloud Run API
⬜ Push image to Artifact Registry
⬜ Deploy to Cloud Run
⬜ Move secrets to Secret Manager












OVERALL LEARNNG CURVE
Docker + containerization      ← systems engineering
Cloud Run deployment           ← systems engineering  
Secret Manager                 ← systems engineering
Cloud Logging + Monitoring     ← systems engineering
Firestore integration          ← systems engineering
Vertex AI ADK                  ← multi-agent
GKE Autopilot                  ← systems engineering
Multi-agent routing            ← multi-agent