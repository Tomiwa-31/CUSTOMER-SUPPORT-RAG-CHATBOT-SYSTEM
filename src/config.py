# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "novabuy_support"

# ── Embedding ──────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
NORMALIZE_EMBEDDINGS = True#normalizing the chunks so they are all in th same scale [2.5, 8.3, 1.2, 6.7],[0.1, 0.3, 0.05, 0.2]
#correct mode#[0.28, 0.93, 0.13, 0.75] [0.27, 0.91, 0.14, 0.74]

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

# ── Retriever ──────────────────────────────────────────
SEARCH_TYPE = "similarity"
TOP_K = 5

# ── LLM ───────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# ── Hybrid Search ──────────────────────────────────────
BM25_TOP_K = 5
SEMANTIC_TOP_K = 5
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [BM25 weight, semantic weight]

# ── Conversation History ───────────────────────────────
MAX_HISTORY_LENGTH = 6  # number of messages to keep (3 exchanges)


### What `ENSEMBLE_WEIGHTS = [0.5, 0.5]` means:

#BM25 results     → 50% influence
#Semantic results → 50% influence
#You can tune this later — for example [0.3, 0.7] would trust semantic search more. But 50/50 is the right starting point.
#Add those to config.py and let's move to Step 4 — updating ingestion.py for BM25!