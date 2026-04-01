# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"


# ── L5 CHANGE: three collections instead of one ───────
COLLECTION_NAME = "novabuy_support"          # kept for reference
BILLING_COLLECTION = "billing_support"
TECHNICAL_COLLECTION = "technical_support"
ACCOUNT_COLLECTION = "account_support"

# ── L5 CHANGE: document to collection mapping ─────────
COLLECTION_DOC_MAP = {
    BILLING_COLLECTION: [
        "04_promotions_loyalty_program.md",
        "05_order_management_faq.md",
    ],
    TECHNICAL_COLLECTION: [
        "03_product_faq.md",
        "06_damaged_wrong_item_troubleshooting.md",
        "02_shipping_delivery_policy.md",
    ],
    ACCOUNT_COLLECTION: [
        "07_account_payments_security.md",
        "01_return_refund_policy.md",
    ],
}

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

# ── Conversation History/Memory ───────────────────────────────
MEMORY_COLLECTION = "chat_sessions"
MAX_HISTORY_LENGTH = 6  # number of messages to keep (3 exchanges)

# ── L5 CHANGE: router categories ──────────────────────
ROUTER_CATEGORIES = ["billing", "technical", "account", "general"]

# ── L5 CHANGE: specialized prompts per agent ──────────
BILLING_PROMPT = """You are a billing specialist for NovaBuy, an ecommerce store.
You handle payment, refund, promotions, loyalty points and order management questions.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that, let me escalate this to a human agent."
Do not make up answers.

Context:
{context}

"""

TECHNICAL_PROMPT = """You are a technical support specialist for NovaBuy, an ecommerce store.
You handle product compatibility, damaged items, shipping, tracking and troubleshooting questions.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that, let me escalate this to a human agent."
Do not make up answers.

Context:
{context}

"""

ACCOUNT_PROMPT = """You are an account security specialist for NovaBuy, an ecommerce store.
You handle login, password, privacy, personal data and account security questions.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that, let me escalate this to a human agent."
Do not make up answers.

Context:
{context}

"""

GENERAL_PROMPT = """You are a helpful customer support agent for NovaBuy, an ecommerce store.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that, let me escalate this to a human agent."
Do not make up answers.

Context:
{context}

"""


### What `ENSEMBLE_WEIGHTS = [0.5, 0.5]` means:

#BM25 results     → 50% influence
#Semantic results → 50% influence
#You can tune this later — for example [0.3, 0.7] would trust semantic search more. But 50/50 is the right starting point.
#Add those to config.py and let's move to Step 4 — updating ingestion.py for BM25!