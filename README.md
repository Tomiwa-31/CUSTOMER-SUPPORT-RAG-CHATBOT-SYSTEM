# NovaBuy Customer Support RAG System

> An enterprise-grade, multi-agent AI customer support system built with LangChain, Groq, ChromaDB, and Google Cloud Platform — progressing from a local prototype to a production-deployed multi-agent platform across 5 maturity levels.

---

## Table of Contents

- [Project Goals](#project-goals)
- [What The Project Solves](#what-the-project-solves)
- [Challenges Faced](#challenges-faced)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Tech Stack](#tech-stack)
- [Architecture Maturity Levels](#architecture-maturity-levels)
- [Setup Instructions](#setup-instructions)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Project Goals

The primary goals of this project were to:

1. Build a production-grade Retrieval Augmented Generation (RAG) system that answers customer support questions grounded in actual company policy documents — eliminating hallucinations
2. Demonstrate progressive system engineering maturity from a local notebook prototype (L1) to a fully deployed, observable, multi-agent platform on GCP (L5)
3. Develop practical skills in AI systems engineering, DevOps, cloud infrastructure, and multi-agent orchestration
4. Produce a portfolio piece that maps directly to enterprise AI engineering responsibilities

---

## What The Project Solves

Traditional LLM-powered chatbots have two critical production problems:

**Problem 1 — Hallucination:**
```
User: "How long do I have to return an item?"
Generic LLM: "You have 60 days to return any item." ← fabricated, wrong
NovaBuy RAG: "You have 30 days from delivery for unopened items,
              14 days for opened items, and 60 days for defective items." ← grounded in policy
```

**Problem 2 — No escalation path:**
```
Generic LLM: Confidently gives wrong answers with no fallback
NovaBuy RAG: Detects low confidence → creates Firestore ticket → 
             routes to human agent with full context
```

**What we built instead:**
- Every answer is grounded in retrieved policy documents — the LLM cannot fabricate
- Confidence scoring detects when retrieval quality is too low
- LLM-as-judge scores answer quality after generation
- Automatic escalation to human agents when quality thresholds are not met
- Persistent conversation memory per user across sessions

---

## Challenges Faced

### 1. Semantic Gap in Retrieval
**Problem:** Casual user queries like *"how long do i have 2 return somthing"* failed to match formal policy document language, causing the system to return irrelevant chunks.

**Solution:** Built a query rewriting layer using Groq that reformulates casual queries into formal policy language before retrieval.

### 2. Single Retrieval Method Was Insufficient
**Problem:** Pure semantic (vector) search missed exact keyword matches like "RMA number" while pure keyword search missed conceptual queries like "send something back."

**Solution:** Implemented hybrid retrieval combining BM25 keyword search and ChromaDB semantic search via LangChain's `EnsembleRetriever` with configurable weights.


### 3. Cold Start Latency
**Problem:** Cloud Run cold starts took 10–15 seconds due to loading sentence-transformers, ChromaDB, and building the RAG chain at startup.

**Solution:** Pre-embedded vectors are baked into the Docker image (no re-embedding on cold start). The `--timeout 300` flag prevents false deployment failures from gcloud CLI timing out before the container finishes starting.

---

## Key Features

### Retrieval
- **Hybrid BM25 + Semantic Search** — `EnsembleRetriever` combines keyword and vector search with configurable weights
- **Query Rewriting** — Groq LLM reformulates casual/misspelled queries before retrieval
- **Structure-Aware Chunking** — `MarkdownHeaderTextSplitter` preserves `##` section context in every chunk

### Multi-Agent System
- **Triage Router** — LLM-powered ticket classifier routes to billing, technical, account, or general agent
- **Specialized Agents** — Each agent has its own ChromaDB collection, BM25 index, and domain-specific prompt personality
- **Orchestrator** — Drop-in chain replacement with `.invoke()` interface for seamless integration with memory

### Quality & Reliability
- **Confidence Scoring** — Retrieval similarity scores flag low-confidence retrievals
- **LLM-as-Judge** — Second Groq call scores grounding, relevance, and completeness (1–5)
- **Fallback Detection** — Pattern matching detects "I don't know" responses before they reach users
- **Automatic Escalation** — Creates structured Firestore tickets with priority levels when quality thresholds are not met

### Memory
- **Firestore Persistent Memory** — Conversation history survives container restarts and scales across multiple Cloud Run instances
- **Session Isolation** — Each `session_id` gets its own Firestore document
- **History Trimming** — `MAX_HISTORY_LENGTH` prevents context window overflow

### Observability
- **Structured JSON Logging** — Every request logs `invocation_id`, `event`, `latency_ms`, `agent_type` as searchable Cloud Logging JSON
- **Cloud Trace** — Span-level breakdown of `query_rewriting`, `hybrid_retrieval`, and `llm_call` timing
- **Cloud Monitoring** — Request count, p50/p95/p99 latency, memory and CPU utilization dashboards
- **Budget Alerts** — GCP spend alerts configured to prevent cost surprises

### Security
- **Secret Manager** — `GROQ_API_KEY` stored in GCP Secret Manager; never hardcoded or passed as environment variables in deploy commands
- **IAM Service Accounts** — Cloud Run accesses Firestore and Cloud Trace via attached service account with least-privilege roles
- **Zero Secrets in Image** — `.env` excluded from Docker image via `.dockerignore`

---

## System Workflow

```
User Request (POST /chat)
         │
         ▼
   RequestLogger
   (invocation_id, timestamp)
         │
         ▼
  Load Session History
   (Firestore → chat_history list)
         │
         ▼
   Triage Router
   (Groq LLM classifies: billing | technical | account | general)
         │
         ▼
  Specialized Agent
  ┌─────────────────────────────────────────┐
  │  Query Rewriter (Groq)                  │
  │         ↓                               │
  │  Hybrid Retriever                       │
  │  ├── BM25 (keyword search)              │
  │  └── Pinecone (semantic search)         │
  │         ↓                               │
  │  EnsembleRetriever (0.5 / 0.5 weights)  │
  │         ↓                               │
  │  Groq LLM + Specialized Prompt          │
  │         ↓                               │
  │  Answer string                          │
  └─────────────────────────────────────────┘
         │
         ▼
   Evaluator
   ├── Fallback detection (pattern match)
   ├── Retrieval confidence score
   └── LLM-as-judge (grounding, relevance, completeness)
         │
         ▼
   Escalation decision?
   ├── YES → Create Firestore ticket (TKT-XXXXXXXX)
   │          Return friendly escalation message
   └── NO  → Return answer
         │
         ▼
   Save Session History
   (append HumanMessage + AIMessage → Firestore)
         │
         ▼
   Structured Log
   (Cloud Logging: event, latency_ms, agent_type, confidence_score)
         │
         ▼
   ChatResponse (JSON)
   {session_id, question, answer, invocation_id,
    latency_ms, confidence_score, escalated, ticket_id, agent_type}
```

---

## Tech Stack

### AI & Retrieval
| Component | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, CPU) |
| Vector store | Pinecone (managed cloud) |
| Keyword search | BM25 via `rank-bm25` |
| Hybrid retrieval | LangChain `EnsembleRetriever` |
| Agent framework | LangChain |
| Query rewriting | Groq LLM chain |
| LLM-as-judge | Groq LLM chain |
| GCS (BM25 pickle storage)  | PostgreSQL (DB source) |


### Backend & API
| Component | Technology |
|---|---|
| REST API | FastAPI |
| ASGI server | Uvicorn |
| Request validation | Pydantic |
| Conversation memory | Firestore (Google Cloud) |
| Escalation tickets | Firestore (Google Cloud) |

### Infrastructure & DevOps
| Component | Technology |
|---|---|
| Containerization | Docker (CPU-only torch) |
| Image registry | GCP Artifact Registry |
| Build pipeline | GCP Cloud Build |
| Hosting | GCP Cloud Run (serverless) |
| Secret management | GCP Secret Manager |
| Structured logging | GCP Cloud Logging |
| Distributed tracing | GCP Cloud Trace + OpenTelemetry |
| Monitoring | GCP Cloud Monitoring |

### Python Tooling
| Component | Technology |
|---|---|
| Package manager | `uv` |
| Environment | Python 3.14 |
| Config management | `python-dotenv` |

---

## Architecture Maturity Levels

```
L1 — Local Prototype ✅
     Basic RAG pipeline · Pinecone · Groq · Local terminal

L2 — Smart Retrieval ✅
     Query rewriting · Hybrid BM25 + semantic · Conversation history

L3 — Production Deployment ✅
     FastAPI REST API · Docker · Cloud Run · Secret Manager
     Structured logging · Cloud Monitoring · Budget alerts

L4 — Memory & Personalization ✅
     Firestore persistent memory · Cloud Trace instrumentation
     Sessions survive container restarts and scale across instances

L5 — Multi-Agent Platform ✅
     Triage router · Billing / Technical / Account specialized agents
     LLM-as-judge eval · Automatic escalation · Firestore tickets
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- `uv` package manager (`pip install uv`)
- Docker Desktop
- GCP account with billing enabled
- Groq API key (free at `console.groq.com`)

---

### Local Development

**1. Clone the repository:**
```bash
git clone https://github.com/Tomiwa-31/CUSTOMER-SUPPORT-RAG-CHATBOT-SYSTEM
.git
cd CUSTOMER-SUPPORT-RAG-CHATBOT-SYSTEM

```

**2. Create virtual environment and install dependencies:**
```bash
uv venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux
uv sync
```

**3. Create `.env` file:**
```bash
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key
DATABASE_URL=your_databse_url
GCS_BUCKET_NAME=your_gcs_bucket_name

```

**4. Run ingestion (first time only):**
```bash
python -m src.ingestion
```
Vectors goes to pinecone and BM25 pickles goes to GCS.

**5. Run the API locally:**
```bash
uvicorn app:app --reload --port 8080
```

**6. Test the API:**
```
http://localhost:8080/docs
```

---


