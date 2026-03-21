from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from src.chain import build_rag_chain, trim_history
from src.logger import RequestLogger
from src.config import MAX_HISTORY_LENGTH

# Store chain and sessions in memory
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build chain once at startup
    print('{"event": "startup", "status": "building_chain"}')
    state["chain"] = build_rag_chain()
    print('{"event": "startup", "status": "ready"}')
    yield # PAUSE, go run the app
    state.clear() #once the app is shutdown, clear the state

app = FastAPI(
    title="NovaBuy Customer Support API",
    version="1.0.0",
    lifespan=lifespan#added lifespan because we want to build the chain once at startup and reuse it for all requests
)

# ── Request / Response models ──────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    invocation_id: str
    latency_ms: int


# ── In-memory session store ────────────────────────────────
# At L4 this gets replaced with Firestore
sessions: dict = {}

# ── Routes ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "novabuy-support"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger = RequestLogger()#start a new log
    logger.info("request_received",
        session_id=request.session_id,
        question=request.question
    )

    chain = state.get("chain")
    if not chain:
        raise HTTPException(status_code=503, detail="Chain not ready")

    # Get or create session history
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    chat_history = sessions[request.session_id]

    try:
        response = chain.invoke({
            "question": request.question,
            "chat_history": trim_history(chat_history, MAX_HISTORY_LENGTH),
            "logger": logger
        })

        # Update session history
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=response))

        logger.info("request_completed",
            session_id=request.session_id,
            answer_length=len(response),
            latency_ms=logger.latency_ms()
        )

        return ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=response,
            invocation_id=logger.invocation_id,
            latency_ms=logger.latency_ms()
        )

    except Exception as e:
        logger.error("request_failed",
            session_id=request.session_id,
            error=str(e),
            latency_ms=logger.latency_ms()
        )
        raise HTTPException(status_code=500, detail=str(e))


#✅ FastAPI endpoint live on localhost
#✅ Session history working
#✅ Invocation ID on every request
#✅ Latency tracked
