from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.evaluator import evaluate_response
from src.orchestrator import Orchestrator
from src.orchestrator import build_orchestrator   
from contextlib import asynccontextmanager
from src.chain import run_with_memory  # ── L4: trim_history removed, run_with_memory added
from src.logger import RequestLogger
from src.config import MAX_HISTORY_LENGTH
from src.config import USE_LLM_JUDGE
from src.escalation import handle_escalation
import tracer
 

 
state = {}
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('{"event": "startup", "status": "building_orchestrator"}')
    state["chain"] = build_orchestrator()
    print('{"event": "startup", "status": "ready"}')
    yield
    state.clear()
 
app = FastAPI(
    title="NovaBuy Customer Support API",
    version="1.0.0",
    lifespan=lifespan
)
 
 
# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
 
class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    invocation_id: str
    latency_ms: int
    confidence_score: float = 1.0    
    escalated: bool = False           
    ticket_id: str = None             
    agent_type: str = "general"   
 
 

 
 
# ── Routes 
@app.get("/health")
def health():
    return {"status": "ok", "service": "novabuy-support"}
 
 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    with tracer.start_as_current_span("chat_request"):
        logger = RequestLogger()
        logger.info("request_received",
            session_id=request.session_id,
            question=request.question
        )
 
        chain = state.get("chain")
        if not chain:
            raise HTTPException(status_code=503, detail="Chain not ready")
 
        try:
        
            response = run_with_memory(
                chain=chain,
                session_id=request.session_id,
                question=request.question,
                logger=logger,
            )
        # ─────────────────────────────────────────────────────────────────────
 
            logger.info("request_completed",
                session_id=request.session_id,
                answer_length=len(response),
                latency_ms=logger.latency_ms()
            )

            eval_result = evaluate_response(
                question=request.question, 
                context="",
                answer=response,    
            )
            if eval_result["should_escalate"]:
                escalation = handle_escalation(
                    session_id=request.session_id,       
                    question=request.question,           
                    answer=response,                     
                    escalation_reason=eval_result["escalation_reason"],  
                    agent_type=orchestrator.last_category,  
                    confidence_score=eval_result["confidence_score"],    
                    llm_judge_scores=eval_result["llm_judge_scores"],    
                    logger=logger,                       
                )
 
                return ChatResponse(
                    session_id=request.session_id,
                    question=request.question,
                    answer=response,
                    invocation_id=logger.invocation_id,
                    latency_ms=logger.latency_ms(),
                    confidence_score=eval_result["confidence_score"],
                    escalated=True,
                    ticket_id=escalation["ticket_id"],
                    agent_type=orchestrator.last_category,
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
 
 
# ✅ FastAPI endpoint live
# ✅ Session history now persists in Firestore across restarts
# ✅ Invocation ID on every request
# ✅ Latency tracked
# ✅ In-memory session dict gone — Cloud Run can now scale to multiple instances
#    without sessions living on just one instance
 
 
