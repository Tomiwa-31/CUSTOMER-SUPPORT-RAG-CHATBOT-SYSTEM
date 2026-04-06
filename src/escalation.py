# src/escalation.py

import uuid
from datetime import datetime
from google.cloud import firestore
from src.config import GROQ_API_KEY
from src.logger import RequestLogger


# ── Escalation collection in Firestore ────────────────
ESCALATION_COLLECTION = "escalation_tickets"


def generate_ticket_id() -> str:
    """Generate a unique ticket ID"""
    return f"TKT-{str(uuid.uuid4())[:8].upper()}"


def create_escalation_ticket(
    session_id: str,
    question: str,
    answer: str,
    escalation_reason: str,
    agent_type: str,
    confidence_score: float,
    llm_judge_scores: dict,
    logger: RequestLogger = None,
) -> dict:
    """
    Creates an escalation ticket in Firestore
    Returns ticket details
    """
    ticket_id = generate_ticket_id()
    timestamp = datetime.utcnow().isoformat() + "Z"


    #creates a ticket object that helps the human understand the issue
    ticket = {
        "ticket_id": ticket_id,
        "session_id": session_id,
        "status": "open",
        "priority": get_priority(escalation_reason),
        "created_at": timestamp,
        "question": question,
        "ai_answer": answer,
        "escalation_reason": escalation_reason,
        "agent_type": agent_type,
        "confidence_score": confidence_score,
        "llm_judge_scores": llm_judge_scores,
        "resolved_at": None,
        "resolution_notes": None,
    }

    try:
        db = firestore.Client()
        db.collection(ESCALATION_COLLECTION).document(ticket_id).set(ticket)

        if logger:
            logger.info("escalation_ticket_created",
                ticket_id=ticket_id,
                session_id=session_id,
                escalation_reason=escalation_reason,
                agent_type=agent_type,
                priority=ticket["priority"],
            )

        print(f"🎫 Escalation ticket created: {ticket_id} "
              f"(reason: {escalation_reason}, priority: {ticket['priority']})")

    except Exception as e:
        print(f"⚠️ Failed to create escalation ticket: {e}")

    return ticket


def get_priority(escalation_reason: str) -> str:
    """
    Assigns priority based on escalation reason
    """
    priority_map = {
        "fallback_response":          "medium",
        "low_retrieval_confidence":   "medium",
        "low_llm_judge_score":        "medium",
        "user_requested":             "high",
    }
    return priority_map.get(escalation_reason, "medium")


def build_escalation_response(ticket: dict) -> str:
    """
    Returns a friendly escalation message to show the user
    """
    return (
        f"I wasn't able to fully answer your question with confidence. "
        f"I've created a support ticket ({ticket['ticket_id']}) and a human agent "
        f"will follow up with you shortly. "
        f"In the meantime, you can reach us directly at support@novabuy.com."
    )


def handle_escalation(
    session_id: str,
    question: str,
    answer: str,
    escalation_reason: str,
    agent_type: str,
    confidence_score: float,
    llm_judge_scores: dict,
    logger: RequestLogger = None,
) -> dict:
    """
    Master escalation handler:
    1. Creates Firestore ticket
    2. Logs the escalation
    3. Returns escalation response to user

    Returns:
    {
        "ticket_id": str,
        "priority": str,
        "message": str,
        "escalated": True
    }
    """
    ticket = create_escalation_ticket(
        session_id=session_id,
        question=question,
        answer=answer,
        escalation_reason=escalation_reason,
        agent_type=agent_type,
        confidence_score=confidence_score,
        llm_judge_scores=llm_judge_scores,
        logger=logger,
    )

    return {
        "ticket_id": ticket["ticket_id"],
        "priority": ticket["priority"],
        "message": build_escalation_response(ticket),
        "escalated": True,
    }


#user_requested            → high    ← user explicitly asked for human
#fallback_response         → medium  ← AI said I don't know
#low_llm_judge_score       → medium  ← answer quality too low
#low_retrieval_confidence  → medium     ← chunks weren't relevant