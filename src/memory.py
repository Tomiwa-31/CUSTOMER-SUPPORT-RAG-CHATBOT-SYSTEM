from google.cloud import firestore
from langchain_core.messages import HumanMessage, AIMessage
from src.config import COLLECTION_NAME
import json

# Firestore collection name for chat history
MEMORY_COLLECTION = "chat_sessions"

def get_firestore_client():
    """Get Firestore client — uses Cloud Run service account automatically"""
    return firestore.Client()

def load_session_history(session_id: str) -> list:
    """Load conversation history for a session from Firestore"""
    try:
        db = get_firestore_client()
        doc_ref = db.collection(MEMORY_COLLECTION).document(session_id)
        doc = doc_ref.get()

        if not doc.exists:
            print(f" No history found for session {session_id} — starting fresh")
            return []

        data = doc.to_dict()#Firestore storage ≈ JSON-like,   Python code needs = dict

        messages = data.get("messages", [])

        # Convert stored dicts back to LangChain message objects
        history = []
        for msg in messages:
            if msg["type"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                history.append(AIMessage(content=msg["content"]))

        print(f" Loaded {len(history)} messages for session {session_id}")
        return history

    except Exception as e:
        print(f" Failed to load session history: {e}")
        return []


def save_session_history(session_id: str, chat_history: list):
    """Save conversation history for a session to Firestore"""
    try:
        db = get_firestore_client()
        doc_ref = db.collection(MEMORY_COLLECTION).document(session_id)

        # Convert LangChain message objects to plain dicts for Firestore
        messages = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"type": "ai", "content": msg.content})

        doc_ref.set({
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages),
            "last_updated": firestore.SERVER_TIMESTAMP
        })

        print(f"Saved {len(messages)} messages for session {session_id}")

    except Exception as e:
        print(f"⚠️ Failed to save session history: {e}")


def delete_session_history(session_id: str):
    """Delete conversation history for a session"""
    try:
        db = get_firestore_client()
        db.collection(MEMORY_COLLECTION).document(session_id).delete()
        print(f" Deleted history for session {session_id}")
    except Exception as e:
        print(f" Failed to delete session history: {e}")


#load_session_history()`** — called at the start of every request:

#session_id → Firestore → returns list of HumanMessage/AIMessage


#**`save_session_history()`** — called after every response:

#session_id + chat_history → converts to dicts → saves to Firestore