 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.retriever import get_retriever
from src.rewriter import build_query_rewriter
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_HISTORY_LENGTH,
)
from src.logger import RequestLogger
from src.tracer import tracer
# ── L4 CHANGE: import Firestore memory functions ──────────────────────────────
from src.memory import load_session_history, save_session_history
# ─────────────────────────────────────────────────────────────────────────────
 
 
def load_llm():
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )
    print(f"✅ LLM ready — {LLM_MODEL}")
    return llm
 
 
def load_prompt():
    template = """You are a helpful customer support agent for NovaBuy, an ecommerce store.
Answer the customer's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information \
to answer that, let me escalate this to a human agent."
Do not make up answers.
 
Context:
{context}
 
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    print("✅ Prompt template ready")
    return prompt
 
 
def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('Header2', 'General')}]\n{doc.page_content}"
        for doc in docs
    )
 
 
def build_rag_chain():
    # ── UNCHANGED — chain has no idea where history comes from ────────────────
    retriever = get_retriever()
    rewriter = build_query_rewriter()
    llm = load_llm()
    prompt = load_prompt()
 
    def rewrite_and_retrieve(inputs):
        with tracer.start_as_current_span("rewrite_and_retrieve"):
            logger = inputs.get("logger")
            question = inputs["question"]
            
            with tracer.start_as_current_span("query_rewriting"):
                rewritten = rewriter.invoke({"question": question})
            
                if logger:
                    logger.info("query_rewritten",
                    original_query=question,
                    rewritten_query=rewritten
                )
 
            with tracer.start_as_current_span("hybrid_retrieval"):
                docs = retriever.invoke(rewritten)
                if logger:
                    logger.info("chunks_retrieved",
                    chunk_count=len(docs),
                    sections=[d.metadata.get("Header2", "General") for d in docs]
                )
 
            return format_docs(docs)
 
    rag_chain = (
        {
            "context": RunnableLambda(rewrite_and_retrieve),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
 
    print("✅ RAG chain ready")
    return rag_chain
    # ─────────────────────────────────────────────────────────────────────────
 
 
def trim_history(chat_history: list, max_length: int) -> list:
    # ── UNCHANGED ─────────────────────────────────────────────────────────────
    """Keep only the last max_length messages"""
    return chat_history[-max_length:]
 
 
# ── L4 CHANGE: new wrapper — this is the only function app.py should call ────
def run_with_memory(
    chain,
    session_id: str,
    question: str,
    logger: RequestLogger = None,
) -> str:
    """
    Full request lifecycle with Firestore memory:
      1. Load history from Firestore
      2. Invoke chain with history
      3. Append new turn
      4. Trim to MAX_HISTORY_LENGTH
      5. Save updated history back to Firestore
      6. Return response string
    """
 
    # 1 — load from Firestore (returns [] for new sessions)
    chat_history = load_session_history(session_id)
 
    if logger:
        logger.info("session_loaded",
            session_id=session_id,
            history_length=len(chat_history),
        )
 
    # 2 — invoke chain (chain is unaware of Firestore — just receives a list)
    response = chain.invoke({
        "question": question,
        "chat_history": trim_history(chat_history, MAX_HISTORY_LENGTH),
        "logger": logger,
    })
 
    # 3 — append the new turn
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
 
    # 4 — trim before saving so Firestore doc doesn't grow forever
    chat_history = trim_history(chat_history, MAX_HISTORY_LENGTH)
 
    # 5 — persist back to Firestore
    save_session_history(session_id, chat_history)
 
    if logger:
        logger.info("session_saved",
            session_id=session_id,
            history_length=len(chat_history),
        )
 
    # 6 — return just the string (same shape as before)
    return response
# ─────────────────────────────────────────────────────────────────────────────
 
 
if __name__ == "__main__":
    chain = build_rag_chain()
 
    # ── L4 CHANGE: use a real session_id instead of an in-memory list ─────────
    # In production this comes from the request (cookie / header / query param).
    # For local testing, hardcode one so turns accumulate across runs.
    TEST_SESSION_ID = "dev-session-001"
 
    test_queries = [
        "how long do i have 2 return somthing",
        "what about damaged items",
        "who pays the return shipping",
    ]
 
    print("\n--- L4 RAG Chain Tests (Firestore memory) ---")
    for query in test_queries:
        print(f"\nCustomer: {query}")
 
        response = run_with_memory(
            chain=chain,
            session_id=TEST_SESSION_ID,
            question=query,
        )
 
        print(f"Agent: {response}")
 
    # To verify persistence: run the script twice.
    # Second run will load the history saved by the first run.
    # ─────────────────────────────────────────────────────────────────────────
 
 
### Dependency chain is still clean and one-directional:
#
# config.py
#     ↑
# retriever.py      memory.py
#          ↑        ↑
#          chain.py
#              ↑
#          app.py  ← next file