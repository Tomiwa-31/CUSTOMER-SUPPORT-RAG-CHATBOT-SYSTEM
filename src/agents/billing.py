# src/agents/billing.py

import pickle
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    CHROMA_DIR,
    BILLING_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    SEARCH_TYPE,
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    ENSEMBLE_WEIGHTS,
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    BILLING_PROMPT,
)
from src.rewriter import build_query_rewriter
from src.ingestion import get_bm25_path


def build_billing_agent():

    # ── Embedding model ────────────────────────────────
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )

    # ── ChromaDB retriever ─────────────────────────────
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=BILLING_COLLECTION
    )
    semantic_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": SEMANTIC_TOP_K}
    )

    # ── BM25 retriever ─────────────────────────────────
    bm25_path = get_bm25_path(BILLING_COLLECTION)
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = BM25_TOP_K

    # ── Ensemble retriever ─────────────────────────────
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=ENSEMBLE_WEIGHTS
    )

    # ── Query rewriter ─────────────────────────────────
    rewriter = build_query_rewriter()

    # ── LLM ───────────────────────────────────────────
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )

    # ── Prompt ─────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", BILLING_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # ── Format docs ────────────────────────────────────
    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('Header2', 'General')}]\n{doc.page_content}"
            for doc in docs
        )

    # ── Rewrite and retrieve ───────────────────────────
    def rewrite_and_retrieve(inputs):
        question = inputs["question"]
        logger = inputs.get("logger")

        rewritten = rewriter.invoke({"question": question})
        if logger:
            logger.info("billing_query_rewritten",
                original_query=question,
                rewritten_query=rewritten,
                agent="billing"
            )

        docs = retriever.invoke(rewritten)
        if logger:
            logger.info("billing_chunks_retrieved",
                chunk_count=len(docs),
                agent="billing"
            )

        return format_docs(docs)

    # ── RAG chain ──────────────────────────────────────
    agent_chain = (
        {
            "context": RunnableLambda(rewrite_and_retrieve),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"✅ Billing agent ready — collection: {BILLING_COLLECTION}")
    return agent_chain


if __name__ == "__main__":
    agent = build_billing_agent()

    test_queries = [
        "how do i apply a promo code",
        "how many loyalty points do i earn per dollar",
        "can i cancel my order",
    ]

    print("\n--- Billing Agent Tests ---")
    for query in test_queries:
        print(f"\nCustomer: {query}")
        response = agent.invoke({
            "question": query,
            "chat_history": [],
        })
        print(f"Agent: {response}")